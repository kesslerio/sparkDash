#!/usr/bin/env python3
"""Retain allowlisted request metadata across inference container restarts."""
import argparse
import json
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import time

FIELDS = {'model','deployment','event','id','started_at','client','status','max_tokens','thinking',
          'reasoning_effort','prompt_tokens','completion_tokens','cached_tokens',
          'time_to_first_token_ms','generation_time_ms','queue_time_ms','mean_itl_ms',
          'tokens_per_second','decode_tokens_per_second','end_to_end_tokens_per_second','elapsed_ms','first_progress_ms','finish_reasons',
          'tool_call','stream_error','interrupted','disconnected',
          'disconnect_before_terminal','response_terminal_seen','response_completed',
          'request_metadata_omitted','response_metadata_omitted'}
FIELDS.update({'profile', 'profile_sha256', 'new_prompt_tokens', 'prefill_time_ms',
               'http_inflight_at_start', 'http_inflight_at_finish'})
FIELDS.update(prefix + key for prefix in ('requested_', 'effective_')
              for key in ('temperature', 'top_p', 'top_k', 'min_p'))
RETENTION = 30 * 86400


def ingest(db, text, now, deployment=None):
    count = 0
    for line in text.splitlines():
        match = re.search(r'(?:QWEN|LLM)_REQUEST(_START)? (\{.*\})$', line)
        if not match:
            continue
        try:
            row = json.loads(match[2])
            if not isinstance(row, dict) or not re.fullmatch(r'[a-f0-9]{32}', str(row.get('id',''))):
                continue
            if type(row.get('started_at')) not in (int,float) or not now-RETENTION <= row['started_at'] <= now+60:
                continue
            row = {k:v for k,v in row.items() if k in FIELDS}
            if deployment is not None:
                row['deployment'] = deployment
            phase = 'start' if match[1] else 'finish'
            db.execute('INSERT OR REPLACE INTO requests VALUES (?,?,?,?)',
                       (row['id'],phase,row['started_at'],json.dumps(row,separators=(',',':'))))
            count += 1
        except (ValueError,TypeError):
            continue
    db.execute('DELETE FROM requests WHERE started_at < ?', (now-RETENTION,))
    return count


def interrupted(row):
    # ASGI can report http.disconnect after a successful streaming response.
    # Preserve raw disconnect evidence, but do not count terminal responses as
    # interrupted solely because the client closed its connection.
    if row.get('interrupted'):
        return True
    if 'disconnect_before_terminal' in row:
        return bool(row['disconnect_before_terminal'])
    return bool(row.get('disconnected') and not row.get('finish_reasons')
                and not row.get('response_terminal_seen') and not row.get('response_completed'))


def connect(path):
    path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    db = sqlite3.connect(path)
    path.chmod(0o600)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS requests (id TEXT, phase TEXT, started_at REAL, data TEXT, PRIMARY KEY(id,phase))')
    db.execute('CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY, since REAL)')
    return db


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,default=Path.home()/'.local/state/llm-telemetry/requests.sqlite3')
    parser.add_argument('--config',type=Path,help='Trusted deployment JSON with telemetry.container and telemetry.deployment')
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--report-hours',type=float)
    args=parser.parse_args()
    db=connect(args.database)
    if args.report_hours is not None:
        if not 0 < args.report_hours <= 720:
            parser.error('report hours must be in (0, 720]')
        cutoff=time.time()-args.report_hours*3600
        rows=[json.loads(row[0]) for row in db.execute("SELECT data FROM requests WHERE phase='finish'")]
        rows=[row for row in rows if row['started_at']+(row.get('elapsed_ms') or 0)/1000 >= cutoff]
        missing=db.execute("SELECT COUNT(*) FROM requests s WHERE s.phase='start' AND s.started_at>=? AND NOT EXISTS (SELECT 1 FROM requests f WHERE f.id=s.id AND f.phase='finish')",(cutoff,)).fetchone()[0]
        report={'hours':args.report_hours,'finished_records':len(rows),'unfinished_records':missing,'clients':{},'deployments':{}}
        for row in rows:
            key=json.dumps([row.get('deployment','legacy-unknown'),row.get('model','unknown')],separators=(',',':'))
            report['deployments'][key]=report['deployments'].get(key,0)+1
        for identity in sorted({(row.get('deployment','legacy-unknown'),row.get('model','unknown'),row.get('client','unknown')) for row in rows}):
            client=json.dumps(identity,separators=(',',':'))
            group=[row for row in rows if (row.get('deployment','legacy-unknown'),row.get('model','unknown'),row.get('client','unknown'))==identity]
            entry={'requests':len(group),'interrupted':sum(interrupted(r) for r in group)}
            for key in ('prompt_tokens','completion_tokens','cached_tokens','queue_time_ms','first_progress_ms','tokens_per_second','decode_tokens_per_second','end_to_end_tokens_per_second'):
                values=[r[key] for r in group if type(r.get(key)) in (float,int)]
                entry[key]={'samples':len(values),'median':statistics.median(values) if values else None}
            report['clients'][client]=entry
        print(json.dumps(report,indent=2))
        return
    if args.config is None:
        parser.error('--config is required for collection')
    while True:
        source = json.loads(args.config.read_text())['telemetry']
        container, deployment = source['container'], source['deployment']
        if not all(isinstance(v,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}',v) for v in (container,deployment)):
            raise ValueError('Invalid telemetry container or deployment identifier')
        now=time.time()
        with db:
            db.execute("DELETE FROM requests WHERE started_at < ?",(now-RETENTION,))
        cursor=db.execute('SELECT since FROM cursor WHERE id=1').fetchone()
        since=cursor[0] if cursor else now-RETENTION
        try:
            result=subprocess.run(['docker','logs','--since',str(int(since)-2),container],capture_output=True,text=True,timeout=15)
            if result.returncode == 0:
                # Cursor uses the start of the read, so records arriving during
                # collection remain eligible. Primary keys remove overlap.
                with db:
                    ingest(db,result.stdout+'\n'+result.stderr,now,deployment)
                    db.execute('INSERT OR REPLACE INTO cursor VALUES (1,?)',(now,))
            else:
                print('Request history: container logs unavailable; retrying',flush=True)
        except subprocess.TimeoutExpired:
            print('Request history: log read timed out; retrying',flush=True)
        if args.once:
            return
        time.sleep(10)


if __name__=='__main__':
    main()
