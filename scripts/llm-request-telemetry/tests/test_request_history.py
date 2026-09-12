import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('history',Path(__file__).resolve().parents[1]/'request_history.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class HistoryTests(unittest.TestCase):
    def test_retains_twenty_nine_days_and_excludes_thirty_one(self):
        with tempfile.TemporaryDirectory() as directory:
            db=m.connect(Path(directory)/'history.sqlite3')
            now=4000000
            for index, days in enumerate((29,31)):
                row={'id':str(index)*32,'started_at':now-days*86400,'profile':'kv24', 'effective_temperature':0,'prompt':'PRIVATE'}
                with db:m.ingest(db,'LLM_REQUEST '+json.dumps(row),now,'deployment')
            rows=[json.loads(r[0]) for r in db.execute('SELECT data FROM requests')]
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['profile'],'kv24')
            self.assertNotIn('prompt',rows[0])
            db.close()

    def test_model_and_deployment_survive_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            db=m.connect(Path(directory)/'history.sqlite3')
            for index, model in enumerate(('qwen', 'next-model')):
                row={'id':str(index)*32,'started_at':1000000,'model':model}
                with db:m.ingest(db,'LLM_REQUEST '+json.dumps(row),1000001,'deployment-'+str(index))
            rows=[json.loads(r[0]) for r in db.execute('SELECT data FROM requests')]
            self.assertEqual({r['model'] for r in rows},{'qwen','next-model'})
            self.assertEqual({r['deployment'] for r in rows},{'deployment-0','deployment-1'})
            db.close()

    def test_normal_stream_close_is_not_an_interruption(self):
        self.assertFalse(m.interrupted({'disconnected': True, 'finish_reasons': ['stop']}))
        self.assertFalse(m.interrupted({'disconnected': True, 'finish_reasons': ['tool_calls']}))
        self.assertTrue(m.interrupted({'disconnected': True, 'finish_reasons': []}))
        self.assertTrue(m.interrupted({'interrupted': True, 'finish_reasons': ['stop']}))

    def test_restart_overlap_retention_and_field_allowlist(self):
        with tempfile.TemporaryDirectory() as directory:
            db=m.connect(Path(directory)/'history.sqlite3')
            row={'id':'a'*32,'started_at':1000000,'client':'abc','prompt':'PRIVATE'}
            line='INFO QWEN_REQUEST '+json.dumps(row)
            with db:
                m.ingest(db,line,1000001)
                m.ingest(db,line,1000001)
            records=db.execute('SELECT data FROM requests').fetchall()
            self.assertEqual(len(records),1)
            self.assertNotIn('PRIVATE',records[0][0])
            with db: m.ingest(db,'',1000001+m.RETENTION)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM requests').fetchone()[0],0)
            db.close()
