import type { ConversationRow, ConversationSource } from "../../api/types";

const SOURCE_LABEL: Record<ConversationSource, string> = {
  openclaw: "OpenClaw",
  hermes: "Hermes",
};

interface ConversationListProps {
  conversations: ConversationRow[];
}

function formatSessionAge(lastUsedAt: number | undefined, now: number): string {
  if (lastUsedAt == null || !Number.isFinite(lastUsedAt)) return "—";
  const delta = Math.max(0, now - lastUsedAt);
  const minute = 60_000;
  const hour = 60 * minute;
  const day = 24 * hour;
  if (delta < minute) return "now";
  if (delta < hour) return `${Math.floor(delta / minute)}m`;
  if (delta < day) return `${Math.floor(delta / hour)}h`;
  if (delta < 7 * day) return `${Math.floor(delta / day)}d`;
  return new Date(lastUsedAt).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function stateLabel(row: ConversationRow, now: number): string {
  if (row.badge === "generating") return "Generating";
  if (row.badge === "stalled") return "Stalled";
  return formatSessionAge(row.lastUsedAt, now);
}

function laneLabel(source: ConversationSource, agent?: string): string {
  const base = SOURCE_LABEL[source];
  return agent ? `${base} · ${agent}` : base;
}

function groupConversations(rows: ConversationRow[]) {
  const groups: { key: string; label: string; rows: ConversationRow[] }[] = [];
  const index = new Map<string, number>();
  for (const row of rows) {
    const key = `${row.source}\0${row.agent ?? ""}`;
    const existing = index.get(key);
    if (existing != null) {
      groups[existing].rows.push(row);
      continue;
    }
    index.set(key, groups.length);
    groups.push({ key, label: laneLabel(row.source, row.agent), rows: [row] });
  }
  return groups;
}

export function ConversationList({ conversations }: ConversationListProps) {
  if (conversations.length === 0) return null;
  const now = Date.now();
  const groups = groupConversations(conversations);

  return (
    <div className="occupancy">
      <div className="occupancy-head">
        <span className="occupancy-head-count">{conversations.length}</span>
        <span className="occupancy-head-title">Sessions</span>
      </div>
      <div className="occupancy-list" aria-label="Occupancy on this LLM">
        {groups.map((group) => (
          <section key={group.key} className="occupancy-lane">
            <h3 className="occupancy-lane-label">{group.label}</h3>
            <ul className="occupancy-lane-list" aria-label={group.label}>
              {group.rows.map((row) => (
                <li key={row.id} className={`occupancy-row is-${row.badge}`}>
                  <span className="occupancy-dot" aria-hidden="true" />
                  <span className="occupancy-handle" title={row.handle}>
                    {row.handle}
                  </span>
                  <span className="occupancy-state">{stateLabel(row, now)}</span>
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
