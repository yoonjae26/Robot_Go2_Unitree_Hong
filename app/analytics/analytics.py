#!/usr/bin/env python3
"""
Operational analytics store — every voice/text command the dashboard executes
(mic, wake-word, text-command panel) is logged to a small local SQLite
database: language used, action(s) executed, tone, success/failure, and a
latency breakdown (STT / LLM / execution). Aggregated for the dashboard's
"운행 기록" (operation log) tab.

This is the piece that answers "could this be commercialized" concretely --
a real product needs to know what people actually ask it to do, in what
language, how often it fails, and where the time goes (STT vs LLM vs the
robot itself). SQLite (stdlib, no new dependency) rather than in-memory only,
so the data survives a dashboard restart.
"""

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "analytics.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS commands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    source TEXT NOT NULL,       -- 'mic' | 'text' | 'wake_word' | 'patrol_greeting'
    language TEXT NOT NULL,     -- 'ko' | 'vi' | 'ko-jeju'
    tone TEXT,
    understood INTEGER NOT NULL,
    success INTEGER NOT NULL,
    actions TEXT,               -- comma-joined action names, e.g. "move,hello"
    stt_ms REAL,
    llm_ms REAL,
    exec_ms REAL,
    error TEXT
)
"""

# Columns added for the runtime-verification layers (see runtime_verification.py).
# Applied via ALTER TABLE below rather than folded into _SCHEMA above, so an
# existing analytics.db from before this change keeps its rows instead of
# being recreated.
_NEW_COLUMNS = {
    # Which verification layer rejected the command, if any: 'schema' |
    # 'intent' | 'intent_infra_error' | 'context' | NULL (accepted, or
    # rejected for an unrelated reason such as STT/LLM failure).
    "reject_layer": "TEXT",
    "reject_reason": "TEXT",
    # Robot/environment state snapshot at verification time -- lets later
    # analysis check e.g. whether a context rejection actually coincided
    # with low battery, rather than trusting the reason string alone.
    "battery_pct": "REAL",
    "is_standing": "INTEGER",
    "obstacle_distance_m": "REAL",
    # Raw JSON text from the action-generation LLM call, for offline
    # false-positive/false-negative analysis of the verifier layers.
    "raw_llm_output": "TEXT",
    # Per-layer latency breakdown (see VerificationResult) -- distinct from
    # the existing exec_ms, which times physical execution, not verification.
    "verify_schema_ms": "REAL",
    "verify_intent_ms": "REAL",
    "verify_context_ms": "REAL",
}


class AnalyticsStore:
    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection):
        existing = {row[1] for row in conn.execute("PRAGMA table_info(commands)").fetchall()}
        for name, sqltype in _NEW_COLUMNS.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE commands ADD COLUMN {name} {sqltype}")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=5.0)

    def record(
        self,
        source: str,
        language: str = "ko",
        tone: Optional[str] = None,
        understood: bool = True,
        success: bool = True,
        actions: Optional[List[str]] = None,
        stt_ms: float = 0.0,
        llm_ms: float = 0.0,
        exec_ms: float = 0.0,
        error: Optional[str] = None,
        reject_layer: Optional[str] = None,
        reject_reason: Optional[str] = None,
        battery_pct: Optional[float] = None,
        is_standing: Optional[bool] = None,
        obstacle_distance_m: Optional[float] = None,
        raw_llm_output: Optional[str] = None,
        verify_schema_ms: float = 0.0,
        verify_intent_ms: float = 0.0,
        verify_context_ms: float = 0.0,
    ):
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO commands "
                    "(ts, source, language, tone, understood, success, actions, stt_ms, llm_ms, exec_ms, error, "
                    "reject_layer, reject_reason, battery_pct, is_standing, obstacle_distance_m, raw_llm_output, "
                    "verify_schema_ms, verify_intent_ms, verify_context_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        time.time(), source, language, tone,
                        int(understood), int(success),
                        ",".join(actions or []),
                        stt_ms, llm_ms, exec_ms, error,
                        reject_layer, reject_reason, battery_pct,
                        None if is_standing is None else int(is_standing),
                        obstacle_distance_m, raw_llm_output,
                        verify_schema_ms, verify_intent_ms, verify_context_ms,
                    ),
                )
        except Exception as e:
            logger.warning(f"Analytics: record failed: {e}")

    def summary(self, recent_limit: int = 15) -> dict:
        """Aggregate stats for the dashboard's 운행 기록 tab: totals, success
        rate, per-language counts, top actions by frequency, average latency
        per stage, and the N most recent commands."""
        try:
            with self._lock, self._connect() as conn:
                conn.row_factory = sqlite3.Row
                total = conn.execute("SELECT COUNT(*) c FROM commands").fetchone()["c"]
                if total == 0:
                    return self._empty_summary()

                success_count = conn.execute(
                    "SELECT COUNT(*) c FROM commands WHERE success = 1"
                ).fetchone()["c"]

                lang_rows = conn.execute(
                    "SELECT language, COUNT(*) c FROM commands GROUP BY language ORDER BY c DESC"
                ).fetchall()
                by_language = [{"language": r["language"], "count": r["c"]} for r in lang_rows]

                # Only the STT->LLM->execute pipeline sources have meaningful
                # latency numbers -- 'direct' (control-panel/D-pad button
                # presses) and 'patrol_greeting' always log 0 for these fields
                # (see dashboard_server.py's _direct()/start_move() and
                # auto_patrol.py's _greet_and_pause()), which would otherwise
                # drag these averages down and misrepresent actual STT/LLM speed.
                avg_row = conn.execute(
                    "SELECT AVG(stt_ms) stt, AVG(llm_ms) llm, AVG(exec_ms) ex "
                    "FROM commands WHERE success = 1 AND source IN ('mic', 'text', 'wake_word')"
                ).fetchone()

                reject_rows = conn.execute(
                    "SELECT reject_layer, COUNT(*) c FROM commands "
                    "WHERE reject_layer IS NOT NULL GROUP BY reject_layer ORDER BY c DESC"
                ).fetchall()
                rejections_by_layer = [{"layer": r["reject_layer"], "count": r["c"]} for r in reject_rows]

                verify_avg_row = conn.execute(
                    "SELECT AVG(verify_schema_ms) sc, AVG(verify_intent_ms) it, AVG(verify_context_ms) ct "
                    "FROM commands WHERE source IN ('mic', 'text', 'wake_word')"
                ).fetchone()

                action_rows = conn.execute(
                    "SELECT actions FROM commands WHERE actions != ''"
                ).fetchall()
                action_counts: dict = {}
                for row in action_rows:
                    for name in row["actions"].split(","):
                        if name:
                            action_counts[name] = action_counts.get(name, 0) + 1
                top_actions = sorted(action_counts.items(), key=lambda kv: -kv[1])[:8]

                recent_rows = conn.execute(
                    "SELECT ts, source, language, tone, understood, success, actions, "
                    "stt_ms, llm_ms, exec_ms, error FROM commands ORDER BY id DESC LIMIT ?",
                    (recent_limit,),
                ).fetchall()
                recent = [dict(r) for r in recent_rows]

                return {
                    "total": total,
                    "success_rate": round(success_count / total * 100, 1),
                    "by_language": by_language,
                    "top_actions": [{"action": a, "count": c} for a, c in top_actions],
                    "avg_stt_ms": round(avg_row["stt"] or 0, 0),
                    "avg_llm_ms": round(avg_row["llm"] or 0, 0),
                    "avg_exec_ms": round(avg_row["ex"] or 0, 0),
                    "rejections_by_layer": rejections_by_layer,
                    "avg_verify_schema_ms": round(verify_avg_row["sc"] or 0, 1),
                    "avg_verify_intent_ms": round(verify_avg_row["it"] or 0, 1),
                    "avg_verify_context_ms": round(verify_avg_row["ct"] or 0, 1),
                    "recent": recent,
                }
        except Exception as e:
            logger.warning(f"Analytics: summary failed: {e}")
            return self._empty_summary()

    @staticmethod
    def _empty_summary() -> dict:
        return {
            "total": 0, "success_rate": 0.0, "by_language": [], "top_actions": [],
            "avg_stt_ms": 0, "avg_llm_ms": 0, "avg_exec_ms": 0, "recent": [],
            "rejections_by_layer": [], "avg_verify_schema_ms": 0,
            "avg_verify_intent_ms": 0, "avg_verify_context_ms": 0,
        }


_store: Optional[AnalyticsStore] = None
_store_lock = threading.Lock()


def get_store() -> AnalyticsStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AnalyticsStore()
    return _store
