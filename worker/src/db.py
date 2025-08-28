"""
db.py — SQLite helper functions for AI Assistant

This module provides:
  - Database path setup
  - Connection helper with row factory
  - Initialization of required tables
  - Small utility for UTC timestamps
"""

import os
import sqlite3
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone

# Path to the SQLite database file (in your repo under src/)
DB_PATH = Path(__file__).parent / "assistant.db"

def get_connection() -> sqlite3.Connection:
    """
    Open a SQLite connection to our DB file with safe defaults.
    - Enables foreign keys (even if we also delete explicitly).
    - Returns rows as tuples by default; we convert to dicts when needed.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def initialize_db() -> None:
    """
    Create tables if they don't exist, using the latest schema.
    This is idempotent and safe to call on startup.
    """
    with get_connection() as conn:
        cur = conn.cursor()

        # meetings: simple container for a session
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                title   TEXT NOT NULL,
                created TEXT DEFAULT (datetime('now'))
            );
            """
        )

        # utterances: one row per transcribed chunk
        # NOTE: includes the new columns ts_iso, confidence, filename
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS utterances (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER NOT NULL,
                ts_iso     TEXT,            -- ISO8601 timestamp (UTC) when we inserted row
                start_ms   INTEGER,         -- optional: audio segment start (ms)
                end_ms     INTEGER,         -- optional: audio segment end (ms)
                text       TEXT NOT NULL,   -- transcript text
                confidence REAL,            -- optional: 0.0..1.0
                filename   TEXT,            -- optional: source wav/segment file
                FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
            );
            """
        )

        # Helpful index for fetching recent utterances by meeting
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_utterances_meeting_id
            ON utterances(meeting_id, id DESC);
            """
        )

        conn.commit()

def new_meeting(title: str) -> int:
    """Insert a new meeting and return its ID."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO meetings (title) VALUES (?)", (title,))
        conn.commit()
        return cur.lastrowid
    
def delete_meeting(meeting_id: int, cascade: bool = True) -> Tuple[int, int]:
    """Delete a meeting by ID."""
    with get_connection() as conn:
        cur = conn.cursor()

        # Check if meeting exists
        cur.execute("SELECT COUNT(1) FROM meetings WHERE id = ?", (meeting_id,))
        if cur.fetchone()[0] == 0:
            return (0, 0)  # nothing to delete

        # Count utterances linked to this meeting
        cur.execute("SELECT COUNT(1) FROM utterances WHERE meeting_id = ?", (meeting_id,))
        utt_count = cur.fetchone()[0]

        # If not cascading and utterances exist, block deletion
        if not cascade and utt_count > 0:
            raise ValueError(
                f"Cannot delete meeting {meeting_id}: {utt_count} utterance(s) exist. "
                "Retry with cascade=True to remove them."
            )

        # Delete utterances first (if any) to avoid FK issues
        deleted_utt = 0
        if cascade and utt_count > 0:
            cur.execute("DELETE FROM utterances WHERE meeting_id = ?", (meeting_id,))
            deleted_utt = cur.rowcount

        # Delete the meeting itself
        cur.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))
        deleted_meeting = cur.rowcount

        # Commit happens automatically when exiting the context manager
        return (deleted_meeting, deleted_utt)

def insert_utterance(
    meeting_id: int,
    text: str,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    confidence: Optional[float] = None,
    filename: Optional[str] = None,
) -> int:
    """Insert a single transcribed utterance (one line of transcript)."""
    # Open a connection (context manager ensures commit/rollback and close)
    with get_connection() as conn:
        cur = conn.cursor()

        # 1) Ensure the meeting exists (avoid orphan rows)
        cur.execute("SELECT id FROM meetings WHERE id = ?", (meeting_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"Meeting {meeting_id} does not exist")

        # 2) Normalize times if provided
        s_ms = start_ms
        e_ms = end_ms
        if s_ms is not None and e_ms is not None and e_ms < s_ms:
            # Swap if someone passed them reversed
            s_ms, e_ms = e_ms, s_ms

        # 3) Clamp confidence if provided
        conf = None if confidence is None else max(0.0, min(1.0, float(confidence)))

        # 4) Insert the utterance
        #    - ts_iso: store current UTC time in ISO format
        from datetime import datetime, timezone
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

        cur.execute(
            """
            INSERT INTO utterances (
                meeting_id, ts_iso, start_ms, end_ms, text, confidence, filename
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (meeting_id, ts_iso, s_ms, e_ms, text, conf, filename),
        )
        utterance_id = cur.lastrowid

        # 5) Done (commit happens automatically when exiting the context manager)
        return utterance_id


def list_utterances_for_meeting(meeting_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    """Return recent utterances for a meeting (newest first)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, meeting_id, ts_iso, start_ms, end_ms, text, confidence, filename
            FROM utterances
            WHERE meeting_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (meeting_id, limit),
        )
        rows = cur.fetchall()

        # Convert row tuples → plain dicts (JSON-friendly)
        items: List[Dict[str, Any]] = []
        for r in rows:
            items.append(
                {
                    "id": r[0],
                    "meeting_id": r[1],
                    "ts_iso": r[2],
                    "start_ms": r[3],
                    "end_ms": r[4],
                    "text": r[5],
                    "confidence": r[6],
                    "filename": r[7],
                }
            )
        return items