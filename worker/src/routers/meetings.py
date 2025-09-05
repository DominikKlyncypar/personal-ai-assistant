from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import PlainTextResponse

from ..db import get_connection, new_meeting, delete_meeting, list_utterances_for_meeting
from ..models.meeting import MeetingMeta, Utterance, ExportJSONResponse
from ..state import State, get_state

router = APIRouter(tags=["meetings"])


def _fetch_meeting_meta(meeting_id: int) -> MeetingMeta:
    with get_connection() as conn:
        cur = conn.cursor()
        # Try to include created_at if available
        cur.execute(
            "SELECT id, title, created_at FROM meetings WHERE id = ?",
            (meeting_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")
        # Some schemas might not have created_at; account for that
        if len(row) == 3:
            return MeetingMeta(id=int(row[0]), title=row[1], created_at=row[2])
        return MeetingMeta(id=int(row[0]), title=row[1])  # type: ignore[arg-type]


def _fetch_meeting_utterances(meeting_id: int) -> List[Utterance]:
    items = list_utterances_for_meeting(meeting_id, limit=10_000)
    return [Utterance(**it) for it in items]


@router.post("/meeting/new")
def v1_meeting_new(title: str = "Untitled") -> Dict[str, Any]:
    mid = new_meeting(title)
    return {"ok": True, "meeting_id": mid, "title": title}


@router.delete("/meeting/{meeting_id}")
def v1_meeting_delete(
    meeting_id: int,
    cascade: bool = Query(default=True, description="If true, also delete all utterances for the meeting."),
) -> Dict[str, Any]:
    try:
        deleted_meeting, deleted_utt = delete_meeting(meeting_id, cascade=cascade)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if deleted_meeting == 0:
        raise HTTPException(status_code=404, detail=f"Meeting {meeting_id} not found")

    return {
        "ok": True,
        "meeting_id": meeting_id,
        "deleted_meeting": deleted_meeting,
        "deleted_utterances": deleted_utt,
    }


@router.post("/meeting/start/{meeting_id}")
def v1_meeting_start(meeting_id: int, state: State = Depends(get_state)) -> Dict[str, Any]:
    # Validate meeting exists
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM meetings WHERE id = ?", (meeting_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Meeting not found")
    state.current_meeting_id = meeting_id
    return {"ok": True, "id": meeting_id}


@router.post("/meeting/stop")
def v1_meeting_stop(state: State = Depends(get_state)) -> Dict[str, Any]:
    state.current_meeting_id = None
    return {"ok": True}


@router.get("/meeting/active")
def v1_meeting_active(state: State = Depends(get_state)) -> Dict[str, Any]:
    return {"id": state.current_meeting_id}


@router.get("/meeting/{meeting_id}/export.json", response_model=ExportJSONResponse)
def v1_export_meeting_json(
    meeting_id: int,
    speakers: str = Query("0", description="'0'|'heuristic'|'auto' (v1 currently supports '0' only)"),
    max_speakers: Optional[int] = Query(None, ge=1, le=20),
) -> ExportJSONResponse:
    if speakers not in {"0", "heuristic", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid speakers mode")
    if speakers != "0":
        # Placeholder: diarization will be added in a later step under services/
        # For now, keep parity with legacy endpoints by returning without labels.
        pass
    meta = _fetch_meeting_meta(meeting_id)
    utterances = _fetch_meeting_utterances(meeting_id)
    return ExportJSONResponse(meeting=meta, utterances=utterances)


@router.get("/meeting/{meeting_id}/export.md", response_class=PlainTextResponse)
def v1_export_meeting_markdown(
    meeting_id: int,
    speakers: str = Query("0", description="'0'|'heuristic'|'auto' (v1 currently supports '0' only)"),
    max_speakers: Optional[int] = Query(None, ge=1, le=20),
) -> str:
    if speakers not in {"0", "heuristic", "auto"}:
        raise HTTPException(status_code=400, detail="Invalid speakers mode")
    # Placeholder: diarization to be introduced later
    meta = _fetch_meeting_meta(meeting_id)
    utterances = _fetch_meeting_utterances(meeting_id)

    header_lines: List[str] = [f"# Meeting {meta.id}: {meta.title or 'Untitled'}"]
    if meta.created_at:
        header_lines.append(f"_Created: {meta.created_at}_")
    header_lines.append("")

    lines: List[str] = []
    for u in utterances:
        ts = u.ts_iso or ""
        text = u.text or ""
        if u.speaker:
            lines.append(f"- **{u.speaker}** [{ts}] {text}")
        else:
            lines.append(f"- [{ts}] {text}")
    return "\n".join(header_lines + lines) + "\n"


@router.get("/meeting/{meeting_id}/utterances")
def v1_list_utterances(
    meeting_id: int,
    since_id: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, meeting_id, ts_iso, start_ms, end_ms, text, confidence, filename
            FROM utterances
            WHERE meeting_id = ? AND id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (meeting_id, since_id, limit),
        )
        rows = cur.fetchall()
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
    next_since = items[-1]["id"] if items else since_id
    return {"ok": True, "items": items, "next_since_id": next_since}
