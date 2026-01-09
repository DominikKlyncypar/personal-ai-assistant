from __future__ import annotations

import json
import os
import re
import ssl
from typing import List, Dict, Any, Optional, Tuple, Set
from urllib import request, error


def _http_post(url: str, headers: Dict[str, str], data: Dict[str, Any], timeout: int = 40) -> Dict[str, Any]:
    body = json.dumps(data).encode("utf-8")
    # Ensure we send a UA
    hdrs = {"User-Agent": "personal-ai-assistant/1.0 python-urllib", **headers}
    req = request.Request(url, data=body, headers=hdrs, method="POST")
    # Be tolerant of environments with custom SSL; allow opt-out verify
    if os.getenv("WORKER_SSL_NO_VERIFY"):
        ctx = ssl._create_unverified_context()  # type: ignore[attr-defined]
    else:
        ctx = ssl.create_default_context()
    try:
        with request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read()
            return json.loads(raw.decode("utf-8"))
    except error.HTTPError as e:
        try:
            payload = e.read().decode("utf-8")
        except Exception:
            payload = str(e)
        raise RuntimeError(f"HTTP {e.code}: {payload}")


def _truncate_text(t: str, max_chars: int = 12000) -> str:
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1000] + "\n...[truncated]...\n" + t[-1000:]


def _bullet_norm_key(text: str) -> str:
    text = text.strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _ensure_leading_capital(text: str) -> str:
    if not text:
        return text
    chars = list(text)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            chars[idx] = ch.upper()
            return "".join(chars)
    if chars:
        chars[0] = chars[0].upper()
    return "".join(chars)


def _heuristic_notes(text: str) -> str:
    # Very simple fallback: split sentences, keep informative ones, group by type
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    action_kw = ("action", "follow up", "follow-up", "todo", "to-do", "will ", "assign", "deadline", "deliver")
    decision_kw = ("decided", "agreed", "concluded", "approved", "chose", "selected")

    actions = [s for s in sentences if any(k in s.lower() for k in action_kw)]
    decisions = [s for s in sentences if any(k in s.lower() for k in decision_kw)]

    # Key points: pick medium-length sentences avoiding duplicates
    seen = set()
    key_points: List[str] = []
    for s in sentences:
        norm = re.sub(r"\W+", " ", s.lower())
        if 40 <= len(s) <= 220 and norm not in seen:
            key_points.append(s)
            seen.add(norm)
        if len(key_points) >= 12:
            break

    used_norms: Set[str] = set()

    action_norms: Set[str] = set()
    for a in actions:
        norm = _bullet_norm_key(a)
        if norm:
            action_norms.add(norm)
    decision_norms: Set[str] = set()
    for d in decisions:
        norm = _bullet_norm_key(d)
        if norm:
            decision_norms.add(norm)
    conflicting_norms = action_norms | decision_norms
    filtered_key_points: List[str] = []
    for kp in key_points:
        norm = _bullet_norm_key(kp)
        if norm and norm in conflicting_norms:
            continue
        filtered_key_points.append(kp)
    key_points = filtered_key_points

    def bullets(lines: List[str]) -> Optional[str]:
        items: List[str] = []
        for l in lines:
            norm = _bullet_norm_key(l)
            if not norm or norm in used_norms:
                continue
            used_norms.add(norm)
            items.append(f"- {_ensure_leading_capital(l.strip())}")
            if len(items) >= 15:
                break
        if not items:
            return None
        return "\n".join(items)

    # Overview: pick the first 2-3 informative sentences as a high-level summary
    overview: List[str] = []
    for s in sentences:
        if 40 <= len(s) <= 200:
            overview.append(s)
        if len(overview) >= 3:
            break

    out = []
    if overview:
        out.append("## Overview\n" + " ".join(overview[:3]))
    kp = bullets(key_points[:8])
    if kp:
        out.append("\n## Key Points\n" + kp)
    dec_lines = bullets(decisions)
    if dec_lines:
        out.append("\n## Decisions\n" + dec_lines)
    act_lines = bullets(actions)
    if act_lines:
        out.append("\n## Action Items\n" + act_lines)
    return "\n".join(out)


def summarize_with_openai(text: str, model: Optional[str] = None) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = model or os.getenv("WORKER_SUMMARY_MODEL", "gpt-4o-mini")

    sys = (
        "You are an expert meeting note-taker. Produce concise, readable notes in markdown. "
        "Synthesize the content; do not copy verbatim transcript lines."
    )
    prompt = (
        "Summarize the following meeting transcript into clean notes.\n"
        "Requirements:\n"
        "- Sections: Overview, Key Points, Decisions, Action Items, Next Steps (omit a section if empty).\n"
        "- Use short bullet points (6-10 items total across sections).\n"
        "- Deduplicate and merge repeated ideas; remove filler and timestamps.\n"
        "- Ensure every bullet starts with a capitalized word.\n"
        "- Avoid repeating the same idea across sections; merge or drop duplicates.\n"
        "- Be factual and concise; no speculation; no speaker names unless necessary.\n"
        "- Keep total length under ~350 words.\n\n"
        "Transcript:\n" + _truncate_text(text)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        res = _http_post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            data=payload,
        )
        content = (
            res.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content or None
    except Exception:
        return None


def summarize_with_groq(text: str, model: Optional[str] = None) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    # Groq uses an OpenAI-compatible Chat Completions API at a different base URL
    model = model or os.getenv("WORKER_GROQ_MODEL", "llama-3.1-70b-versatile")

    sys = (
        "You are an expert meeting note-taker. Produce concise, readable notes in markdown. "
        "Synthesize the content; do not copy verbatim transcript lines."
    )
    prompt = (
        "Summarize the following meeting transcript into clean notes.\n"
        "Requirements:\n"
        "- Sections: Overview, Key Points, Decisions, Action Items, Next Steps (omit a section if empty).\n"
        "- Use short bullet points (6-10 items total across sections).\n"
        "- Deduplicate and merge repeated ideas; remove filler and timestamps.\n"
        "- Ensure every bullet starts with a capitalized word.\n"
        "- Avoid repeating the same idea across sections; merge or drop duplicates.\n"
        "- Be factual and concise; no speculation; no speaker names unless necessary.\n"
        "- Keep total length under ~350 words.\n\n"
        "Transcript:\n" + _truncate_text(text)
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        res = _http_post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            data=payload,
        )
        content = (
            res.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content or None
    except Exception:
        return None


def _provider_and_model() -> Tuple[str, Optional[str]]:
    provider = (os.getenv("WORKER_SUMMARY_PROVIDER") or "auto").lower()
    model = None
    if provider == "openai":
        model = os.getenv("WORKER_SUMMARY_MODEL", "gpt-4o-mini")
    elif provider == "groq":
        model = os.getenv("WORKER_GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-70b-versatile"
    return provider, model


def _chat_via_provider(messages: List[Dict[str, str]], temperature: float = 0.2) -> Optional[str]:
    """Send messages to the selected provider (OpenAI/Groq). Returns content or None.

    Falls back according to WORKER_SUMMARY_PROVIDER: openai|groq|auto
    """
    provider, model_hint = _provider_and_model()

    def _openai_call() -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        model = model_hint or os.getenv("WORKER_SUMMARY_MODEL", "gpt-4o-mini")
        payload = {"model": model, "messages": messages, "temperature": temperature}
        base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        try:
            res = _http_post(
                f"{base.rstrip('/')}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                data=payload,
            )
            return (
                res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                or None
            )
        except Exception:
            return None

    def _groq_call() -> Optional[str]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        model = model_hint or os.getenv("WORKER_GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-70b-versatile"
        payload = {"model": model, "messages": messages, "temperature": temperature}
        base = os.getenv("GROQ_API_BASE") or "https://api.groq.com/openai/v1"
        try:
            res = _http_post(
                f"{base.rstrip('/')}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                data=payload,
            )
            return (
                res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                or None
            )
        except Exception:
            return None

    if provider == "openai":
        return _openai_call()
    if provider == "groq":
        return _groq_call()
    # auto
    return _openai_call() or _groq_call()


def _chunk_text(text: str, max_chars: int = 6000) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # try to break on paragraph or sentence boundary
        slice_ = text[start:end]
        cut = max(slice_.rfind("\n\n"), slice_.rfind(". "))
        if cut == -1 or end == len(text):
            cut = len(slice_)
        chunks.append(slice_[:cut].strip())
        start += cut
    return [c for c in chunks if c]


def _preclean_join(utterances: List[Dict[str, Any]], strict: bool = False) -> str:
    """Pre-clean transcript: drop timestamps and common filler, dedupe short lines.

    If strict=True, filter more aggressively (remove short/rhetorical/filler lines).
    """
    lines: List[str] = []
    seen = set()
    filler_prefix = tuple(s.lower() for s in ("well,", "and ", "so ", "um ", "uh ", "okay,", "ok,", "yeah,", "alright,"))
    for u in utterances:
        t = (u.get("text") or "").strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith(filler_prefix):
            # keep if it contains an actual action/decision keyword
            if not any(k in low for k in ("decid", "agree", "deadline", "action", "follow", "assign", "next step")):
                continue
        norm = re.sub(r"\W+", " ", low)
        if len(t) < (12 if strict else 8):
            continue
        if strict:
            # drop rhetorical/ack lines
            if low in {"agreed", "agreed.", "okay", "ok", "yeah", "yes", "no", "right", "sure"}:
                continue
            if low.endswith("?") and not any(k in low for k in ("decid", "plan", "schedule", "assign", "deliver", "why", "how")):
                continue
        if norm in seen:
            continue
        seen.add(norm)
        lines.append(t)
    return "\n".join(lines)


def _try_parse_json(s: str) -> Optional[Dict[str, Any]]:
    """Attempt to extract and parse a JSON object from the model output.
    Tries whole string first, then searches for the first {...} block.
    """
    s = s.strip()
    # Direct attempt
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Find first JSON-like block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        frag = s[start : end + 1]
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def _map_chunk_to_struct(chunk: str, strict: bool = False) -> Optional[Dict[str, Any]]:
    """Ask the model to extract structured bullets from a chunk as JSON only.
    Schema: {"points": ["..."], "decisions": ["..."], "actions": [{"item": "...", "owner": "?", "due": "?"}]}
    """
    sys = (
        "Extract structured, concise meeting notes as pure JSON. Do not include any text outside JSON."
    )
    usr = (
        "From this transcript chunk, extract: points[], decisions[], actions[].\n"
        "- points: short, rewritten key items (no quotes).\n"
        "- decisions: concise decisions made (omit filler like 'Agreed').\n"
        "- actions: objects with item, and optional owner/due if clearly stated.\n"
        + ("- Be extra strict: ignore pleasantries, rhetorical or vague lines.\n" if strict else "")
        + "Return JSON only in the form:\n"
        + '{"points":[],"decisions":[],"actions":[{"item":"","owner":"","due":""}]}\n\n'
        + f"Chunk:\n{chunk}"
    )
    out = _chat_via_provider([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], temperature=(0.1 if strict else 0.2))
    if not out:
        return None
    obj = _try_parse_json(out)
    if not obj:
        return None
    # Normalize
    pts = obj.get("points") or []
    dec = obj.get("decisions") or []
    act = obj.get("actions") or []
    # Ensure types
    pts = [str(x).strip() for x in pts if str(x).strip()]
    dec = [str(x).strip() for x in dec if str(x).strip()]
    norm_act = []
    for a in act:
        if isinstance(a, dict):
            item = str(a.get("item") or "").strip()
            if not item:
                continue
            owner = str(a.get("owner") or "").strip() or None
            due = str(a.get("due") or "").strip() or None
            norm_act.append({"item": item, "owner": owner, "due": due})
        elif isinstance(a, str) and a.strip():
            norm_act.append({"item": a.strip(), "owner": None, "due": None})
    return {"points": pts, "decisions": dec, "actions": norm_act}


def _reduce_merge_structs(items: List[Dict[str, Any]], strict: bool = False) -> Optional[Dict[str, Any]]:
    """Ask the model to merge per-chunk structures into a final, deduplicated JSON structure."""
    # Compose a compact JSON array to feed the model
    compact = json.dumps(items, ensure_ascii=False)
    sys = (
        "Merge meeting notes into a concise final JSON. Deduplicate and rewrite succinctly."
    )
    usr = (
        "Merge the following array of per-chunk notes into final JSON.\n"
        "Schema: {points:[], decisions:[], actions:[{item, owner?, due?}]}.\n"
        "- Deduplicate similar items; remove filler and rhetorical lines.\n"
        "- Prevent repeating the same idea across points, decisions, and actions; merge overlaps.\n"
        + ("- Keep total points 4–8, decisions 0–6, actions 0–8.\n" if strict else "- Keep total points 6–10, decisions 0–8, actions 0–10.\n")
        + "Return JSON only.\n\n"
        + compact
    )
    out = _chat_via_provider([
        {"role": "system", "content": sys},
        {"role": "user", "content": usr},
    ], temperature=(0.1 if strict else 0.2))
    if not out:
        return None
    return _try_parse_json(out)


def _render_notes_from_struct(data: Dict[str, Any], overview: Optional[str] = None) -> str:
    lines: List[str] = []
    if overview:
        lines.append("## Overview")
        lines.append(overview.strip())
        lines.append("")
    pts = [p for p in (data.get("points") or []) if isinstance(p, str) and p.strip()]
    dec = [d for d in (data.get("decisions") or []) if isinstance(d, str) and d.strip()]
    acts = [a for a in (data.get("actions") or []) if isinstance(a, dict)]
    used_norms: Set[str] = set()

    def append_section(title: str, entries: List[str]) -> None:
        section_lines: List[str] = []
        for entry in entries:
            norm = _bullet_norm_key(entry)
            if not norm or norm in used_norms:
                continue
            used_norms.add(norm)
            section_lines.append(f"- {_ensure_leading_capital(entry.strip())}")
        if section_lines:
            lines.append(title)
            lines.extend(section_lines)
            lines.append("")

    if pts:
        append_section("## Key Points", pts)
    if dec:
        append_section("## Decisions", dec)
    if acts:
        action_lines: List[str] = []
        for a in acts:
            item = (a.get("item") or "").strip()
            if not item:
                continue
            owner = (a.get("owner") or "").strip()
            due = (a.get("due") or "").strip()
            suffix = []
            if owner:
                suffix.append(owner)
            if due:
                suffix.append(due)
            tail = f" ({', '.join(suffix)})" if suffix else ""
            action_lines.append(f"{item}{tail}")
        if action_lines:
            append_section("## Action Items", action_lines)
    return "\n".join(lines).strip()


# --------------------------- Quality hardening ----------------------------
_STOPWORDS = {
    "the","a","an","and","or","but","to","of","in","on","for","with","is","are","was","were","be","been","being",
    "that","this","it","as","at","by","from","we","i","you","they","he","she","them","us","our","your","my","me",
    "do","does","did","will","would","should","could","can","may","might","have","has","had","let","lets","let's",
    "ok","okay","yeah","yes","no","right","sure","thank","thanks","please"
}

def _norm_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return toks

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def _build_corpus_sentences(raw: str) -> List[List[str]]:
    # Split into rough sentences and normalize
    parts = re.split(r"(?<=[.!?])\s+|\n+", raw)
    sents = []
    for p in parts:
        t = p.strip()
        if len(t) < 6:
            continue
        toks = _norm_tokens(t)
        if len(toks) >= 3:
            sents.append(toks)
    return sents

def _too_similar_to_corpus(text: str, corpus: List[List[str]], jac=0.65, contain=0.8) -> bool:
    ptoks = _norm_tokens(text)
    if len(ptoks) < 3:
        return False
    for ctoks in corpus:
        if not ctoks:
            continue
        if _jaccard(ptoks, ctoks) >= jac:
            return True
        # containment: proportion of point tokens present in corpus sentence
        inter = len(set(ptoks) & set(ctoks))
        if inter / len(set(ptoks)) >= contain:
            return True
    return False

def _postprocess_struct(data: Dict[str, Any], raw: str, strict: bool = False) -> Dict[str, Any]:
    corpus = _build_corpus_sentences(raw)
    # Filter points
    pts = [p for p in (data.get("points") or []) if isinstance(p, str) and p.strip()]
    clean_pts: List[str] = []
    for p in pts:
        if _too_similar_to_corpus(p, corpus, jac=0.6 if strict else 0.7, contain=0.75 if strict else 0.85):
            continue
        clean_pts.append(p.strip())
    # Filter decisions
    dec = [d for d in (data.get("decisions") or []) if isinstance(d, str) and d.strip()]
    clean_dec: List[str] = []
    for d in dec:
        dl = d.strip().lower().strip(".?!")
        if dl in {"agreed","ok","okay","yes","yeah","no","right","sure"}:
            continue
        if _too_similar_to_corpus(d, corpus, jac=0.55 if strict else 0.65, contain=0.75 if strict else 0.85):
            continue
        clean_dec.append(d.strip())
    # Filter actions
    acts_in = [a for a in (data.get("actions") or []) if isinstance(a, dict)]
    clean_act: List[Dict[str, Any]] = []
    seen_items = set()
    for a in acts_in:
        item = (a.get("item") or "").strip()
        if not item:
            continue
        if _too_similar_to_corpus(item, corpus, jac=0.55 if strict else 0.65, contain=0.75 if strict else 0.85):
            continue
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if key in seen_items:
            continue
        seen_items.add(key)
        out = {"item": item}
        if a.get("owner"):
            out["owner"] = a["owner"]
        if a.get("due"):
            out["due"] = a["due"]
        clean_act.append(out)

    # Cap counts stricter in strict mode
    if strict:
        clean_pts = clean_pts[:6]
        clean_dec = clean_dec[:5]
        clean_act = clean_act[:6]
    else:
        clean_pts = clean_pts[:10]
        clean_dec = clean_dec[:8]
        clean_act = clean_act[:10]

    return {"points": clean_pts, "decisions": clean_dec, "actions": clean_act}


# ------------------------------- Diagnostics ------------------------------
def summary_diagnostics() -> Dict[str, Any]:
    """Return info about summary provider configuration and a tiny probe call result."""
    provider, model_hint = _provider_and_model()
    info: Dict[str, Any] = {
        "provider": provider,
        "openai_key": bool(os.getenv("OPENAI_API_KEY")),
        "groq_key": bool(os.getenv("GROQ_API_KEY")),
        "model": model_hint,
    }
    def _probe_openai(model: str) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "OPENAI_API_KEY not set"
        payload = {"model": model, "messages": [
            {"role": "system", "content": "Respond with OK only."},
            {"role": "user", "content": "Say OK"},
        ], "temperature": 0.0}
        base = os.getenv("OPENAI_API_BASE") or "https://api.openai.com/v1"
        try:
            res = _http_post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                data=payload,
            )
            content = (
                res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                info["probe_sample"] = content[:80]
                return None
            return "OpenAI probe returned empty response"
        except Exception as e:
            return str(e)

    def _probe_groq(model: str) -> Optional[str]:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "GROQ_API_KEY not set"
        payload = {"model": model, "messages": [
            {"role": "system", "content": "Respond with OK only."},
            {"role": "user", "content": "Say OK"},
        ], "temperature": 0.0}
        base = os.getenv("GROQ_API_BASE") or "https://api.groq.com/openai/v1"
        try:
            res = _http_post(
                f"{base.rstrip('/')}/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
                data=payload,
            )
            content = (
                res.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if content:
                info["probe_sample"] = content[:80]
                return None
            return "Groq probe returned empty response"
        except Exception as e:
            return str(e)

    err = None
    if provider == "openai":
        err = _probe_openai(model_hint or os.getenv("WORKER_SUMMARY_MODEL", "gpt-4o-mini"))
    elif provider == "groq":
        err = _probe_groq(model_hint or os.getenv("WORKER_GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-70b-versatile")
    else:
        # auto: try openai then groq
        err = _probe_openai(os.getenv("WORKER_SUMMARY_MODEL", "gpt-4o-mini"))
        if err:
            err = _probe_groq(os.getenv("WORKER_GROQ_MODEL") or os.getenv("GROQ_MODEL") or "llama-3.1-70b-versatile")
    info["probe_ok"] = err is None
    if err:
        info["error"] = err
    return info


def make_notes_from_utterances(utterances: List[Dict[str, Any]], strict: bool = False) -> str:
    # Ensure chronological order (oldest → newest) so we don't bias to the tail
    try:
        utterances = sorted(utterances, key=lambda u: int(u.get("id", 0)))
    except Exception:
        pass

    # Pre-clean lines and build text
    raw = _preclean_join(utterances, strict=strict)

    # Map-Reduce style summarization to discourage copy-paste:
    # 1) Chunk transcript → per-chunk distilled bullets
    # 2) Reduce all bullets into final structured notes
    chunks = _chunk_text(raw, max_chars=6000)

    # Try provider-backed JSON extraction + reduction first
    structs: List[Dict[str, Any]] = []
    for ch in chunks:
        s = _map_chunk_to_struct(ch, strict=strict)
        if s:
            structs.append(s)

    if structs:
        merged = _reduce_merge_structs(structs, strict=strict)
        if merged:
            # Postprocess to remove overlap with transcript and junk decisions
            merged = _postprocess_struct(merged, raw, strict=strict)
            # Optional: ask for a 1–2 sentence overview
            overview = None
            try:
                js_in = json.dumps(merged, ensure_ascii=False)
                sys3 = "Write a 1–2 sentence overview (<=60 words), plain text, no quotes."
                usr3 = "Based on this final meeting notes JSON, produce the overview only (no markdown):\n" + js_in
                ov = _chat_via_provider([
                    {"role": "system", "content": sys3},
                    {"role": "user", "content": usr3},
                ], temperature=(0.2 if strict else 0.3))
                if ov:
                    overview = ov.strip().replace("\n", " ")
            except Exception:
                overview = None
            return _render_notes_from_struct(merged, overview=overview)

    # Fallback heuristic if provider unavailable or failed
    return _heuristic_notes(raw)
