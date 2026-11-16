"""Answer extraction and grading."""
from __future__ import annotations

import re

_FINAL = re.compile(r"FINAL ANSWER\s*[:\-]\s*(.+)", re.IGNORECASE | re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def content_to_text(content) -> str:
    """AIMessage.content can be a str or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)
    return str(content)


def extract_final(text: str) -> str:
    text = text or ""
    m = _FINAL.search(text)
    ans = m.group(1) if m else (text.strip().splitlines() or [""])[-1]
    return ans.strip().strip("*`").strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("'\"`.").strip()).lower()


def is_correct(answer: str, gold: str, answer_type: str, rel_tol: float = 1e-3) -> bool:
    if not answer:
        return False
    if answer_type == "number":
        nums = _NUM.findall(answer)
        if not nums:
            return False
        g = float(gold)
        for n in nums:
            try:
                v = float(n.replace(",", ""))
            except ValueError:
                continue
            if abs(v - g) <= max(rel_tol * abs(g), 0.011):
                return True
        return False
    if answer_type == "iso_date":
        return gold in answer
    if answer_type == "weekday":
        found = [d for d in _WEEKDAYS if d in answer.lower()]
        return found == [gold.lower()]
    if answer_type == "name":
        return _norm(gold) in _norm(answer)
    # exact string (reversals, sorted words): exact after trimming quotes/space, case-insensitive
    return _norm(answer) == _norm(gold)
