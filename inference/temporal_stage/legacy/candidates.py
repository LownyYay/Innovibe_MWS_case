"""Date candidates and fact-to-date features. Never reads reference facts/dates."""
from collections import defaultdict
from datetime import date, timedelta
import calendar
import math
import re

from temporal import mentions, resolve_mention, iso_day

ATTRS = ("service_state", "ticket_status", "tariff", "agreement", "action")
WORDS = ("был", "была", "были", "раньше", "прежде", "ранее", "стар", "сейчас", "теперь",
         "сегодня", "вчера", "уже", "ещ", "потом", "после", "до", "принят", "работ",
         "назнач", "выполн", "закрыт", "отмен", "переш", "смен", "соглас", "завтра")
COARSE = re.compile(r"\b(?:на\s+(?:прошлой|следующей)\s+неделе|в\s+(?:прошлом|следующем)\s+месяце|месяц\s+назад|через\s+месяц|на\s+днях)\b", re.I)


def tokens(text):
    return {w[:5] for w in re.findall(r"[а-яёa-z0-9]+", (text or "").lower().replace("ё", "е")) if len(w) > 2}


def overlap(a, b):
    return len(a & b) / max(1, min(len(a), len(b)))


def cues(row):
    base = iso_day(row["started_at"])
    found = []
    for turn in row["turns"]:
        text = turn["text"]
        for hit in mentions(text):
            for role in ("from", "deadline"):
                try:
                    value = resolve_mention(hit, base, role)
                except (ValueError, OverflowError):
                    continue
                item = {"date": value[:10], "iso": value, "idx": turn["idx"],
                        "start": hit.start(), "stop": hit.end(), "expression": hit.group(),
                        "prefix": (hit.groupdict().get("prefix") or "").lower(),
                        "role": role, "coarse": False, "text": text,
                        "speaker": turn.get("speaker", "")}
                found.append(item)
        for hit in COARSE.finditer(text):
            expression = hit.group().lower()
            if "неделе" in expression:
                start = base - timedelta(days=base.weekday() + 7) if "прошлой" in expression else base + timedelta(days=7 - base.weekday())
                days = [start + timedelta(days=i) for i in range(7)]
            elif "месяце" in expression:
                month = base.month + (-1 if "прошлом" in expression else 1)
                year = base.year + (month - 1) // 12
                month = (month - 1) % 12 + 1
                days = [date(year, month, d) for d in range(1, calendar.monthrange(year, month)[1] + 1)]
            elif "месяц" in expression:
                sign = -1 if "назад" in expression else 1
                days = [base + timedelta(days=sign * d) for d in range(28, 32)]
            else:
                days = [base - timedelta(days=d) for d in range(2, 6)]
            for day in days:
                found.append({"date": day.isoformat(), "iso": day.isoformat(), "idx": turn["idx"],
                              "start": hit.start(), "stop": hit.end(), "expression": hit.group(),
                              "prefix": "", "role": "range", "coarse": True,
                              "text": text, "speaker": turn.get("speaker", "")})
    # Preserve duplicate occurrences as evidence; deduplicate equivalent rule resolutions.
    return base, list({(x["date"], x["idx"], x["start"], x["stop"], x["role"]): x for x in found}.values())


def sentence(text, start, stop):
    left = max([text.rfind(s, 0, start) for s in ".!?;\n"] + [-1]) + 1
    positions = [text.find(s, stop) for s in ".!?;\n"]
    right = min([p for p in positions if p >= 0] + [len(text)])
    return text[left:right]


def options(row, fact, field, all_facts, found=None):
    """Return candidate days and numeric features using public dialogue + predicted facts."""
    base, hits = found if found is not None else cues(row)
    grouped = defaultdict(list)
    for hit in hits:
        grouped[hit["date"]].append(hit)
    grouped.setdefault(base.isoformat(), [])
    if field == "valid_to":
        grouped[None] = []
    source = fact.get("source") or {}
    idx = source.get("turn_idx", -999)
    text = next((t["text"] for t in row["turns"] if t["idx"] == idx), "")
    quote = source.get("quote") or ""
    qstart = text.find(quote) if quote else -1
    qstop = qstart + len(quote) if qstart >= 0 else -1
    val = tokens(fact.get("value", ""))
    obj = tokens(fact.get("object", {}).get("name", ""))
    qt = tokens(quote)
    low = (quote + " " + text).lower()
    value_low = fact.get("value", "").lower()
    same = [f for f in all_facts if f.get("attribute") == fact.get("attribute")
            and (f.get("object", {}).get("name") or "") == (fact.get("object", {}).get("name") or "")]
    anchors = sorted({(f.get("source") or {}).get("turn_idx", idx) for f in same})
    next_idx = min([a for a in anchors if a > idx] + [999])
    prev_idx = max([a for a in anchors if a < idx] + [-999])
    days = sorted(d for d in grouped if d is not None)
    result = []
    for day, evidence in grouped.items():
        offset = (date.fromisoformat(day) - base).days if day is not None else 0
        x = [float(fact.get("attribute") == a) for a in ATTRS]
        x += [float(field == "valid_to"), float(day is None), float(day == base.isoformat()),
              max(-365, min(365, offset)) / 31, math.copysign(math.log1p(abs(offset)), offset),
              float(offset < 0), float(offset > 0), float(offset == 0),
              float(day == (days[0] if days else None)), float(day == (days[-1] if days else None)),
              float(fact.get("polarity", 1) > 0), min(len(same), 8),
              float(idx == min(anchors or [idx])), float(idx == max(anchors or [idx]))]
        x += [float(w in low) for w in WORDS]
        x += [float(w in quote.lower()) for w in WORDS]
        x += [float(w in value_low) for w in WORDS]
        if evidence:
            local = [h for h in evidence if h["idx"] == idx]
            in_quote = [h for h in local if qstart >= 0 and h["start"] >= qstart and h["stop"] <= qstop]
            distance = min(abs(h["idx"] - idx) for h in evidence)
            distances = [min(abs(h["start"] - qstop), abs(h["stop"] - qstart)) for h in local] if qstart >= 0 else []
            contexts = [tokens(sentence(h["text"], h["start"], h["stop"])) for h in evidence]
            nearest = min(evidence, key=lambda h: (abs(h["idx"] - idx), abs(h["start"] - max(qstart, 0))))
            prefix = nearest["prefix"]
            x += [len(evidence), len({h["idx"] for h in evidence}), float(bool(local)), float(bool(in_quote)),
                  min(distance, 20), 1 / (1 + distance),
                  max(-20, min(20, nearest["idx"] - idx)),
                  min(distances + [1000]) / 100,
                  float(bool(local) and qstart >= 0 and nearest["stop"] <= qstart),
                  max(overlap(val, c) for c in contexts), max(overlap(obj, c) for c in contexts),
                  max(overlap(qt, c) for c in contexts),
                  min(abs(h["idx"] - next_idx) for h in evidence),
                  min(abs(h["idx"] - prev_idx) for h in evidence),
                  float(nearest["speaker"] == "operator"), float(nearest["coarse"]),
                  float(prefix in {"с", "со", "начиная с"}), float(prefix in {"до", "по", "к"}),
                  float(nearest["role"] == "deadline"),
                  float(any(h["idx"] == next_idx for h in evidence)),
                  float(any(h["idx"] == prev_idx for h in evidence))]
            e_low = nearest["expression"].lower()
            x += [float(w in e_low) for w in ("вчера", "сегодня", "завтра", "назад", "через", "недел", "месяц", "прошл", "следующ")]
            context = sentence(nearest["text"], nearest["start"], nearest["stop"]).lower()
            window = nearest["text"][max(0, nearest["start"]-70):nearest["stop"]+70].lower()
            x += [float(w in context) for w in WORDS]
            x += [float(w in window) for w in WORDS]
            x += [overlap(val, tokens(window)), overlap(obj, tokens(window)), overlap(qt, tokens(window))]
        else:
            x += [0.0] * (30 + 2 * len(WORDS) + 3)
        result.append((day, x, evidence))
    assert len({len(x) for _, x, _ in result}) == 1
    return result
