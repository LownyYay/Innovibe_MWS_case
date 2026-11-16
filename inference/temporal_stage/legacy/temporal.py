"""Normalize verbatim temporal spans using the call's local calendar."""
from datetime import date, datetime, time, timedelta
import re

FIELDS = ("valid_from", "valid_to")
ISO_DAY = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
ISO_VALUE = re.compile(
    rf"{ISO_DAY}(?:[T ][0-9]{{2}}:[0-9]{{2}}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?(?:Z|[+-][0-9]{2}:[0-9]{2})?)?"
)
RELATIVE = {
    "позавчера": -2, "вчера": -1, "сегодня": 0, "завтра": 1, "послезавтра": 2,
    "вчерашнего дня": -1, "сегодняшнего дня": 0, "завтрашнего дня": 1,
}
NUMBERS = {
    "один": 1, "одну": 1, "одного": 1, "два": 2, "две": 2, "пару": 2,
    "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8,
    "девять": 9, "десять": 10,
}
WEEKDAYS = {
    word: index
    for index, forms in enumerate((
        "понедельник понедельника", "вторник вторника", "среду среды",
        "четверг четверга", "пятницу пятницы", "субботу субботы",
        "воскресенье воскресенья",
    ))
    for word in forms.split()
}
MONTHS = {
    word: index
    for index, word in enumerate(
        "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), 1
    )
}
NUMBER = r"(?:[0-9]{1,3}|" + "|".join(NUMBERS) + ")"
UNIT = r"(?:день|дня|дней|неделю|недели|недель)"
CLOCK = r"[0-9]{1,2}:[0-9]{2}(?::[0-9]{2})?"
CALENDAR = (
    rf"{ISO_DAY}|[0-9]{{1,2}}(?P<sep>[./-])[0-9]{{1,2}}"
    r"(?:(?P=sep)(?:[0-9]{4}|[0-9]{2}))?|"
    rf"[0-9]{{1,2}}\s+(?:{'|'.join(MONTHS)})(?:\s+[0-9]{{4}}(?:\s+года)?)?"
)
BODY = "|".join((
    rf"(?P<calendar>{CALENDAR})",
    rf"(?P<ahead>через\s+(?:{NUMBER}\s+)?{UNIT})",
    rf"(?P<ago>(?:{NUMBER}\s+)?{UNIT}\s+(?:тому\s+)?назад|дн(?:ей|я)\s+{NUMBER}\s+назад)",
    rf"(?P<relative>{'|'.join(RELATIVE)})",
    rf"(?P<weekday>{'|'.join(WEEKDAYS)})",
))
MENTION = re.compile(
    rf"(?<!\w)(?:(?P<prefix>начиная\s+с|с|со|до|по|к|в|во|на)\s+)?"
    rf"(?:(?P<week>прошл\w*|следующ\w*|ближайш\w*)\s+)?"
    rf"(?:(?:{BODY})(?:(?:\s+(?:в|к)\s+|[T ])(?P<clock>{CLOCK}))?"
    rf"|(?P<clock_only>{CLOCK}))(?!\w|[./:-][0-9])",
    re.IGNORECASE,
)
UNRESOLVED_CUE = re.compile(
    r"\b(?:сейчас|недавно|накануне|давно|на днях|через|назад|утром|вечером|"
    r"после|перед|в прошл\w+|в следующ\w+|на прошл\w+|на следующ\w+)\b",
    re.IGNORECASE,
)
MODIFIER = re.compile(r"\b(?:не|около|примерно|почти|или|либо)\s*$", re.IGNORECASE)


def iso_day(value):
    """Validate the written calendar day and any supplied clock or offset."""
    if not isinstance(value, str) or ISO_VALUE.fullmatch(value) is None:
        raise ValueError("invalid ISO date")
    year, month, day = map(int, value[:10].split("-"))
    result = date(year, month, day)
    if len(value) > 10:
        time(int(value[11:13]), int(value[14:16]))
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    return result


def calendar_day(expression, base):
    parts = re.split(r"[./\s-]+", expression)
    if len(parts[0]) == 4:
        return date(*map(int, parts))
    day = int(parts[0])
    month = MONTHS[parts[1]] if parts[1] in MONTHS else int(parts[1])
    year = int(parts[2]) if len(parts) > 2 else base.year
    if len(parts) > 2 and len(parts[2]) == 2:
        year += 2000
    return date(year, month, day)


def resolve_mention(match, base, role):
    """Resolve a complete regex match. Ambiguous week/month ranges have no day."""
    parts = {
        key: " ".join(value.casefold().split())
        for key, value in match.groupdict().items() if value
    }
    prefix = parts.get("prefix")
    week = parts.get("week")
    if week and "weekday" not in parts:
        raise ValueError("week modifier requires a weekday")
    if "calendar" in parts:
        day = calendar_day(parts["calendar"], base)
    elif "relative" in parts:
        day = base + timedelta(days=RELATIVE[parts["relative"]])
    elif "ahead" in parts or "ago" in parts:
        words = parts.get("ahead", parts.get("ago")).split()
        count = next(
            (int(word) if word.isdigit() else NUMBERS[word]
             for word in words if word.isdigit() or word in NUMBERS),
            1,
        )
        unit = 7 if any(word.startswith("недел") for word in words) else 1
        sign = 1 if "ahead" in parts else -1
        day = base + timedelta(days=sign * unit * count)
    elif "weekday" in parts:
        weekday = WEEKDAYS[parts["weekday"]]
        if week and week.startswith(("прошл", "следующ")):
            shift = -7 if week.startswith("прошл") else 7
            day = base + timedelta(days=weekday - base.weekday() + shift)
        elif (week or prefix in {"до", "по", "к"} or role == "deadline") and prefix not in {"с", "со", "начиная с"}:
            day = base + timedelta(days=(weekday - base.weekday()) % 7 or 7)
        else:
            day = base - timedelta(days=(base.weekday() - weekday) % 7 or 7)
    elif prefix in {"в", "во", "до", "к", "с", "со", "начиная с", "по"}:
        day = base
    else:
        raise ValueError("clock requires a temporal preposition")
    clock = parts.get("clock", parts.get("clock_only"))
    if clock:
        return datetime.combine(day, time(*map(int, clock.split(":")))).isoformat()
    return day.isoformat()


def mentions(text):
    """Yield whole temporal matches, preserving Unicode offsets in the original text."""
    for match in MENTION.finditer(text):
        if not MODIFIER.search(text[:match.start()]):
            yield match


def boundary_role(attribute, field):
    if field == "valid_to":
        return "to"
    return "deadline" if attribute == "agreement" else "from"


def resolve_boundary(evidence, turns, base, role):
    """Return an ISO value and offsets for a unique, supported literal mention."""
    if not isinstance(evidence, dict) or evidence.get("status") != "explicit":
        raise ValueError("expected explicit temporal span")
    idx, expression = evidence.get("turn_idx"), evidence.get("expression")
    if type(idx) is not int or idx not in turns or not isinstance(expression, str) or not expression:
        raise ValueError("missing expression or invalid turn_idx")
    text = turns[idx]
    # Require both a unique literal occurrence and a complete, unmodified regex match.
    occurrences = list(re.finditer(r"(?<!\w)" + re.escape(expression) + r"(?!\w)", text))
    if len(occurrences) != 1:
        raise ValueError("expression is absent or ambiguous")
    offset = occurrences[0].span()
    hit = next((match for match in mentions(text) if match.span() == offset), None)
    if hit is None:
        raise ValueError("unsupported or partial temporal expression")
    if role == "from" and (hit["prefix"] or "").casefold() in {"до", "по", "к"}:
        raise ValueError("end or deadline expression used as the start")
    value = resolve_mention(hit, base, role)
    iso_day(value)
    return value, list(offset)

