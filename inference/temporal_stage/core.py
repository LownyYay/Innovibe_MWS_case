"""Model-free temporal task contract. Inference never reads reference annotations."""
import copy
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'legacy'))
from candidates import cues
from temporal import boundary_role, iso_day, resolve_boundary

ATTRS = frozenset({'service_state', 'ticket_status', 'tariff', 'agreement'})
SYSTEM = '''Определи календарные границы уже извлечённых фактов из телефонного разговора.
Верни только JSON: {"dates":[[индекс_факта,"YYYY-MM-DD","YYYY-MM-DD" или null],...]}. Для каждого перечисленного факта ровно одна строка в исходном порядке.
Начало не может быть null, конец null означает открытый интервал. Начало не позже конца.
service_state, tariff и ticket_status датируются началом/окончанием соответствующего состояния. agreement датируется согласованным сроком действия, а не моментом разговора о нём.
Связывай дату с тем же объектом, событием и эпизодом; учитывай отрицания, старые/новые состояния, поправки и согласие клиента. Чужая дата в другой части разговора не является датой факта.
Относительные сроки считай относительно started_at. calendar_hints — только календарная расшифровка выражений, а не готовая привязка к фактам. Неоднозначный диапазон сам по себе не определяет точный день.
Если точное начало не установлено, используй день started_at; если окончание не установлено, используй null. Не меняй содержание фактов.'''


def read_jsonl(path):
    rows = [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    index_unique(rows)
    return rows


def index_unique(rows):
    ids = [r['call_id'] for r in rows]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError('Invalid/duplicate call IDs')
    return {r['call_id']: r for r in rows}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def public_call(row):
    # Training may explicitly strip reference annotations before constructing a prompt.
    return {k: copy.deepcopy(row[k]) for k in ('call_id', 'started_at', 'turns')}


def task_indices(facts):
    return [i for i, f in enumerate(facts) if f['attribute'] in ATTRS]


def calendar_hints(row):
    _, found = cues(row)
    grouped = {}
    for hit in found:
        key = (hit['idx'], hit['expression'], hit['role'], hit['coarse'])
        grouped.setdefault(key, set()).add(hit['date'])
    result = []
    for (idx, expr, role, coarse), days in sorted(grouped.items()):
        ds = sorted(days)
        result.append({'turn': idx, 'expression': expr, 'role': role,
                       'days': ds if len(ds) <= 7 else {'first': ds[0], 'last': ds[-1], 'range_only': True},
                       'coarse': coarse})
    return result


def messages(row, facts):
    if 'facts' in row:
        raise ValueError('Public prompt input must not contain gold facts')
    # Whitelist model-visible fields: no IDs, incumbent dates, spans, confirmed or labels.
    visible = []
    for i in task_indices(facts):
        f = facts[i]
        source = f.get('source') or {}
        visible.append({'i': i, 'object': {k: f.get('object', {}).get(k) for k in ('type', 'name')},
                        'attribute': f['attribute'], 'value': f['value'],
                        'polarity': f.get('polarity', 1), 'speaker': f.get('speaker'),
                        'source': {k: source.get(k) for k in ('turn_idx', 'quote')}})
    content = {'started_at': row['started_at'], 'turns': row['turns'],
               'calendar_hints': calendar_hints(row), 'tasks': visible}
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps(content, ensure_ascii=False, separators=(',', ':'))}]


def training_target(reference_facts):
    dates = []
    for i in task_indices(reference_facts):
        f = reference_facts[i]
        start = iso_day(f['valid_from']).isoformat()
        end = iso_day(f['valid_to']).isoformat() if f.get('valid_to') is not None else None
        if end is not None and start > end:
            raise ValueError('Invalid reference interval')
        dates.append([i, start, end])
    return json.dumps({'dates': dates}, ensure_ascii=False, separators=(',', ':'))


def parse_dates(text, facts):
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {'dates'} or not isinstance(obj['dates'], list):
        raise ValueError('Expected only dates list')
    expected = task_indices(facts)
    if len(obj['dates']) != len(expected):
        raise ValueError('Wrong proposal count')
    result = {}
    for item, index in zip(obj['dates'], expected, strict=True):
        if not isinstance(item, list) or len(item) != 3 or type(item[0]) is not int or item[0] != index:
            raise ValueError('Proposal indices/order mismatch')
        _, start, end = item
        for value in (start, end):
            if value is None:
                continue
            if not isinstance(value, str) or len(value) != 10 or date.fromisoformat(value).isoformat() != value:
                raise ValueError('Expected exact ISO day')
        if start is None or (end is not None and start > end):
            raise ValueError('Invalid proposed interval')
        result[index] = {'valid_from': start, 'valid_to': end}
    return result


def same_content(raw_fact, pred_fact):
    a = {k: copy.deepcopy(v) for k, v in raw_fact.items() if k not in ('valid_from', 'valid_to')}
    b = {k: copy.deepcopy(v) for k, v in pred_fact.items() if k not in ('valid_from', 'valid_to')}
    if 'id' not in a.get('object', {}):
        b.get('object', {}).pop('id', None)
    return a == b


def protected(row, raw_fact, field):
    evidence = raw_fact.get(field)
    if evidence is None:
        return True  # raw null end is explicit open-ended contract
    if not isinstance(evidence, dict):
        raise ValueError('Malformed raw boundary')
    status = evidence.get('status')
    if status == 'implicit':
        return True
    if status == 'unresolved':
        return False
    if status != 'explicit':
        raise ValueError('Unknown raw boundary status')
    turns = {t['idx']: t['text'] for t in row['turns']}
    if len(turns) != len(row['turns']):
        raise ValueError('Duplicate turns')
    try:
        resolve_boundary(evidence, turns, iso_day(row['started_at']), boundary_role(raw_fact['attribute'], field))
        return True
    except (ValueError, OverflowError):
        # Preserve the explicit-repeat starts repaired in the previous experiment too.
        if field == 'valid_from':
            # Count/resolve all complete mentions, same role and unique local quote.
            from temporal import mentions, resolve_mention
            import re
            idx, expr = evidence.get('turn_idx'), evidence.get('expression')
            if type(idx) is int and idx in turns and isinstance(expr, str) and expr:
                text = turns[idx]
                spans = [m.span() for m in re.finditer(r'(?<!\w)' + re.escape(expr) + r'(?!\w)', text)]
                hits = {m.span(): m for m in mentions(text)}
                role = boundary_role(raw_fact['attribute'], field)
                try:
                    values = [resolve_mention(hits[s], iso_day(row['started_at']), role) for s in spans
                              if role != 'from' or (hits[s]['prefix'] or '').casefold() not in {'до', 'по', 'к'}]
                    source = raw_fact.get('source') or {}
                    quote = source.get('quote') or ''
                    qs = [m.start() for m in re.finditer(r'(?=' + re.escape(quote) + ')', text)] if quote else []
                    return (len(spans) >= 2 and len(values) == len(spans) and len(set(values)) == 1
                            and source.get('turn_idx') == idx and len(qs) == 1
                            and any(qs[0] <= a and b <= qs[0] + len(quote) for a, b in spans))
                except (KeyError, ValueError, OverflowError):
                    pass
        return False


def apply_dates(row, raw, pred, proposals):
    if 'facts' in row or not (row['call_id'] == raw['call_id'] == pred['call_id']):
        raise ValueError('Public/gold or call ID mismatch')
    if raw.get('error') or raw.get('generation_error'):
        raise ValueError('Raw error')
    rf = json.loads(raw['text'])['facts']
    if len(rf) != len(pred['facts']) or any(not same_content(a, b) for a, b in zip(rf, pred['facts'], strict=True)):
        raise ValueError('Raw/pred count, content or order mismatch')
    proposals = parse_dates(json.dumps({'dates':[[i,p['valid_from'],p['valid_to']]
                                                for i,p in proposals.items()]}),pred['facts'])
    result = copy.deepcopy(pred)
    logs = []
    for i in task_indices(pred['facts']):
        fact, original = result['facts'][i], pred['facts'][i]
        candidate = copy.deepcopy(original)
        statuses = {}
        for field in ('valid_from', 'valid_to'):
            old = original.get(field)
            if old is not None:
                iso_day(old)
            if protected(row, rf[i], field):
                statuses[field] = 'protected_raw_evidence'
                continue
            new = proposals[i][field]
            if (old[:10] if old else None) == new:
                statuses[field] = 'same_day'
            else:
                candidate[field] = new
                statuses[field] = 'changed'
        if candidate.get('valid_from') is None or (candidate.get('valid_to') is not None
                  and candidate['valid_from'][:10] > candidate['valid_to'][:10]):
            statuses = {k: 'conflict_with_protected_boundary' for k in statuses}
            candidate = copy.deepcopy(original)
        for field in ('valid_from', 'valid_to'):
            old, new = original.get(field), candidate.get(field)
            fact[field] = new
            logs.append({'call_id': row['call_id'], 'fact_index': i, 'field': field,
                         'before': old, 'after': new, 'proposed_day': proposals[i][field],
                         'reason': statuses[field], 'changed': old != new})
    # No other fact/record fields may change.
    a, b = copy.deepcopy(pred), copy.deepcopy(result)
    for x, y in zip(a['facts'], b['facts'], strict=True):
        for field in ('valid_from', 'valid_to'):
            x.pop(field, None); y.pop(field, None)
    if a != b:
        raise AssertionError('Non-temporal mutation')
    return result, logs
