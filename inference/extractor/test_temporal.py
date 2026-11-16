import json
import unittest

from model_io import training_target
from temporal import iso_day
from validation import decode_prediction


def fact(start, end=None, *, value="в работе", idx=1, quote=None, attribute="ticket_status"):
    return {
        "object": {"id": "ticket:1", "name": "заявка 1"},
        "attribute": attribute, "value": value, "polarity": 1,
        "valid_from": start, "valid_to": end,
        "source": {"turn_idx": idx, "quote": quote},
    }


def call(*texts):
    return {
        "call_id": "test",
        "started_at": "2026-03-01T00:15:00+03:00",
        "turns": [{"idx": idx, "text": text} for idx, text in enumerate(texts, 1)],
    }


def span(expression, idx=1):
    return {"status": "explicit", "turn_idx": idx, "expression": expression}


class TemporalTest(unittest.TestCase):
    def test_calendar_validation(self):
        self.assertEqual(iso_day("2024-02-29").isoformat(), "2024-02-29")
        for value in ("2026-02-29", "2026-04-31T08:00:00", "2026-01-01T24:00:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                iso_day(value)

    def test_cross_turn_deadline_uses_local_day_and_explicit_clock(self):
        row = call("Договорились о выезде", "Завтра к 14:30 мастер приедет.")
        prediction, issues = decode_prediction(
            {"facts": [fact(span("Завтра к 14:30", 2), attribute="agreement")]}, row
        )
        self.assertEqual(issues, [])
        self.assertEqual(prediction["facts"][0]["valid_from"], "2026-03-02T14:30:00")

    def test_relative_and_absolute_dates(self):
        for expression, expected in (
            ("с пятницы", "2026-02-27"),
            ("дней пять назад", "2026-02-24"),
            ("через две недели", "2026-03-15"),
            ("28 февраля", "2026-02-28"),
            ("28.02.2026", "2026-02-28"),
        ):
            with self.subTest(expression=expression):
                row = call(expression)
                prediction, issues = decode_prediction({"facts": [fact(span(expression))]}, row)
                self.assertEqual(issues, [])
                self.assertEqual(prediction["facts"][0]["valid_from"], expected)

    def test_invalid_or_fabricated_span_keeps_fact(self):
        for text, evidence in (
            ("31.02.2026", span("31.02.2026")),
            ("вчера", span("позавчера")),
            ("с пятницы", span("пятницы")),
            ("вчера, именно вчера", span("вчера")),
            ("до завтра", span("до завтра")),
        ):
            with self.subTest(text=text, evidence=evidence):
                prediction, issues = decode_prediction({"facts": [fact(evidence)]}, call(text))
                self.assertEqual(prediction["facts"][0]["valid_from"], "2026-03-01")
                self.assertEqual(issues[0]["field"], "valid_from")

    def test_legacy_bad_end_keeps_valid_start_and_other_facts(self):
        row = call("Заявка в работе.")
        prediction, issues = decode_prediction(
            {"facts": [fact("2026-02-28", "2026-02-30"), fact("2026-03-01")]}, row, legacy_iso=True
        )
        self.assertEqual(len(prediction["facts"]), 2)
        self.assertEqual(prediction["facts"][0]["valid_from"], "2026-02-28")
        self.assertIsNone(prediction["facts"][0]["valid_to"])
        self.assertEqual(issues[0]["field"], "valid_to")

    def test_target_uses_grounded_state_transition_and_preserves_gold(self):
        row = call("Заявка принята.", "Заявка со вчера в работе.")
        row["facts"] = [
            fact("2026-02-26", "2026-02-28", value="принята"),
            fact("2026-02-28", idx=2, quote="Заявка со вчера в работе."),
        ]
        target = json.loads(training_target(row))
        self.assertEqual(target["facts"][0]["valid_to"], span("со вчера", 2))
        self.assertEqual(target["facts"][1]["valid_from"], span("со вчера", 2))
        self.assertNotIn("id", target["facts"][1]["object"])
        self.assertEqual(row["facts"][0]["valid_to"], "2026-02-28")


if __name__ == "__main__":
    unittest.main()

