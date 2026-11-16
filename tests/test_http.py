import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from service import ThreadingHTTPServer, handler_for


class HttpTests(unittest.TestCase):
    def test_readiness_remains_available_during_extraction(self):
        started = threading.Event()
        finish = threading.Event()
        row = {
            "call_id": "demo", "started_at": "2026-11-16T09:00:00",
            "turns": [{"idx": 0, "speaker": "operator", "text": "Заявка в работе.", "labels": ["private"]}],
            "facts": [{"label": "gold"}],
        }

        class Pipeline:
            variant = "A_official_train_val_r32"
            mode = "full"

            def __call__(self, call):
                started.set()
                if not finish.wait(5):
                    raise RuntimeError("Readiness was blocked by inference")
                if set(call) != {"call_id", "started_at", "turns"} or "labels" in call["turns"][0]:
                    raise ValueError("Unexpected private fields")
                return {"call_id": call["call_id"], "facts": [{
                    "object": {"name": "заявка"}, "attribute": "ticket_status", "value": "в работе",
                    "polarity": 1, "valid_from": "2026-11-16", "valid_to": None, "speaker": "client",
                    "source": {"turn_idx": 0, "span": [0, 100], "quote": "выдуманная цитата"},
                }]}

        with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Pipeline())) as server:
            origin = f"http://127.0.0.1:{server.server_port}"

            def request(path, body=None):
                data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
                req = Request(origin + path, data, {"Content-Type": "application/json"})
                with urlopen(req, timeout=6) as response:
                    return json.load(response), response.headers.get("X-Evidence-Issue-Count")

            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            result = []
            worker = threading.Thread(target=lambda: result.append(request("/v1/extract", row)))
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                ready, _ = request("/readyz")
                self.assertEqual(ready["variant"], "A_official_train_val_r32")
                finish.set()
                worker.join(5)
                self.assertEqual(len(result), 1)
                output, warnings = result[0]
                self.assertEqual(warnings, "3")
                self.assertEqual(output["facts"][0]["speaker"], "operator")
                self.assertEqual(output["facts"][0]["source"], {"turn_idx": 0, "span": None, "quote": None})
                with self.assertRaises(HTTPError) as error:
                    request("/extract", {"turns": []})
                self.assertEqual(error.exception.code, 422)
            finally:
                finish.set()
                worker.join(6)
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
