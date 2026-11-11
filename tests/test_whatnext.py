"""Unit and end-to-end test suite for the WhatNext Hermes Agent plugin."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as whatnext_plugin  # noqa: E402
import clone_client  # noqa: E402
import memory_signals  # noqa: E402
import predictor  # noqa: E402
import store  # noqa: E402
from demo import playground  # noqa: E402


class _MockContext:
    profile_name = "default"
    plugin_id = "whatnext"

    def __init__(self) -> None:
        self.hooks: List[str] = []
        self.tools: List[str] = []
        self.rpcs: Dict[str, Any] = {}
        self.commands: Dict[str, Any] = {}
        self.cli_commands: List[str] = []
        self.skills: List[str] = []
        self.aux_tasks: List[str] = []

    def get_config(self, _key: str, default: Any = None) -> Any:
        return default

    def register_auxiliary_task(self, key: str, **_kwargs: Any) -> None:
        self.aux_tasks.append(key)

    def register_hook(self, name: str, _cb: Any, **_kwargs: Any) -> None:
        self.hooks.append(name)

    def register_tool(self, name: str, **_kwargs: Any) -> None:
        self.tools.append(name)

    def register_rpc(self, name: str, handler: Any) -> None:
        self.rpcs[name] = handler

    def register_command(self, name: str, handler: Any, **_kwargs: Any) -> None:
        self.commands[name] = handler

    def register_cli_command(self, name: str, **_kwargs: Any) -> None:
        self.cli_commands.append(name)

    def register_skill(self, name: str, _path: Path, **_kwargs: Any) -> None:
        self.skills.append(name)


class WhatNextTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="whatnext-test-")
        self.home = Path(self._tmp.name)
        os.environ["HERMES_HOME"] = str(self.home)
        os.environ["WHATNEXT_DATA_DIR"] = str(self.home / "plugin-data" / "whatnext")
        os.environ["WHATNEXT_STANDALONE_DEMO"] = "1"
        os.environ.pop("CLONE_API_KEY", None)

        mem_dir = self.home / "memories"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "USER.md").write_text(
            "# User\n1. Prefers terse Python and Rust prompts.\n2. Runs pytest right after edits.\n",
            encoding="utf-8",
        )
        (mem_dir / "MEMORY.md").write_text(
            "# Memory\n1. Always keep composer latency below 50ms in local fallback.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        os.environ.pop("CLONE_API_KEY", None)
        self._tmp.cleanup()

    def test_empty_chat_starter_uses_memory_and_workspace(self) -> None:
        bundle = predictor.generate_prediction_bundle(
            session_id="new",
            mode="next_prompt",
            draft="",
            messages=[],
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(bundle["mode"], "next_prompt")
        self.assertTrue(bundle["memory_used"])
        cands = bundle["candidates"]
        self.assertGreaterEqual(len(cands), 2)
        joined = " ".join(c["text"] for c in cands)
        self.assertIn("whatnext", joined.lower())

    def test_post_turn_question_commits_to_concrete_answer(self) -> None:
        msgs = [
            {"role": "user", "content": "Add SQLite persistence for predictions."},
            {
                "role": "assistant",
                "content": (
                    "Should we store predictions in:\n"
                    "1. WAL-mode SQLite under plugin-data\n"
                    "2. Plain JSON files per session?"
                ),
            },
        ]
        bundle = predictor.generate_prediction_bundle(
            session_id="sess-q",
            mode="next_prompt",
            messages=msgs,
            cwd=str(REPO_ROOT),
        )
        cands = bundle["candidates"]
        self.assertGreaterEqual(len(cands), 2)
        self.assertEqual(cands[0]["intent"], "answer")
        self.assertIn("WAL-mode SQLite", cands[0]["text"])

    def test_complete_draft_returns_non_duplicating_suffix(self) -> None:
        bundle = predictor.generate_prediction_bundle(
            session_id="sess-draft",
            mode="complete_draft",
            draft="run ",
            draft_revision=4,
            messages=[],
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(bundle["mode"], "complete_draft")
        self.assertEqual(bundle["draft_revision"], 4)
        cand = bundle["candidates"][0]
        self.assertTrue(cand["text"].lower().startswith("run "))
        self.assertTrue(len(cand["completion_suffix"]) > 3)
        self.assertEqual("run " + cand["completion_suffix"], cand["text"])

    def test_stale_revision_is_discarded(self) -> None:
        saved_newer = store.save_prediction(
            "sess-rev",
            mode="complete_draft",
            request_id="req_new",
            context_revision=2,
            draft_revision=10,
            draft_prefix="fix bug",
            candidates=[{"text": "fix bug in parser", "completion_suffix": " in parser", "intent": "debug"}],
        )
        self.assertIsNotNone(saved_newer)

        stale_attempt = store.save_prediction(
            "sess-rev",
            mode="complete_draft",
            request_id="req_old",
            context_revision=2,
            draft_revision=7,
            draft_prefix="fix",
            candidates=[{"text": "fix typo", "completion_suffix": " typo", "intent": "debug"}],
        )
        self.assertIsNone(stale_attempt)
        active = store.get_prediction("sess-rev", mode="complete_draft")
        self.assertEqual(active["request_id"], "req_new")

    def test_human_vs_suggestion_origin_isolation_and_funnel_metrics(self) -> None:
        # 1. Two genuine human turns -> learn implement -> verify transition
        store.observe_submitted_turn("sess-m", "Implement the new predictor module", explicit_origin="human")
        store.observe_submitted_turn("sess-m", "Run pytest across all tests", explicit_origin="human")

        mv = store.get_movement_profile(current_intent="implement")
        self.assertEqual(mv["human_turns_learned"], 2)
        self.assertEqual(mv["predicted_next_intents"][0]["intent"], "verify")

        # 2. Present + Accept + Edit + Submit a suggestion -> must record funnel events, NOT human_movements
        store.record_observation_event("presented", session_id="sess-m", request_id="r1", mode="next_prompt")
        store.mark_accepted_candidate(
            "sess-m",
            request_id="r1",
            mode="next_prompt",
            accepted_text="Commit the verified changes and push",
            candidate_index=0,
        )
        res = store.observe_submitted_turn("sess-m", "Commit the verified changes now")
        self.assertEqual(res["origin"], "suggestion")
        self.assertGreater(res["edit_distance"], 0.0)

        # Human movement count stays strictly 2 (no feedback pollution from accepted suggestion!)
        mv_after = store.get_movement_profile()
        self.assertEqual(mv_after["human_turns_learned"], 2)

        metrics = store.get_evaluation_metrics()
        self.assertEqual(metrics["presented"], 1)
        self.assertEqual(metrics["accepted"], 1)
        self.assertEqual(metrics["edited"], 1)
        self.assertEqual(metrics["submitted"], 1)
        self.assertEqual(metrics["acceptance_rate"], 1.0)

        # 3. Clear feedback resets all learned tables
        store.clear_feedback_memory()
        self.assertEqual(store.get_movement_profile()["human_turns_learned"], 0)
        self.assertEqual(store.get_evaluation_metrics()["presented"], 0)

    def test_secret_scrubbing_in_memory_signals(self) -> None:
        fake_token = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2"
        (self.home / "memories" / "MEMORY.md").write_text(
            f"Never leak {fake_token} in prompts.", encoding="utf-8"
        )
        bundle = memory_signals.read_user_memory_bundle(enabled=True)
        self.assertNotIn(fake_token, bundle["preferences_text"])
        self.assertIn("[REDACTED]", bundle["preferences_text"])

    def test_clone_api_wire_protocol_against_mock_server(self) -> None:
        received: List[Dict[str, Any]] = []

        class _CloneMockHandler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                ln = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(ln).decode("utf-8")) if ln else {}
                received.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                resp = {
                    "candidates": [
                        {
                            "completion": "Run pytest -q and summarize pass counts.",
                            "intent": "verify",
                            "confidence": 0.94,
                            "reason": "Clone API mock",
                        }
                    ]
                }
                raw = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        server = HTTPServer(("127.0.0.1", 0), _CloneMockHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            os.environ["CLONE_API_KEY"] = "test_clone_app_key"
            bundle = predictor.generate_prediction_bundle(
                session_id="sess-clone",
                mode="next_prompt",
                messages=[{"role": "user", "content": "Updated store.py"}, {"role": "assistant", "content": "Done."}],
                settings={"backend": "clone_api", "clone_api_base_url": f"http://127.0.0.1:{port}"},
            )
            self.assertEqual(bundle["backend"], "clone_api")
            self.assertEqual(bundle["candidates"][0]["text"], "Run pytest -q and summarize pass counts.")
            self.assertEqual(received[0]["path"], "/v1/predictions")
            self.assertEqual(received[0]["auth"], "Bearer test_clone_app_key")
            self.assertEqual(received[0]["body"]["mode"], "next_prompt")
            self.assertIn("user_preferences", received[0]["body"])
        finally:
            server.shutdown()
            server.server_close()

    def test_plugin_registration_and_slash_commands(self) -> None:
        ctx = _MockContext()
        whatnext_plugin.register(ctx)
        self.assertEqual(sorted(ctx.hooks), ["on_session_start", "post_llm_call", "pre_llm_call"])
        self.assertEqual(sorted(ctx.tools), ["whatnext_analytics", "whatnext_predict"])
        self.assertIn("predict", ctx.rpcs)
        self.assertIn("whatnext", ctx.commands)
        self.assertIn("movements", ctx.skills)

        slash = ctx.commands["whatnext"]
        out_next = slash("", session_id="slash-sess")
        self.assertIn("WhatNext Predicted Movements", out_next)

        out_comp = slash("complete run ", session_id="slash-sess")
        self.assertIn("WhatNext Draft Completion", out_comp)

        out_stats = slash("stats", session_id="slash-sess")
        self.assertIn("WhatNext Evaluation Scorecard", out_stats)

    def test_playground_http_e2e(self) -> None:
        srv = playground.create_server(host="127.0.0.1", port=0, hermes_home=str(self.home))
        port = srv.server_address[1]
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{port}"
        try:
            html = urllib.request.urlopen(f"{base}/", timeout=5).read().decode("utf-8")
            self.assertIn("WhatNext for Hermes Agent", html)

            req = urllib.request.Request(
                f"{base}/api/predict",
                data=json.dumps({"session_id": "e2e", "mode": "complete_draft", "draft": "run ", "draft_revision": 1}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            pred = json.loads(urllib.request.urlopen(req, timeout=5).read().decode("utf-8"))
            self.assertEqual(pred["mode"], "complete_draft")
            self.assertTrue(len(pred["candidates"]) >= 1)

            send_req = urllib.request.Request(
                f"{base}/api/send",
                data=json.dumps({"session_id": "e2e", "text": "Implement feature X", "origin": "human", "messages": []}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            send_res = json.loads(urllib.request.urlopen(send_req, timeout=5).read().decode("utf-8"))
            self.assertEqual(send_res["submitted"]["origin"], "human")
            self.assertTrue(len(send_res["prediction"]["candidates"]) >= 1)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
