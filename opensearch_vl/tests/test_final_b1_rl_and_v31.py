#!/usr/bin/env python3
"""Offline unit tests (0 GPU, 0 external network) for final-battle RL Plan B1 edits:
  1. R.advantages single-sided clamping for max_turns & fatal (keeping ordinary wrong negative).
  2. is_yesno_gold + R.combine + Roller.one() evidence_judge gating vs hit/dev parity.
  3. Policy.__init__ optimizer.pt resume overriding saved lr with new --lr (1.5e-5).
  4. Mock HTTP tests for SUMMARY_VERSION=v1 vs v3.1 (default v1 parity, v3.1 fixed payload
     immune to SUMMARY_EXTRA_BODY, pipeline thread-local question suffix stripping,
     text_search_v3_100w cache namespace isolation, all-passage failure -> Tool execution error
     without [:500] fallback, RAW_LOG_DB reuse, and V3_100W_PROMPT parity).
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import pickle
import socketserver
import sqlite3
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest import mock

OPENSEARCH_VL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPENSEARCH_VL))

try:
    import tqdm  # noqa: F401
except ImportError:
    _tqdm_mod = types.ModuleType("tqdm")
    _tqdm_mod.tqdm = lambda it, *a, **kw: it
    sys.modules["tqdm"] = _tqdm_mod

from opensearch_infer import config, pipeline, search, tool_cache  # noqa: E402
from rl import evidence_judge as EJ  # noqa: E402
from rl import reward as R  # noqa: E402
from rl import train_grpo as G  # noqa: E402


class TestAdvantagesClamping(unittest.TestCase):
    def test_rloo_clamps_max_turns_and_fatal_only(self):
        group = [
            {"reward": 1.0, "stop": "response", "fatal": False},
            {"reward": 0.0, "stop": "response", "fatal": False},
            {"reward": 0.0, "stop": "max_turns", "fatal": False},
            {"reward": 0.0, "stop": "fatal", "fatal": True},
        ]
        clamped_flags = [r["fatal"] or r["stop"] == "max_turns" for r in group]
        adv = R.advantages([r["reward"] for r in group], clamped_flags, estimator="rloo")
        self.assertGreater(adv[0], 0.0)
        self.assertLess(adv[1], 0.0)
        self.assertEqual(adv[2], 0.0)
        self.assertEqual(adv[3], 0.0)
        # Ensure r["fatal"] itself is NOT mutated for max_turns
        self.assertFalse(group[2]["fatal"])
        self.assertTrue(group[3]["fatal"])

    def test_grpo_clamps_max_turns_and_fatal_only(self):
        group = [
            {"reward": 1.0, "stop": "response", "fatal": False},
            {"reward": 0.2, "stop": "response", "fatal": False},
            {"reward": 0.0, "stop": "response", "fatal": False},
            {"reward": 0.0, "stop": "max_turns", "fatal": False},
            {"reward": 0.0, "stop": "fatal", "fatal": True},
        ]
        clamped_flags = [r["fatal"] or r["stop"] == "max_turns" for r in group]
        adv = R.advantages([r["reward"] for r in group], clamped_flags, estimator="grpo")
        self.assertGreater(adv[0], 0.0)
        self.assertLess(adv[2], 0.0)
        self.assertEqual(adv[3], 0.0)
        self.assertEqual(adv[4], 0.0)


class TestYesNoAndShortGoldEvidence(unittest.TestCase):
    def setUp(self):
        R.set_weights(0.8, 0.2)

    def test_is_yesno_gold_vs_is_short_gold(self):
        for g in ("2023", "南沙宮", "3", "42", "USA"):
            self.assertFalse(R.is_yesno_gold(g), f"{g!r} should not be yes/no")
            self.assertTrue(R.is_short_gold(g), f"{g!r} should remain short_gold")
        for g in ("Yes", "YES", "no", "是", "否", "True", "false", "", "  "):
            self.assertTrue(R.is_yesno_gold(g), f"{g!r} should be yes/no")
            self.assertTrue(R.is_short_gold(g), f"{g!r} should be short_gold")

    def _run_roller_one(self, gold: str, query_signal: str, evidence: bool, ev_return: bool = True, stop: str = "response"):
        roller = object.__new__(G.Roller)
        roller.args = argparse.Namespace(query_signal=query_signal, fatal_k=3)
        roller.runner = object()
        roller.cfg = object()
        roller.stop = threading.Event()

        turns = [
            {
                "response_text": '<think>search</think><tool_call>{"name": "text_search", "arguments": {"query": "q"}}</tool_call>',
                "tool_call": '{"name": "text_search", "arguments": {"query": "q"}}',
                "tool_output": f"Tool execution result:\nThe exact fact is {gold}.",
                "tool_error": False,
            },
            {
                "response_text": "<think>done</think><response>wrong answer</response>",
                "tool_call": None,
                "tool_output": None,
                "tool_error": False,
            },
        ]
        if stop == "max_turns":
            turns = [turns[0]]
        fake_traj = {"stop_reason": stop, "turns": turns, "fatal_turn": None, "contents": []}
        row = {
            "data_id": "q1",
            "prompt": [{"role": "user", "content": "What is the answer?"}],
            "answer": gold,
            "aliases": [],
        }

        with mock.patch.object(pipeline, "process_single_case", return_value=fake_traj), \
                mock.patch.object(G.judge, "judge_common", return_value={"acc": 0, "judge_failed": False}), \
                mock.patch.object(EJ, "judge", return_value=ev_return) as mock_ej:
            res = roller.one(row, idx=0, k=0, outdir="/tmp", evidence=evidence)
        return res, mock_ej

    def test_evidence_mode_allows_short_non_yesno_gold(self):
        for gold in ("2023", "南沙宮", "3"):
            res, mock_ej = self._run_roller_one(gold, query_signal="evidence", evidence=True, ev_return=True)
            self.assertEqual(mock_ej.call_count, 1, f"EJ.judge must be called for {gold!r}")
            self.assertTrue(res["ev_called"])
            self.assertAlmostEqual(res["query"], 1.0)
            self.assertAlmostEqual(res["reward"], 0.2)

    def test_evidence_mode_skips_yesno_gold(self):
        for gold in ("Yes", "否"):
            res, mock_ej = self._run_roller_one(gold, query_signal="evidence", evidence=True, ev_return=True)
            self.assertEqual(mock_ej.call_count, 0, f"EJ.judge must be skipped for {gold!r}")
            self.assertFalse(res["ev_called"])
            self.assertAlmostEqual(res["query"], 0.0)
            self.assertAlmostEqual(res["reward"], 0.0)

    def test_hit_and_dev_modes_keep_bprime1_short_gold_skip(self):
        for gold in ("2023", "南沙宮", "3"):
            # Dev mode (evidence=False)
            res_dev, mock_ej_dev = self._run_roller_one(gold, query_signal="evidence", evidence=False)
            self.assertEqual(mock_ej_dev.call_count, 0)
            self.assertAlmostEqual(res_dev["query"], 0.0)
            self.assertAlmostEqual(res_dev["reward"], 0.0)
            # Hit mode (query_signal="hit")
            res_hit, mock_ej_hit = self._run_roller_one(gold, query_signal="hit", evidence=True)
            self.assertEqual(mock_ej_hit.call_count, 0)
            self.assertAlmostEqual(res_hit["query"], 0.0)
            self.assertAlmostEqual(res_hit["reward"], 0.0)

    def test_max_turns_keeps_fatal_false_and_fatal_turn_none(self):
        res, mock_ej = self._run_roller_one("2023", query_signal="evidence", evidence=True, stop="max_turns")
        self.assertEqual(mock_ej.call_count, 0)
        self.assertEqual(res["stop"], "max_turns")
        self.assertFalse(res["fatal"])
        self.assertIsNone(res["fatal_turn"])
        self.assertAlmostEqual(res["reward"], 0.0)


class TestPolicyResumeLrOverride(unittest.TestCase):
    def test_resume_optimizer_overrides_saved_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            latest = Path(tmpdir) / "adapters" / "latest"
            latest.mkdir(parents=True)
            (latest / "adapter_config.json").write_text("{}", encoding="utf-8")
            opt_path = latest / "optimizer.pt"
            saved_state = {"param_groups": [{"lr": 3e-5, "betas": (0.9, 0.999)}], "state": {"step": 20}}
            with open(opt_path, "wb") as fh:
                pickle.dump(saved_state, fh)

            class FakeParam:
                requires_grad = True
                def numel(self):
                    return 16

            class FakeModel:
                def __init__(self):
                    self.config = types.SimpleNamespace(use_cache=True)
                    self.base_model = types.SimpleNamespace(model=object())
                    self._p = FakeParam()

                def gradient_checkpointing_enable(self, **kwargs):
                    pass

                def parameters(self):
                    return [self._p]

                def named_parameters(self):
                    return [("language_model.layers.0.q_proj.lora_A", self._p)]

            class FakeAdamW:
                def __init__(self, params, lr, betas, weight_decay):
                    self.param_groups = [{"lr": lr, "betas": betas, "weight_decay": weight_decay}]
                    self.loaded = False

                def load_state_dict(self, state):
                    # Mimic PyTorch AdamW.load_state_dict restoring saved lr (3e-5)!
                    self.loaded = True
                    self.param_groups = [dict(pg) for pg in state["param_groups"]]

            def _fake_load(path, map_location=None):
                with open(path, "rb") as fh:
                    return pickle.load(fh)

            fake_torch = types.ModuleType("torch")
            fake_torch.bfloat16 = "bfloat16"
            fake_torch.optim = types.SimpleNamespace(AdamW=FakeAdamW)
            fake_torch.load = _fake_load

            fake_peft = types.ModuleType("peft")
            fake_peft.LoraConfig = lambda **kw: kw
            fake_peft.PeftModel = types.SimpleNamespace(
                from_pretrained=lambda model, resume, is_trainable=True: model
            )
            fake_peft.get_peft_model = lambda model, cfg: model

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoModelForImageTextToText = types.SimpleNamespace(
                from_pretrained=lambda *a, **kw: FakeModel()
            )

            args = argparse.Namespace(
                model="/fake/model",
                run_dir=tmpdir,
                lora_r=32,
                lora_alpha=64,
                seed=42,
                lr=1.5e-5,
            )
            fake_roller = types.SimpleNamespace(runner=object())

            with mock.patch.dict(
                sys.modules,
                {"torch": fake_torch, "peft": fake_peft, "transformers": fake_transformers},
            ):
                policy = G.Policy(args, fake_roller)

            self.assertTrue(policy.opt.loaded, "optimizer.pt state_dict must be loaded on resume")
            self.assertEqual(policy.opt.param_groups[0]["lr"], 1.5e-5)


class _MockHTTPState:
    def __init__(self):
        self.lock = threading.Lock()
        self.search_calls = []
        self.summary_calls = []
        self.summary_status = 200
        self.summary_text = "Neutral factual summary from 27B."


class _ReusableTCPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class TestSummaryV1VsV31MockHTTP(unittest.TestCase):
    def setUp(self):
        self.state = _MockHTTPState()
        state = self.state

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/v1/tools/search":
                    with state.lock:
                        state.search_calls.append(body)
                    resp = {
                        "results": [
                            {
                                "title": "Test Page 1",
                                "url": "https://example.org/p1",
                                "content": "A" * 800,
                            },
                            {
                                "title": "Test Page 2",
                                "url": "https://example.org/p2",
                                "content": "B" * 800,
                            },
                        ]
                    }
                    payload = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                elif self.path == "/v1/chat/completions":
                    with state.lock:
                        state.summary_calls.append(body)
                        status = state.summary_status
                        text = state.summary_text
                    if status != 200:
                        payload = b"internal error"
                        self.send_response(status)
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                    else:
                        resp = {"choices": [{"message": {"content": text}}]}
                        payload = json.dumps(resp).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(payload)))
                        self.end_headers()
                        self.wfile.write(payload)
                else:
                    self.send_response(404)
                    self.end_headers()

        self.server = _ReusableTCPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"
        search.set_current_question("")

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        search.set_current_question("")

    def test_v3_prompt_exact_parity_with_run_infer_v3sum(self):
        ref_path = Path("/home/sar/work/ImagentRL/output/diag-v3/run_infer_v3sum.py")
        if ref_path.exists():
            src = ref_path.read_text(encoding="utf-8")
            start = src.index('V3_100W_PROMPT = """') + len('V3_100W_PROMPT = """')
            end = src.index('"""', start)
            ref_prompt = src[start:end]
            self.assertEqual(search.V3_100W_PROMPT, ref_prompt)

    def test_v1_default_and_v31_payload_and_cache_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = os.path.join(tmpdir, "cache.sqlite")
            env_patch = {
                "TOOL_CACHE_DB": cache_db,
            }
            with mock.patch.dict(os.environ, env_patch, clear=False), \
                    mock.patch.object(config, "DEEPSEEK_API_KEY", "fake-ds-key"), \
                    mock.patch.object(config, "DEEPSEEK_API_BASE", self.base_url), \
                    mock.patch.object(config, "QWEN_API_BASE", f"{self.base_url}/v1"), \
                    mock.patch.object(config, "QWEN_MODEL_NAME", "summariser"), \
                    mock.patch.object(config, "QWEN_API_KEY", ""), \
                    mock.patch.object(config, "SUMMARY_VERSION", "v1"), \
                    mock.patch.object(
                        config,
                        "SUMMARY_EXTRA_BODY",
                        {"top_p": 0.8, "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
                    ):
                # 1) Default (SUMMARY_VERSION unset) -> v1
                os.environ.pop("SUMMARY_VERSION", None)
                self.assertEqual(config.get_summary_version(), "v1")
                out_v1 = search.text_search("Mount Fuji height", top_k=2)
                self.assertTrue(out_v1.startswith("Tool execution result:"))
                self.assertEqual(len(self.state.search_calls), 1)
                self.assertEqual(len(self.state.summary_calls), 2)
                v1_payload = self.state.summary_calls[0]
                self.assertEqual(v1_payload["max_tokens"], 1024)
                self.assertEqual(v1_payload["top_p"], 0.8)  # v1 honors SUMMARY_EXTRA_BODY
                self.assertIn("Based on the following webpage content", v1_payload["messages"][0]["content"])

                # Second v1 call hits v1 cache
                out_v1_cached = search.text_search("Mount Fuji height", top_k=2)
                self.assertEqual(out_v1_cached, out_v1)
                self.assertEqual(len(self.state.search_calls), 1)

                # 2) Switch to v3.1: MUST miss v1 cache, ignore SUMMARY_EXTRA_BODY top_p=0.8,
                #    use fixed v3.1 params + V3_100W_PROMPT + cleaned question
                os.environ["SUMMARY_VERSION"] = "v3.1"
                self.assertEqual(config.get_summary_version(), "v3.1")
                search.set_current_question("What is the height in meters?")
                out_v3 = search.text_search("Mount Fuji height", top_k=2)
                self.assertTrue(out_v3.startswith("Tool execution result:"))
                self.assertEqual(len(self.state.search_calls), 2)
                self.assertEqual(len(self.state.summary_calls), 4)
                v3_payload = self.state.summary_calls[2]
                self.assertEqual(v3_payload["model"], "summariser")
                self.assertEqual(v3_payload["max_tokens"], 280)
                self.assertEqual(v3_payload["temperature"], 0.3)
                self.assertEqual(v3_payload["top_p"], 0.95)  # NOT 0.8!
                self.assertEqual(v3_payload["top_k"], 20)
                self.assertEqual(v3_payload["chat_template_kwargs"], {"enable_thinking": False})
                v3_msg = v3_payload["messages"][0]["content"]
                self.assertIn("Attribute/Unit Hint", v3_msg)
                self.assertIn("What is the height in meters?", v3_msg)

                # Same (query, top_k, question) hits v3.1 cache
                out_v3_cached = search.text_search("Mount Fuji height", top_k=2)
                self.assertEqual(out_v3_cached, out_v3)
                self.assertEqual(len(self.state.search_calls), 2)

                # Different question -> different cache key under text_search_v3_100w
                search.set_current_question("Which prefecture is it in?")
                search.text_search("Mount Fuji height", top_k=2)
                self.assertEqual(len(self.state.search_calls), 3)

                # Inspect SQLite cache table directly to verify namespaces
                con = sqlite3.connect(cache_db)
                rows = con.execute("SELECT tool, k FROM cache").fetchall()
                con.close()
                tools_seen = [r[0] for r in rows]
                self.assertEqual(tools_seen.count("text_search"), 1)
                self.assertEqual(tools_seen.count("text_search_v3_100w"), 2)
                keys_seen = {r[1] for r in rows}
                self.assertIn(
                    tool_cache.key("text_search_v3_100w", "Mount Fuji height", 2, "What is the height in meters?"),
                    keys_seen,
                )

                # Verify invalid SUMMARY_VERSION raises ValueError
                os.environ["SUMMARY_VERSION"] = "v31"
                with self.assertRaises(ValueError):
                    config.get_summary_version()
                os.environ.pop("SUMMARY_VERSION", None)

    def test_v31_all_passages_failed_returns_error_no_fallback_and_no_cache(self):
        self.state.summary_status = 500
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = os.path.join(tmpdir, "cache.sqlite")
            with mock.patch.dict(os.environ, {"TOOL_CACHE_DB": cache_db, "SUMMARY_VERSION": "v3.1"}, clear=False), \
                    mock.patch.object(config, "DEEPSEEK_API_KEY", "fake-ds-key"), \
                    mock.patch.object(config, "DEEPSEEK_API_BASE", self.base_url), \
                    mock.patch.object(config, "QWEN_API_BASE", f"{self.base_url}/v1"), \
                    mock.patch.object(config, "QWEN_MODEL_NAME", "summariser"), \
                    mock.patch.object(search.time, "sleep", return_value=None):
                search.set_current_question("How tall?")
                out = search.text_search("fail query", top_k=2)
                self.assertTrue(out.startswith("Tool execution error:"))
                self.assertNotIn("AAAA", out)  # MUST NOT silently truncate content[:500]
                # Must not be cached
                con = sqlite3.connect(cache_db)
                cnt = con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
                con.close()
                self.assertEqual(cnt, 0)

    def test_v31_raw_log_db_reuse_skips_deepseek_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = os.path.join(tmpdir, "cache.sqlite")
            raw_db = os.path.join(tmpdir, "raw.sqlite")
            con = sqlite3.connect(raw_db)
            con.execute("CREATE TABLE raw (q TEXT, top_k INTEGER, v TEXT, t REAL)")
            con.execute(
                "INSERT INTO raw VALUES (?,?,?,?)",
                (
                    "cached raw query",
                    2,
                    json.dumps([{"url": "https://r.example/1", "title": "Saved Raw", "content": "Raw content"}]),
                    0.0,
                ),
            )
            con.commit()
            con.close()
            with mock.patch.dict(
                os.environ,
                {"TOOL_CACHE_DB": cache_db, "RAW_LOG_DB": raw_db, "SUMMARY_VERSION": "v3.1"},
                clear=False,
            ), \
                    mock.patch.object(config, "DEEPSEEK_API_KEY", "fake-ds-key"), \
                    mock.patch.object(config, "DEEPSEEK_API_BASE", self.base_url), \
                    mock.patch.object(config, "QWEN_API_BASE", f"{self.base_url}/v1"), \
                    mock.patch.object(config, "QWEN_MODEL_NAME", "summariser"):
                search.set_current_question("What year?")
                out = search.text_search("cached raw query", top_k=2)
                self.assertTrue(out.startswith("Tool execution result:"))
                self.assertEqual(len(self.state.search_calls), 0, "RAW_LOG_DB hit must skip DeepSeek search")
                self.assertEqual(len(self.state.summary_calls), 1)
                self.assertIn("Saved Raw", out)

    def test_pipeline_thread_local_question_stripping_and_cleanup(self):
        captured = {}

        class FakeRunner:
            prompt_in_system = True

            def __init__(self):
                self.calls = 0

            def infer(self, contents, system_instruction, cfg):
                self.calls += 1
                if self.calls == 1:
                    return {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": '<think>search</think><tool_call>{"name": "text_search", "arguments": {"query": "test"}}</tool_call>'
                                        }
                                    ]
                                },
                                "finishReason": "STOP",
                            }
                        ]
                    }
                return {
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "<think>done</think><response>ok</response>"}]},
                            "finishReason": "STOP",
                        }
                    ]
                }

        def fake_execute_tool(*args, **kwargs):
            captured["question"] = search.get_current_question()
            return "Tool execution result:\nok", {}

        raw_prompt = (
            "Which architect designed this bridge? "
            "[HINT - for this run only] Try searching in French. "
            "[PRIVILEGED REFERENCE ANSWER] Gustave Eiffel"
        )
        row = {
            "data_id": "case_strip",
            "category": "demo",
            "data_source": "test",
            "prompt": [{"role": "user", "content": raw_prompt}],
            "images": [],
            "answer": "Gustave Eiffel",
        }
        with tempfile.TemporaryDirectory() as tmpdir, \
                mock.patch.object(pipeline.tools, "execute_tool", side_effect=fake_execute_tool):
            pipeline.process_single_case(row=row, runner=FakeRunner(), output_dir=tmpdir, case_idx=0)

        self.assertEqual(captured["question"], "Which architect designed this bridge?")
        self.assertEqual(search.get_current_question(), "")

        # Also test when [PRIVILEGED REFERENCE ANSWER appears before [HINT
        self.assertEqual(
            pipeline._clean_question("Q text [PRIVILEGED REFERENCE ANSWER: X] [HINT - for this run only] Y"),
            "Q text",
        )

    def test_image_search_cache_key_unchanged(self):
        img = b"fake-jpeg-bytes"
        expected_k = tool_cache.key("image_search", hashlib.sha256(img).hexdigest(), config.IMAGE_SEARCH_DOC_COUNT)
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_db = os.path.join(tmpdir, "cache.sqlite")
            con = sqlite3.connect(cache_db)
            con.execute("CREATE TABLE cache (k TEXT PRIMARY KEY, tool TEXT, v TEXT, t REAL)")
            con.execute(
                "INSERT INTO cache VALUES (?,?,?,?)",
                (expected_k, "image_search", 'Tool execution result:\n[{"title": "Cached Hit"}]', 0.0),
            )
            con.commit()
            con.close()
            with mock.patch.dict(os.environ, {"TOOL_CACHE_DB": cache_db, "SUMMARY_VERSION": "v3.1"}, clear=False):
                out = search.image_search_bytes(img)
                self.assertIn("Cached Hit", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
