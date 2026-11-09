"""Loop filter and error-recovery flags of select_sft_subset.check_structure.

    python -m pytest SFT/scripts/tests/test_select_sft_subset.py -q
"""
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "select_sft_subset", Path(__file__).resolve().parents[1] / "select_sft_subset.py")
sel = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sel)

TOOLS = json.dumps([{"name": "text_search"}])
END = {"from": "gpt", "value": "<think>y</think><response>\\boxed{A}</response>"}


def sample(*pairs):
    turns = []
    for query, obs in pairs:
        call = json.dumps({"name": "text_search", "arguments": {"query": query}})
        turns += [{"from": "gpt", "value": f"<think>x</think><tool_call>{call}</tool_call>"},
                  {"from": "observation", "value": obs}]
    return {"tools": TOOLS, "conversations": [{"from": "human", "value": "q"}, *turns, END]}


def check(*pairs):
    return sel.check_structure(sample(*pairs))


def test_repeat_after_success_is_a_loop():
    assert check(("a", "Paris is the capital"), ("a", "Paris is the capital"))[0] == "loop:repeated_tool_call"


def test_repeat_after_failure_is_allowed_and_counts_as_recovery():
    err, info = check(("a", "Tool execution error: timeout"), ("a", "Paris is the capital"))
    assert err is None and info["tool_error"] and info["recovered_after_error"]


def test_repeat_after_empty_result_is_a_loop():
    assert check(("a", "No results found."), ("a", "No results found."))[0] == "loop:repeated_tool_call"
    assert check(("a", '{"results": [], "error": null}'), ("a", "x"))[0] == "loop:repeated_tool_call"


def test_new_arguments_after_error_are_allowed():
    err, info = check(("a", '{"success": false, "error": "429"}'), ("b", "found it"))
    assert err is None and info["recovered_after_error"]


def test_error_word_inside_results_is_not_a_failure():
    err, info = check(("a", "Result 1: Common error codes in HTTP ... timed out"), ("b", "y"))
    assert err is None and not info["tool_error"]


def test_error_without_later_success_is_not_recovery():
    err, info = check(("a", "x"), ("b", "Error: upstream unavailable"))
    assert err is None and info["tool_error"] and not info["recovered_after_error"]
