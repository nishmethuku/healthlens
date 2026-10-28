import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "judged_eval", Path(__file__).resolve().parent.parent / "eval" / "judged_eval.py"
)
judged_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judged_eval)


def test_parse_grades_reads_json_inside_extra_text():
    assert judged_eval.parse_grades('Sure: {"grades": [2, 0, 1]} done', 3) == [2, 0, 1]


def test_parse_grades_rejects_wrong_length_or_values():
    assert judged_eval.parse_grades('{"grades": [2, 1]}', 3) is None
    assert judged_eval.parse_grades('{"grades": [2, 5, 1]}', 3) is None
    assert judged_eval.parse_grades("not json", 3) is None


def test_summarize_separates_strict_and_lenient():
    result = judged_eval.summarize([2, 2, 1, 0])
    assert result == {"strict": 0.5, "lenient": 0.75}
    assert judged_eval.summarize([]) == {"strict": 0.0, "lenient": 0.0}
