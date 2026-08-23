"""
tests/test_rationale_explanations.py — Stage 11: confirms every rationale_code
actually written anywhere in the codebase has a real (non-fallback)
plain-language explanation, and that compound comma-joined trails get every
code translated, in order -- not just the first one.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.rationale_explanations import RATIONALE_EXPLANATIONS, explain_code, explain_rationale_codes

REPO_ROOT = Path(__file__).resolve().parents[1]


def _codes_written_in_source() -> set[str]:
    """Same grep the module's own docstring documents having been run --
    re-derived here so this test fails loudly if a future stage adds a new
    rationale_code and forgets to add its explanation, rather than relying
    on the docstring staying accurate by hand."""
    codes: set[str] = set()
    pattern_kw = re.compile(r'rationale_code\s*=\s*"([A-Z][A-Z_]*)"')
    pattern_bare = re.compile(r'"([A-Z][A-Z_]{2,})"')
    hard_rule_files = [REPO_ROOT / "policy" / "constraints.py"]
    scan_files = list((REPO_ROOT).rglob("*.py"))
    for f in scan_files:
        if "/.venv/" in str(f) or "/__pycache__/" in str(f):
            continue
        text = f.read_text()
        codes.update(pattern_kw.findall(text))
    for f in hard_rule_files:
        text = f.read_text()
        for code in pattern_bare.findall(text):
            if code in ("EIV_MAX",):
                continue
            codes.add(code)
    codes.add("EIV_MAX")  # policy/eiv.py appends this literal, not matched by either regex above
    return codes


class TestEveryWrittenCodeHasAnExplanation:
    def test_no_fallback_placeholder_for_any_real_code(self):
        written = _codes_written_in_source()
        missing = {code for code in written if code not in RATIONALE_EXPLANATIONS}
        assert not missing, f"rationale_code(s) written in source but missing from RATIONALE_EXPLANATIONS: {missing}"

    def test_every_mapped_explanation_is_nonempty_prose(self):
        for code, explanation in RATIONALE_EXPLANATIONS.items():
            assert isinstance(explanation, str) and len(explanation) > 15, f"{code} has a suspiciously short explanation"


class TestExplainCode:
    def test_known_code_returns_its_mapped_explanation(self):
        assert explain_code("EIV_MAX") == RATIONALE_EXPLANATIONS["EIV_MAX"]

    def test_unknown_code_returns_a_visible_fallback_not_a_crash(self):
        result = explain_code("SOME_FUTURE_CODE_NOT_YET_MAPPED")
        assert "no plain-language explanation" in result
        assert "SOME_FUTURE_CODE_NOT_YET_MAPPED" in result


class TestExplainRationaleCodes:
    def test_single_code(self):
        result = explain_rationale_codes("EIV_MAX")
        assert result == [{"code": "EIV_MAX", "explanation": RATIONALE_EXPLANATIONS["EIV_MAX"]}]

    def test_compound_trail_translates_every_code_in_order(self):
        result = explain_rationale_codes("FREQUENCY_CAP,HIGH_VALUE_REQUIRES_APPROVAL,EIV_MAX")
        assert [r["code"] for r in result] == ["FREQUENCY_CAP", "HIGH_VALUE_REQUIRES_APPROVAL", "EIV_MAX"]
        assert result[0]["explanation"] == RATIONALE_EXPLANATIONS["FREQUENCY_CAP"]
        assert result[1]["explanation"] == RATIONALE_EXPLANATIONS["HIGH_VALUE_REQUIRES_APPROVAL"]
        assert result[2]["explanation"] == RATIONALE_EXPLANATIONS["EIV_MAX"]

    def test_none_or_empty_returns_empty_list(self):
        assert explain_rationale_codes(None) == []
        assert explain_rationale_codes("") == []
