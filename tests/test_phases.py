"""Phase 1 (diagnosis) and Phase 5 (report) tests — scripted LLM, no server.

Both phases are interpretive: they may never change which strategy is applied, may
never invent a number, and may never cost a series when they fail. Those three
properties are what most of this file checks.

Run:  python -m pytest tests/test_phases.py -q
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestrator_react import phases as PH
from orchestrator_react import pipeline as PL
from orchestrator_react import pool as POOL
from orchestrator_react import prompts as P
from orchestrator_react import tools as T
from orchestrator_react.config import LLMRole, ReactConfig
from orchestrator_react.llm import LLMError, ScriptedLLM

from test_ingest_and_pool import MODELS, fake_repo  # noqa: E402
from test_orchestrator_react import make_state  # noqa: E402


def cards(config: ReactConfig | None = None):
    s = make_state(config=config or ReactConfig())
    phase2 = POOL.run_phase2(s, s.config)
    return s, T.series_profile(s), phase2["report"]


def diagnosis_json(**over) -> str:
    payload = {
        "regime": "seasonal_dominated",
        "predictability": "high",
        "combination_hint": "weighted",
        "risks": ["only three windows"],
        "narrative": "Strong yearly cycle with a mild trend; weights should pay off.",
    }
    payload.update(over)
    return json.dumps(payload)


class Broken:
    name = "broken"

    def complete(self, system, user):
        raise LLMError("connection refused")


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — deterministic variant (the control arm)
# ══════════════════════════════════════════════════════════════════════════════


def test_deterministic_diagnosis_has_the_full_schema():
    _, series, pool = cards()
    d = PH.deterministic_diagnosis(series, pool)
    assert set(d) >= {"regime", "predictability", "combination_hint", "risks", "narrative", "source"}
    assert d["source"] == "deterministic"
    assert d["regime"] in PH.REGIMES
    assert d["predictability"] in PH.PREDICTABILITY
    assert d["combination_hint"] in PH.HINTS


def test_deterministic_diagnosis_reads_the_regime():
    _, series, pool = cards()
    strong = dict(series, trend_strength=0.9, seasonal_strength=0.1)
    assert PH.deterministic_diagnosis(strong, pool)["regime"] == "trend_dominated"
    seasonal = dict(series, trend_strength=0.1, seasonal_strength=0.9)
    assert PH.deterministic_diagnosis(seasonal, pool)["regime"] == "seasonal_dominated"
    flat = dict(series, trend_strength=0.1, seasonal_strength=0.1)
    assert PH.deterministic_diagnosis(flat, pool)["regime"] == "noisy"
    both = dict(series, trend_strength=0.9, seasonal_strength=0.9)
    assert PH.deterministic_diagnosis(both, pool)["regime"] == "mixed"


def test_unstable_ranking_suggests_a_robust_combination():
    _, series, pool = cards()
    unstable = dict(pool, ranking_stability={"mean_kendall_tau": 0.1, "verdict": "unstable"})
    d = PH.deterministic_diagnosis(series, unstable)
    assert d["combination_hint"] == "robust"
    assert any("ranking" in r for r in d["risks"])


def test_three_windows_are_always_flagged_as_a_risk():
    _, series, pool = cards()
    d = PH.deterministic_diagnosis(series, pool)
    assert any("3 validation windows" in r or "high variance" in r for r in d["risks"])


def test_diagnosis_survives_empty_cards():
    d = PH.deterministic_diagnosis({}, {})
    assert d["regime"] in PH.REGIMES and d["narrative"]


def test_run_diagnosis_without_a_client_is_the_deterministic_one():
    s, series, pool = cards()
    d = PH.run_diagnosis(s, series, None, pool)
    assert d["source"] == "deterministic"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 1 — LLM variant
# ══════════════════════════════════════════════════════════════════════════════


def test_llm_diagnosis_is_parsed_and_labelled():
    s, series, pool = cards()
    d = PH.run_diagnosis(s, series, ScriptedLLM([diagnosis_json()]), pool)
    assert d["source"] == "llm"
    assert d["regime"] == "seasonal_dominated"
    assert d["combination_hint"] == "weighted"
    assert d["risks"] == ["only three windows"]
    assert "yearly cycle" in d["narrative"]


def test_llm_diagnosis_accepts_prose_around_the_json():
    s, series, pool = cards()
    wrapped = f"Here is my reading:\n```json\n{diagnosis_json()}\n```\nHope that helps."
    d = PH.run_diagnosis(s, series, ScriptedLLM([wrapped]), pool)
    assert d["source"] == "llm" and d["regime"] == "seasonal_dominated"


def test_llm_diagnosis_strips_think_blocks():
    s, series, pool = cards()
    raw = f"<think>let me look at the numbers</think>{diagnosis_json()}"
    d = PH.run_diagnosis(s, series, ScriptedLLM([raw]), pool)
    assert d["source"] == "llm"


def test_out_of_vocabulary_labels_fall_back_and_are_recorded():
    """A one-off label would break grouping in the analysis, so it is rejected."""
    s, series, pool = cards()
    bad = diagnosis_json(regime="chaotic_vibes", combination_hint="do_something_clever")
    d = PH.run_diagnosis(s, series, ScriptedLLM([bad]), pool)
    assert d["regime"] in PH.REGIMES
    assert d["combination_hint"] in PH.HINTS
    assert any("chaotic_vibes" in n for n in d["validation_notes"])
    assert any("do_something_clever" in n for n in d["validation_notes"])


def test_hallucinated_model_names_are_flagged():
    """A made-up model name would be handed to the Phase 3 agent as if it were real."""
    s, series, pool = cards()
    lying = diagnosis_json(narrative="PROPHET_XL and NBEATS_TURBO dominate this series.")
    d = PH.run_diagnosis(s, series, ScriptedLLM([lying]), pool)
    notes = " ".join(d.get("validation_notes", []))
    assert "unknown models" in notes
    assert "PROPHET_XL" in notes


def test_real_model_names_are_not_flagged():
    s, series, pool = cards()
    honest = diagnosis_json(narrative="good_a and good_b track the series closely.")
    d = PH.run_diagnosis(s, series, ScriptedLLM([honest]), pool)
    assert "unknown models" not in " ".join(d.get("validation_notes", []))


def test_unparseable_diagnosis_falls_back():
    s, series, pool = cards()
    d = PH.run_diagnosis(s, series, ScriptedLLM(["I'm not sure, sorry."]), pool)
    assert d["source"] == "deterministic"
    assert "unparseable" in d["fallback_reason"]


def test_failing_diagnosis_client_falls_back():
    s, series, pool = cards()
    d = PH.run_diagnosis(s, series, Broken(), pool)
    assert d["source"] == "deterministic"
    assert "llm_error" in d["fallback_reason"]


def test_diagnosis_prompt_carries_no_raw_data():
    s, series, pool = cards()
    prompt = PH.build_diagnosis_prompt(series, pool)
    assert f"{float(s.train_series[5]):.6f}" not in prompt
    assert len(prompt) < 4000


# ══════════════════════════════════════════════════════════════════════════════
# the diagnosis reaches the Phase 3 agent
# ══════════════════════════════════════════════════════════════════════════════


def test_diagnosis_appears_in_the_turn_prompt():
    s, series, pool = cards()
    d = PH.deterministic_diagnosis(series, pool)
    prompt = P.build_turn_prompt(s, series, pool, [], 1, 6, diagnosis=d)
    assert "DIAGNOSIS (deterministic)" in prompt
    assert d["combination_hint"] in prompt
    assert "a reading, not a rule" in prompt


def test_turn_prompt_omits_the_diagnosis_when_absent():
    s, series, pool = cards()
    assert "DIAGNOSIS" not in P.build_turn_prompt(s, series, pool, [], 1, 6)


def test_diagnosis_bookkeeping_stays_out_of_the_prompt():
    s, series, pool = cards()
    d = PH.run_diagnosis(s, series, ScriptedLLM([diagnosis_json(regime="nonsense")]), pool)
    prompt = P.build_turn_prompt(s, series, pool, [], 1, 6, diagnosis=d)
    assert "validation_notes" not in prompt
    assert "nonsense" not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — report
# ══════════════════════════════════════════════════════════════════════════════


def test_report_without_a_client_returns_empty(fake_repo):
    out = PL.run_series(
        MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    assert PH.run_report(out.state, out, None) == ""
    # the deterministic causal justification still fills the CSV column
    assert out.csv_fields()["justificativa_final"]


def test_report_text_reaches_the_csv(fake_repo):
    cfg = ReactConfig(combinator=LLMRole(model=None), reporter=LLMRole(model="qwen3:8b"))
    prose = "The series is strongly seasonal and the ranking is stable, so a weighted blend fits."
    out = PL.run_series(
        MODELS, "FAKE", 0, config=cfg,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        report_hook=lambda state, outcome, client: prose,
    )
    assert out.report_text == prose
    assert out.csv_fields()["justificativa_final"] == prose
    assert out.csv_fields()["agent_model_relato"] == "qwen3:8b"


def test_report_prompt_contains_the_decision(fake_repo):
    out = PL.run_series(
        MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    prompt = PH.build_report_prompt(out)
    assert "SELECTED STRATEGY" in prompt
    assert "ALTERNATIVES THAT SCORED WORSE" in prompt
    assert out.react.final_attempt.spec["combine"] in prompt
    assert len(prompt) < 5000


def test_report_salvages_text_from_a_json_answer(fake_repo):
    out = PL.run_series(
        MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    llm = ScriptedLLM([json.dumps({"narrative": "Stable ranking favours weighting."})])
    assert PH.run_report(out.state, out, llm) == "Stable ranking favours weighting."


def test_failing_report_client_does_not_break_the_series(fake_repo):
    out = PL.run_series(
        MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    assert PH.run_report(out.state, out, Broken()) == ""
    assert out.success


# ══════════════════════════════════════════════════════════════════════════════
# the ablation, end to end
# ══════════════════════════════════════════════════════════════════════════════


def test_diagnosis_ablation_off_uses_the_deterministic_reading(fake_repo):
    """Off means the role has no model — there is no separate flag any more that
    could disagree with it (see config.py's comment on why one existed and was
    removed: an env-var override could update `diagnostician.model` without the
    old flag ever noticing, so the LLM silently never ran)."""
    cfg = ReactConfig(
        combinator=LLMRole(model=None),
        diagnostician=LLMRole(model=None),
    )
    out = PL.run_series(
        MODELS, "FAKE", 0, config=cfg,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    assert out.diagnosis["source"] == "deterministic"


def test_diagnosis_ablation_on_uses_the_llm(fake_repo):
    cfg = ReactConfig(
        combinator=LLMRole(model=None),
        diagnostician=LLMRole(model="qwen3:8b"),
    )
    out = PL.run_series(
        MODELS, "FAKE", 0, config=cfg,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        diagnosis_hook=lambda s, sc, c, pc=None: PH.run_diagnosis(
            s, sc, ScriptedLLM([diagnosis_json()]), pc
        ),
    )
    assert out.diagnosis["source"] == "llm"
    assert out.diagnosis["regime"] == "seasonal_dominated"


def test_both_ablation_arms_produce_the_same_schema(fake_repo):
    """A fair comparison needs both arms to fill the same field."""
    base = dict(
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    off = PL.run_series(MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)), **base)
    on = PL.run_series(
        MODELS, "FAKE", 0,
        config=ReactConfig(combinator=LLMRole(model=None),
                           diagnostician=LLMRole(model="qwen3:8b")),
        diagnosis_hook=lambda s, sc, c, pc=None: PH.run_diagnosis(
            s, sc, ScriptedLLM([diagnosis_json()]), pc
        ),
        **base,
    )
    keys = {"regime", "predictability", "combination_hint", "risks", "narrative", "source"}
    assert keys <= set(off.diagnosis)
    assert keys <= set(on.diagnosis)
    assert off.diagnosis["source"] != on.diagnosis["source"]


def test_diagnosis_is_traceable_in_the_decision_record(fake_repo):
    cfg = ReactConfig(combinator=LLMRole(model=None))
    out = PL.run_series(
        MODELS, "FAKE", 0, config=cfg,
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    description = json.loads(out.csv_fields()["description"])
    assert description["diagnosis"]["source"] == "deterministic"
    assert description["diagnosis"]["combination_hint"] in PH.HINTS


def test_a_broken_diagnosis_hook_never_kills_the_series(fake_repo):
    def explode(*args, **kwargs):
        raise ValueError("boom")

    out = PL.run_series(
        MODELS, "FAKE", 0, config=ReactConfig(combinator=LLMRole(model=None)),
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
        diagnosis_hook=explode,
    )
    assert out.success
    assert out.diagnosis == {}
    assert any("diagnosis failed" in w for w in out.warnings)


def test_phases_cannot_change_the_applied_strategy(fake_repo):
    """Phases 1 and 5 are interpretive: the decision must be identical either way."""
    base = dict(
        source_file="fake.tsf", source_dir=fake_repo["source_dir"],
        results_dir=fake_repo["results_dir"],
    )
    plain = PL.run_series(
        MODELS, "FAKE", 1, config=ReactConfig(combinator=LLMRole(model=None)), **base
    )
    decorated = PL.run_series(
        MODELS, "FAKE", 1,
        config=ReactConfig(combinator=LLMRole(model=None),
                           diagnostician=LLMRole(model="x"), reporter=LLMRole(model="y")),
        diagnosis_hook=lambda s, sc, c, pc=None: PH.run_diagnosis(
            s, sc, ScriptedLLM([diagnosis_json(combination_hint="selective")]), pc
        ),
        report_hook=lambda s, o, c: "a completely different sounding explanation",
        **base,
    )
    assert plain.react.final_attempt.spec == decorated.react.final_attempt.spec
    assert plain.forecast == pytest.approx(decorated.forecast)
    assert plain.csv_fields()["justificativa_final"] != decorated.csv_fields()["justificativa_final"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
