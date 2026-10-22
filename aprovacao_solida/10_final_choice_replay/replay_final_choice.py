"""Offline replay of the v11 blind final selection.

Reads the v10x artifacts, rebuilds the family shortlist and the blind prompt,
calls the selector LLM, and records the chosen candidate's test metrics. The
test actuals/metrics are used only AFTER the choice, for evaluation; they never
enter the prompt.

This is the fast way to iterate on the v11 prompt without re-running the 12-turn
ReAct loop: the artifact already stores every attempt's `validation_score`,
`spec`, `test_forecast` and `test_smape`.

Examples (server):

    # Prompt construction only, no LLM
    python3 aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
        --run orchestrator_react_v10x_low --datasets ETTH1 --limit 2 --dry-run

    # Smoke with the real selector
    REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
        aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
        --run orchestrator_react_v10x_low --datasets ETTH1 --limit 2 \
        --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11

    # Full replay
    nohup env REACT_OLLAMA_URL=http://127.0.0.1:11435 python3 \
        aprovacao_solida/10_final_choice_replay/replay_final_choice.py \
        --run orchestrator_react_v10x_low \
        --datasets ANP_MONTHLY ETTH1 ETTH2 ETTM1 ETTM2 NN5_WEEKLY_DATASET \
        --model gpt-oss:20b --reasoning low --temperature 0.2 --llm-seed 11 \
        > logs/v11_final_choice_replay_low.log 2>&1 &
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from orchestrator_react import final_choice as FC  # noqa: E402
from orchestrator_react import ingest as ing  # noqa: E402
from orchestrator_react.config import LLMRole, ReactConfig  # noqa: E402
from orchestrator_react.data_source import load_series_source  # noqa: E402
from orchestrator_react.llm import LLMClient, LLMError, build_client, check_client  # noqa: E402
from run_tsf_orchestrator import DEFAULT_MODELS, parse_reasoning  # noqa: E402
from run_tsf_batch import DATASET_SOURCES, DEFAULT_SOURCE_DIR  # noqa: E402


RESULT_COLUMNS = (
    "run", "model", "dataset", "series", "n_attempts", "shortlist_size",
    "leader_id", "leader_method", "leader_validation_score", "leader_test_smape",
    "argmin_id", "argmin_test_smape", "leader_replaced_with",
    "shortlist_oracle_smape", "top3_oracle_smape", "global_oracle_smape",
    "decision", "fallback", "choice", "confidence", "reason",
    "chosen_id", "chosen_method", "chosen_validation_score", "chosen_test_smape",
    "override", "prompt_chars", "shortlist_specs", "leader_label",
    "parse_error", "llm_seed", "temperature", "shuffle_seed", "error",
)

OFFICIAL_DATASETS = (
    "ANP_MONTHLY",
    "ETTH1",
    "ETTH2",
    "ETTM1",
    "ETTM2",
    "NN5_WEEKLY_DATASET",
)


@dataclass
class ReplayAttempt:
    attempt_id: str
    spec: Dict[str, Any]
    score: float
    test_forecast: List[float]
    test_smape: Optional[float]
    validation_rank: Optional[int]


class MockClient:
    """Deterministic stand-in used by --mock-choice; no server needed."""

    name = "mock"

    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, system: str, user: str) -> str:
        return self.response


def _prompt_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Only what was actually sent to the model plus its parsed answer.

    The true A/B/C mapping (attempt ids, methods, validation scores) lives in
    `mapping.jsonl`/`results.csv`, never in this file. Keeping the two separate
    makes the no-leakage audit unambiguous.
    """
    return {
        "dataset": row.get("dataset"),
        "series": row.get("series"),
        "system": row.get("prompt_system", ""),
        "user": row.get("prompt_user", ""),
        "raw_response": row.get("raw_response", ""),
        "choice": row.get("choice"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
    }


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "run"


def _load_artifact(
    run: str, dataset: str, index: int, results_dir: str
) -> Optional[Dict[str, Any]]:
    path = os.path.join(results_dir, run, "llm_artifacts", dataset, f"dataset_{index}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _attempts_from_artifact(payload: Dict[str, Any]) -> List[ReplayAttempt]:
    trace = payload.get("ranking_trace") or {}
    out: List[ReplayAttempt] = []
    for row in trace.get("attempts") or []:
        forecast = row.get("test_forecast")
        if not isinstance(forecast, (list, tuple)) or not forecast:
            continue
        score = row.get("validation_score")
        if score is None or not np.isfinite(float(score)):
            continue
        out.append(
            ReplayAttempt(
                attempt_id=str(row.get("id") or row.get("attempt_id") or ""),
                spec=dict(row.get("spec") or {}),
                score=float(score),
                test_forecast=[float(v) for v in forecast],
                test_smape=(
                    float(row["test_smape"])
                    if row.get("test_smape") is not None
                    and np.isfinite(float(row.get("test_smape")))
                    else None
                ),
                validation_rank=(
                    int(row["validation_rank"])
                    if row.get("validation_rank") is not None
                    else None
                ),
            )
        )
    return out


def _min_optional(values: Iterable[Optional[float]]) -> Optional[float]:
    usable = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return min(usable) if usable else None


def _artifact_indices(run: str, dataset: str, results_dir: str) -> List[int]:
    pattern = os.path.join(results_dir, run, "llm_artifacts", dataset, "dataset_*.json")
    out: List[int] = []
    for path in glob.glob(pattern):
        match = re.search(r"dataset_(\d+)\.json$", path)
        if match:
            out.append(int(match.group(1)))
    return sorted(set(out))


def _load_dataset_state_builder(
    dataset: str, results_dir: str, source_dir: str
):
    """Returns a function index -> ReactState, with frames/source cached."""
    models = list(DEFAULT_MODELS)
    frames = ing.load_dataset_frames(models, dataset, results_dir=results_dir)
    try:
        bad = ing.find_misaligned_models(
            models, dataset, 0, results_dir=results_dir, frames=frames, n_windows=3
        )
    except Exception:
        bad = {}
    drop = sorted(bad)
    source_file = DATASET_SOURCES.get(dataset)
    if not source_file:
        raise KeyError(f"no .tsf source registered for {dataset!r}")
    source = load_series_source(
        source_file,
        n_expected_series=ing.count_series(dataset, models[0], results_dir=results_dir),
        source_dir=source_dir,
    )
    cfg = ReactConfig()

    def load(index: int):
        return ing.load_series(
            models,
            dataset,
            int(index),
            source=source,
            config=cfg,
            results_dir=results_dir,
            frames=frames,
            drop_models=drop,
        )

    return load


def _decide_replay(
    payload: Dict[str, Any],
    attempts: Sequence[ReplayAttempt],
    state,
    config: ReactConfig,
    dataset: str,
    index: int,
    client: Optional[LLMClient],
    allow_keep: bool,
    dry_run: bool,
    mock_response: Optional[str],
) -> Dict[str, Any]:
    original_leader = min(attempts, key=lambda a: a.score)
    leader = original_leader
    forecast_for = lambda attempt: np.asarray(attempt.test_forecast, dtype=float)

    row: Dict[str, Any] = {
        "run": config.name,
        "dataset": dataset,
        "series": int(index),
        "n_attempts": len(attempts),
        "shortlist_size": 0,
        "leader_id": original_leader.attempt_id,
        "leader_method": str(original_leader.spec.get("combine", "?")),
        "leader_validation_score": round(float(original_leader.score), 6),
        "leader_test_smape": original_leader.test_smape,
        "argmin_id": original_leader.attempt_id,
        "argmin_test_smape": original_leader.test_smape,
        "leader_replaced_with": None,
        "shortlist_oracle_smape": None,
        "top3_oracle_smape": _min_optional(
            a.test_smape
            for a in attempts
            if a.validation_rank is not None and a.validation_rank <= 3
        ),
        "global_oracle_smape": _min_optional(a.test_smape for a in attempts),
        "decision": "fallback",
        "fallback": None,
        "choice": "",
        "confidence": "",
        "reason": "",
        "chosen_id": original_leader.attempt_id,
        "chosen_method": str(original_leader.spec.get("combine", "?")),
        "chosen_validation_score": round(float(original_leader.score), 6),
        "chosen_test_smape": original_leader.test_smape,
        "override": False,
        "prompt_chars": 0,
        "shortlist_specs": "",
    }

    try:
        ctx = FC.build_series_context(state, payload.get("series_card") or {})
    except Exception as exc:
        row["fallback"] = f"context_error:{type(exc).__name__}"
        row["error"] = str(exc)[:400]
        return row

    max_abs = float(config.final_selector_degenerate_factor) * max(
        abs(float(ctx["last_actual"])), float(ctx["scale"]), 1.0
    )
    shortlist = FC.build_family_shortlist(
        attempts,
        forecast_for,
        max_candidates=config.final_selector_max_candidates,
        max_abs=max_abs,
    )
    row["shortlist_size"] = len(shortlist)
    if not shortlist:
        row["fallback"] = "all_candidates_degenerate"
        return row

    # If the v10 argmin itself is degenerate, the effective KEEP target is the
    # best plausible family winner. The original argmin is still recorded above
    # so the comparison against v10 stays honest.
    if not any(e.attempt.attempt_id == original_leader.attempt_id for e in shortlist):
        leader = shortlist[0].attempt
        row["leader_replaced_with"] = leader.attempt_id
    row.update(
        {
            "leader_id": leader.attempt_id,
            "leader_method": str(leader.spec.get("combine", "?")),
            "leader_validation_score": round(float(leader.score), 6),
            "leader_test_smape": leader.test_smape,
            "chosen_id": leader.attempt_id,
            "chosen_method": str(leader.spec.get("combine", "?")),
            "chosen_validation_score": round(float(leader.score), 6),
            "chosen_test_smape": leader.test_smape,
        }
    )
    row["shortlist_oracle_smape"] = _min_optional(
        e.attempt.test_smape for e in shortlist
    )
    if len(shortlist) < 2:
        row["fallback"] = "shortlist_too_small"
        return row

    entries, _ = FC.blind_shortlist(
        shortlist, ctx, dataset, index, config.final_selector_shuffle_seed
    )
    system, user = FC.build_final_prompt(
        payload.get("series_card") or {},
        payload.get("diagnosis") or {},
        ctx,
        entries,
        allow_keep=allow_keep,
    )
    row["prompt_chars"] = len(system) + len(user)
    row["leader_label"] = next(
        (e.label for e in entries if e.attempt.attempt_id == leader.attempt_id), None
    )
    row["shortlist_specs"] = json.dumps(
        [
            {
                "label": e.label,
                "id": e.attempt.attempt_id,
                "method": e.attempt.spec.get("combine"),
                "validation_score": round(float(e.attempt.score), 6),
                "features": e.blind.features if e.blind else {},
            }
            for e in entries
        ],
        ensure_ascii=False,
    )

    if dry_run:
        row["decision"] = "dry_run"
        row["prompt_system"] = system
        row["prompt_user"] = user
        return row

    raw = ""
    if mock_response is not None:
        raw = mock_response
    elif client is not None:
        try:
            raw = client.complete(system, user)
        except LLMError as exc:
            row["fallback"] = "llm_error"
            row["error"] = str(exc)[:400]
            row["prompt_system"] = system
            row["prompt_user"] = user
            return row
        except Exception as exc:
            row["fallback"] = "llm_error"
            row["error"] = f"{type(exc).__name__}: {exc}"[:400]
            row["prompt_system"] = system
            row["prompt_user"] = user
            return row
    else:
        row["fallback"] = "no_client"
        row["prompt_system"] = system
        row["prompt_user"] = user
        return row

    parsed = FC.parse_choice(raw, FC.LABELS[: len(entries)], allow_keep=allow_keep)
    chosen, decision, fallback = FC.decide(
        parsed, entries, leader, ctx, config, forecast_for, allow_keep=allow_keep
    )
    row.update(
        {
            "decision": decision,
            "fallback": fallback,
            "choice": parsed.choice,
            "confidence": parsed.confidence,
            "reason": parsed.reason[:500],
            "chosen_id": chosen.attempt_id,
            "chosen_method": str(chosen.spec.get("combine", "?")),
            "chosen_validation_score": round(float(chosen.score), 6),
            "chosen_test_smape": chosen.test_smape,
            "override": bool(decision == "override"),
            "raw_response": raw,
            "prompt_system": system,
            "prompt_user": user,
            "parse_error": parsed.parse_error,
        }
    )
    return row


def _summary(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    rows: List[Dict[str, Any]] = []
    for dataset, group in results.groupby("dataset"):
        usable = group.copy()
        for col in (
            "chosen_test_smape",
            "leader_test_smape",
            "argmin_test_smape",
            "shortlist_oracle_smape",
            "top3_oracle_smape",
            "global_oracle_smape",
        ):
            usable[col] = pd.to_numeric(usable.get(col), errors="coerce")
        usable = usable[usable["chosen_test_smape"].notna() & usable["argmin_test_smape"].notna()]
        if usable.empty:
            continue
        chosen = usable["chosen_test_smape"].to_numpy(dtype=float)
        leader = usable["argmin_test_smape"].to_numpy(dtype=float)
        decisions = usable["decision"].astype(str).str.lower()
        rows.append(
            {
                "dataset": dataset,
                "n": int(len(usable)),
                "argmin_smape": float(np.mean(leader)),
                "chosen_smape": float(np.mean(chosen)),
                "shortlist_oracle_smape": float(usable["shortlist_oracle_smape"].mean()),
                "top3_oracle_smape": float(usable["top3_oracle_smape"].mean()),
                "global_oracle_smape": float(usable["global_oracle_smape"].mean()),
                "override_rate": float((decisions == "override").mean()),
                "keep_rate": float((decisions == "keep").mean()),
                "fallback_rate": float((decisions == "fallback").mean()),
                "win_rate": float(np.mean(chosen < leader - 1e-12)),
                "loss_rate": float(np.mean(chosen > leader + 1e-12)),
                "tie_rate": float(np.mean(np.abs(chosen - leader) <= 1e-12)),
                "mean_gain_vs_argmin": float(np.mean(leader - chosen)),
                "median_gain_vs_argmin": float(np.median(leader - chosen)),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        # `n` is a count, not a rate: the aggregate row reports the total, not the
        # mean of the per-dataset counts.
        mean_row = {"dataset": "MEAN_DATASETS", "n": int(summary["n"].sum())}
        for col in summary.columns:
            if col in ("dataset", "n"):
                continue
            mean_row[col] = float(summary[col].mean())
        summary = pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="orchestrator_react_v10x_low")
    parser.add_argument("--datasets", nargs="+", default=list(OFFICIAL_DATASETS))
    parser.add_argument("--indices", nargs="+", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="maximum series per dataset")
    parser.add_argument("--results-dir", default=ing.DEFAULT_RESULTS_DIR)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--reasoning", default=None, metavar="LEVEL")
    parser.add_argument("--llm-seed", type=int, default=11)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--no-keep", dest="allow_keep", action="store_false", default=True)
    parser.add_argument(
        "--no-low-confidence-keep",
        dest="low_confidence_keep",
        action="store_false",
        default=True,
    )
    parser.add_argument("--degenerate-factor", type=float, default=100.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="build prompts and write them to prompts.jsonl; no LLM call")
    parser.add_argument("--mock-choice", default=None,
                        help="testing only: respond as if the model said this choice "
                             "(e.g. A, B, KEEP); no server needed")
    parser.add_argument("--resume", action="store_true",
                        help="skip (dataset, series) pairs already in results.csv")
    parser.add_argument("--max-llm-failures", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir or os.path.join(
        "outputs/resultados/v11_final_choice_replay",
        f"{_slug(args.run)}_{_slug(args.model)}",
    )
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.csv")
    prompts_path = os.path.join(output_dir, "prompts.jsonl")
    mapping_path = os.path.join(output_dir, "mapping.jsonl")
    summary_path = os.path.join(output_dir, "summary.csv")
    command_path = os.path.join(output_dir, "run_command.txt")

    base_url = args.base_url or os.environ.get("REACT_OLLAMA_URL", "http://127.0.0.1:11434")
    config = ReactConfig()
    config.name = args.run
    config.final_selector_max_candidates = int(args.max_candidates)
    config.final_selector_allow_keep = bool(args.allow_keep)
    config.final_selector_low_confidence_keep = bool(args.low_confidence_keep)
    config.final_selector_shuffle_seed = int(args.shuffle_seed)
    config.final_selector_degenerate_factor = float(args.degenerate_factor)

    existing: set = set()
    if args.resume and os.path.exists(results_path):
        try:
            old = pd.read_csv(results_path, sep=";")
            existing = {
                (str(row["dataset"]), int(row["series"]))
                for _, row in old.iterrows()
                if pd.notna(row.get("dataset")) and pd.notna(row.get("series"))
            }
        except Exception:
            existing = set()

    client: Optional[LLMClient] = None
    mock_response = None
    if not args.dry_run:
        if args.mock_choice is not None:
            mock_response = (
                f"CHOICE: {str(args.mock_choice).strip()}\n"
                "CONFIDENCE: high\n"
                "REASON: mock replay response"
            )
            client = MockClient(mock_response)
        else:
            role = LLMRole(
                model=args.model,
                temperature=float(args.temperature),
                base_url=base_url,
                seed=int(args.llm_seed),
                reasoning=parse_reasoning(args.reasoning),
            )
            client = build_client(role)
            ok, detail = check_client(client)
            if not ok:
                print(f"LLM preflight failed: {detail}", file=sys.stderr)
                return 2

    with open(command_path, "w", encoding="utf-8") as fh:
        fh.write(
            f"# started: {datetime.now(timezone.utc).isoformat(timespec='seconds')}\n"
            f"# cwd: {os.getcwd()}\n"
            f"# base_url: {base_url}\n\n"
            + " ".join([sys.executable] + sys.argv)
            + "\n"
        )

    header_written = os.path.exists(results_path) and os.path.getsize(results_path) > 0
    llm_failures = 0
    total = 0
    abort = False
    started = time.time()

    for dataset in args.datasets:
        if abort:
            break
        indices = _artifact_indices(args.run, dataset, args.results_dir)
        if not indices:
            print(f"[{dataset}] no artifacts under {args.run}", flush=True)
            continue
        if args.indices:
            wanted = set(int(i) for i in args.indices)
            indices = [i for i in indices if i in wanted]
        if args.limit is not None:
            indices = indices[: int(args.limit)]
        if not indices:
            continue

        try:
            load_state = _load_dataset_state_builder(dataset, args.results_dir, args.source_dir)
        except Exception as exc:
            print(f"[{dataset}] could not load data: {type(exc).__name__}: {exc}", flush=True)
            continue

        print(f"[{dataset}] {len(indices)} series", flush=True)
        for index in indices:
            total += 1
            key = (dataset, int(index))
            if key in existing:
                print(f"  {dataset} {index}: resume-skip", flush=True)
                continue
            payload = _load_artifact(args.run, dataset, index, args.results_dir)
            if payload is None:
                print(f"  {dataset} {index}: artifact missing", flush=True)
                continue
            attempts = _attempts_from_artifact(payload)
            if not attempts:
                print(f"  {dataset} {index}: no attempts in artifact", flush=True)
                continue

            try:
                ingested = load_state(index)
                row = _decide_replay(
                    payload=payload,
                    attempts=attempts,
                    state=ingested.state,
                    config=config,
                    dataset=dataset,
                    index=index,
                    client=client,
                    allow_keep=args.allow_keep,
                    dry_run=args.dry_run,
                    mock_response=mock_response,
                )
            except Exception as exc:
                row = {
                    "run": args.run,
                    "dataset": dataset,
                    "series": int(index),
                    "decision": "fallback",
                    "fallback": f"replay_error:{type(exc).__name__}",
                    "error": str(exc)[:400],
                }
                print(f"  {dataset} {index}: ERROR {type(exc).__name__}: {exc}", flush=True)

            row.setdefault("model", args.model)
            row.setdefault("llm_seed", args.llm_seed)
            row.setdefault("temperature", args.temperature)
            row.setdefault("shuffle_seed", args.shuffle_seed)

            if args.dry_run:
                with open(prompts_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(_prompt_record(row), ensure_ascii=False, default=str) + "\n")
                with open(mapping_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                print(
                    f"  {dataset} {index}: dry-run shortlist={row.get('shortlist_size')} "
                    f"chars={row.get('prompt_chars')}",
                    flush=True,
                )
                continue

            if row.get("fallback") == "llm_error":
                llm_failures += 1
            else:
                llm_failures = 0

            # Keep the result CSV schema fixed even when optional diagnosis
            # fields (parse_error/error) are present on only some rows.
            result_row = {key: row.get(key) for key in RESULT_COLUMNS}
            frame = pd.DataFrame([result_row])
            frame.to_csv(
                results_path,
                sep=";",
                mode="a",
                header=not header_written,
                index=False,
            )
            header_written = True
            with open(prompts_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(_prompt_record(row), ensure_ascii=False, default=str) + "\n")
            with open(mapping_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

            print(
                f"  {dataset} {index}: {row.get('decision')} "
                f"choice={row.get('choice') or '-'} "
                f"chosen={row.get('chosen_test_smape')} "
                f"argmin={row.get('leader_test_smape')} "
                f"fallback={row.get('fallback') or '-'}",
                flush=True,
            )
            if args.max_llm_failures and llm_failures >= int(args.max_llm_failures):
                print(
                    f"aborting after {llm_failures} consecutive LLM failures",
                    file=sys.stderr,
                    flush=True,
                )
                abort = True
                break

    if not args.dry_run and os.path.exists(results_path):
        results = pd.read_csv(results_path, sep=";")
        summary = _summary(results)
        if not summary.empty:
            summary.to_csv(summary_path, sep=";", index=False)
            print("\nSUMMARY", flush=True)
            print(summary.to_string(index=False), flush=True)
        else:
            print("\nNo usable rows to summarise.", flush=True)

    print(
        f"\nDone in {time.time() - started:.1f}s. Output: {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
