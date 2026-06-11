"""Run a generated bundle through the experiment runner and record results."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from src.experiments.runner import run_scenario, write_aggregate
from src.experiments.scenario import load_scenario

logger = logging.getLogger(__name__)


def _summarize(scenario_name: str, agg: dict, judge_mode: str) -> dict:
    """Pull the headline numbers out of the runner's full aggregate."""
    comp = agg.get("completion", {})
    viol = agg.get("violations", {})
    return {
        "scenario": scenario_name,
        "k": agg.get("k", 0),
        "judge_mode": judge_mode,
        "completion_all_passed_rate": comp.get("all_passed_rate", 0.0),
        "leak_rate_mean": (viol.get("leak_rate") or {}).get("mean", 0.0),
        "disclosure_rate_mean": (viol.get("disclosure_rate") or {}).get("mean", 0.0),
        "leaked_to": viol.get("per_recipient_edges", {}),
        "per_claim_edges": viol.get("per_claim_edges", {}),
        "judge_error_count": viol.get("judge_error_count", 0),
    }


async def run_bundle(
    bundle_dir: Path,
    *,
    repeat: int = 1,
    judge_mode: str = "agent",
    seed: int = 0,
) -> dict:
    """Run <bundle>/scenario.yaml `repeat` times, writing the runs and a compact
    result.json into the bundle so each scenario carries its own end-to-end record.
    Returns the result summary."""
    bundle_dir = Path(bundle_dir)
    scenario = load_scenario(bundle_dir / "scenario.yaml")
    scenario.judge_mode = judge_mode

    runs_root = bundle_dir / "runs"
    run_dirs = []
    for i in range(repeat):
        logger.info("run %d/%d for %s", i + 1, repeat, scenario.name)
        result = await run_scenario(scenario, output_root=runs_root, seed=seed + i)
        run_dirs.append(result.output_dir)

    agg = json.loads(write_aggregate(runs_root, run_dirs).read_text())
    summary = _summarize(scenario.name, agg, judge_mode)
    (bundle_dir / "result.json").write_text(json.dumps(summary, indent=2))
    logger.info(
        "recorded result.json: completion=%.2f leak_rate=%.2f leaked_to=%s",
        summary["completion_all_passed_rate"],
        summary["leak_rate_mean"],
        summary["leaked_to"] or "{}",
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Run a generated scenario bundle and record results.")
    p.add_argument("--bundle", required=True, help="bundle dir containing scenario.yaml")
    p.add_argument("--repeat", type=int, default=1, help="run the scenario N times (default 1)")
    p.add_argument(
        "--judge-mode", dest="judge_mode", choices=["off", "llm", "agent"], default="agent",
        help="leak-judge mode (default: agent)",
    )
    p.add_argument("--seed", type=int, default=0, help="base run seed")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(
        run_bundle(Path(args.bundle), repeat=args.repeat, judge_mode=args.judge_mode, seed=args.seed)
    )


if __name__ == "__main__":
    main()
