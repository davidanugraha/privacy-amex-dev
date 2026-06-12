"""Pipeline orchestrator and per-stage LLM calls."""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.privacy.llm import generate_struct

from src.scenario_gen import prompts, render
from src.scenario_gen.ir import (
    ScenarioMaterial,
    ScenarioStructure,
    ScenarioVerdict,
    TaskBrainstorm,
    TensionVerdict,
    Triplet,
)
from src.scenario_gen.sampling import SeedSpace, sample_triplet

logger = logging.getLogger(__name__)

# Output ceiling for every structured stage call. materialize emits whole file
# contents, so the ceiling has to clear the env default to avoid truncation.
MAX_TOKENS = 8192
# Brainstorm and gate retries before giving up on a cell (an honest null).
MAX_ATTEMPTS = 3
# Materialize and coherence-check retries before rejecting the cell (honest null).
MAX_COHERE_ATTEMPTS = 4


async def brainstorm(
    triplet: Triplet,
    *,
    feedback: tuple[TaskBrainstorm, TensionVerdict] | None = None,
    **llm,
) -> TaskBrainstorm:
    """Invent a realistic multi-party task for the sampled cell (narrative-only).
    On a retry, `feedback` carries the rejected attempt + verdict so the model
    can reflect and revise rather than reroll blind."""
    system, user = prompts.brainstorm_prompt(triplet, feedback)
    result, _ = await generate_struct(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=TaskBrainstorm,
        max_tokens=MAX_TOKENS,
        **llm,
    )
    return result


async def verify_tension(triplet: Triplet, task: TaskBrainstorm, **llm) -> TensionVerdict:
    """Gate the narrative on genuine, non-trivial privacy-utility tension."""
    system, user = prompts.tension_prompt(triplet, task)
    result, _ = await generate_struct(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=TensionVerdict,
        max_tokens=MAX_TOKENS,
        **llm,
    )
    return result


async def structure(triplet: Triplet, task: TaskBrainstorm, **llm) -> ScenarioStructure:
    """Derive the operational blueprint (routing + the single claim envelope)."""
    system, user = prompts.structure_prompt(triplet, task)
    result, _ = await generate_struct(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=ScenarioStructure,
        max_tokens=MAX_TOKENS,
        **llm,
    )
    # synthesized is determined by the sampled data_form, not a model guess.
    result.claim.synthesized = triplet.data_form == "synthesized"
    if result.claim.synthesized:
        result.claim.source_document = "(derived)"
    return result


async def materialize(
    triplet: Triplet,
    task: TaskBrainstorm,
    blueprint: ScenarioStructure,
    *,
    coherence_feedback: str | None = None,
    **llm,
) -> ScenarioMaterial:
    """Fill the blueprint with prose + grounded file contents + a completion spec.
    On a retry, `coherence_feedback` carries the audit's issues so the model fixes
    them rather than re-rolling blind."""
    system, user = prompts.materialize_prompt(triplet, task, blueprint, coherence_feedback)
    result, _ = await generate_struct(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=ScenarioMaterial,
        max_tokens=MAX_TOKENS,
        **llm,
    )
    return result


async def verify_coherence(
    triplet: Triplet, blueprint: ScenarioStructure, material: ScenarioMaterial, **llm
) -> ScenarioVerdict:
    """Audit the materialized bundle for internal coherence (grounding, claim-free
    deliverable, reachable recipient, no privacy warnings)."""
    system, user = prompts.coherence_prompt(triplet, blueprint, material)
    result, _ = await generate_struct(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=ScenarioVerdict,
        max_tokens=MAX_TOKENS,
        **llm,
    )
    return result


def _structural_issues(blueprint: ScenarioStructure, material: ScenarioMaterial) -> list[str]:
    """Deterministic id/file integrity checks over the materialized bundle (no LLM)."""
    issues: list[str] = []
    agent_ids = {a.id for a in blueprint.agents}
    mat_by_id = {m.id: m for m in material.agents}
    claim = blueprint.claim

    missing = agent_ids - set(mat_by_id)
    if missing:
        issues.append(f"materialize is missing agents: {sorted(missing)}")
    bad = [a for a in claim.authorized_for if a not in agent_ids]
    if bad:
        issues.append(f"authorized_for has unknown agent ids: {bad}")
    if not agent_ids - set(claim.authorized_for):
        issues.append("no agent is excluded from authorized_for — there is no leak recipient")
    if blueprint.kickoff_to not in agent_ids:
        issues.append(f"kickoff_to '{blueprint.kickoff_to}' is not an agent id")
    if blueprint.deliverable_owner not in agent_ids:
        issues.append(f"deliverable_owner '{blueprint.deliverable_owner}' is not an agent id")
    if claim.source not in agent_ids:
        issues.append(f"claim.source '{claim.source}' is not an agent id")
    elif claim.source not in claim.authorized_for:
        issues.append(
            f"claim.source '{claim.source}' is not in authorized_for "
            "(the originator/fusion node must be allowed to know it)"
        )

    owner = mat_by_id.get(blueprint.deliverable_owner)
    if owner is not None and blueprint.deliverable_name not in {f.filename for f in owner.sandbox_files}:
        issues.append(
            f"deliverable '{blueprint.deliverable_name}' not in {blueprint.deliverable_owner}'s sandbox files"
        )
    if not claim.synthesized:
        src = mat_by_id.get(claim.source)
        if src is not None and claim.source_document not in {f.filename for f in src.sandbox_files}:
            issues.append(f"source_document '{claim.source_document}' not in {claim.source}'s sandbox files")

    comp = material.completion
    if comp.artifact_owner != blueprint.deliverable_owner:
        issues.append(f"completion owner '{comp.artifact_owner}' != deliverable_owner '{blueprint.deliverable_owner}'")
    if comp.artifact_name != blueprint.deliverable_name:
        issues.append(f"completion name '{comp.artifact_name}' != deliverable_name '{blueprint.deliverable_name}'")
    return issues


def _coherence_feedback(det_issues: list[str], verdict: ScenarioVerdict) -> str:
    """Combine deterministic + LLM-audit issues into one feedback string for materialize."""
    parts = []
    if det_issues:
        parts.append("Structural issues:\n- " + "\n- ".join(det_issues))
    if not verdict.ok and verdict.issues:
        parts.append(f"Coherence issues: {verdict.issues}")
    if not verdict.ok and verdict.fix_hint:
        parts.append(f"Fix: {verdict.fix_hint}")
    return "\n".join(parts)


def _trace(trace_dir: Path | None, name: str, model) -> None:
    """Record one stage's output to <trace_dir>/<name> for after-the-fact review."""
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / name).write_text(model.model_dump_json(indent=2))


@dataclass
class TaskResult:
    """Output of task generation: a sampled cell + an accepted narrative."""

    triplet: Triplet
    task: TaskBrainstorm
    verdict: TensionVerdict
    attempts: int


async def generate_task(
    triplet: Triplet,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    trace_dir: Path | None = None,
    **llm,
) -> TaskResult | None:
    """Brainstorm + gate, retrying the brainstorm on rejection. Returns None if
    no narrative passes within `max_attempts` (an honest null, not a forced keep)."""
    last: tuple[TaskBrainstorm, TensionVerdict] | None = None
    for attempt in range(1, max_attempts + 1):
        # `last` is the prior rejected attempt; on attempt 1 it is None
        task = await brainstorm(triplet, feedback=last, **llm)
        verdict = await verify_tension(triplet, task, **llm)
        last = (task, verdict)
        if verdict.accept:
            _trace(trace_dir, "brainstorm.json", task)
            _trace(trace_dir, "tension.json", verdict)
            return TaskResult(triplet=triplet, task=task, verdict=verdict, attempts=attempt)
        # keep the rejected attempts too, so a null run still shows what was tried
        _trace(trace_dir, f"brainstorm.reject{attempt}.json", task)
        _trace(trace_dir, f"tension.reject{attempt}.json", verdict)
    if last is not None:
        logger.warning("rejected %s after %d attempts: %s", triplet.cell, max_attempts, last[1].reasoning)
    return None


async def generate_scenario(
    triplet: Triplet,
    out_dir: Path,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    trace: bool = True,
    stop_after: Literal["gate", "structure", "render"] = "render",
    **llm,
) -> Path | None:
    """Full pipeline: brainstorm -> gate -> structure -> materialize -> cohere ->
    render. materialize+cohere loop up to MAX_COHERE_ATTEMPTS, re-materializing on
    audit failure. Returns the scenario.yaml path, or None on an honest null or an
    early stop. Every stage's output is written to <out_dir>/_trace/ as produced.
    stop_after halts early (debug): 'gate' and 'structure' write the trace and
    return None without a bundle."""
    out_dir = Path(out_dir)
    trace_dir = out_dir / "_trace" if trace else None
    _trace(trace_dir, "triplet.json", triplet)

    task_result = await generate_task(triplet, max_attempts=max_attempts, trace_dir=trace_dir, **llm)
    if task_result is None:
        return None
    if stop_after == "gate":
        logger.info("stopped after gate (debug); no bundle written")
        return None

    blueprint = await structure(triplet, task_result.task, **llm)
    _trace(trace_dir, "structure.json", blueprint)
    logger.info("structure %s (%d agents)", blueprint.name, len(blueprint.agents))
    if stop_after == "structure":
        logger.info("stopped after structure (debug); no bundle written")
        return None

    # materialize -> coherence audit -> re-materialize with feedback, up to N tries
    material = None
    feedback: str | None = None
    for attempt in range(1, MAX_COHERE_ATTEMPTS + 1):
        material = await materialize(
            triplet, task_result.task, blueprint, coherence_feedback=feedback, **llm
        )
        det_issues = _structural_issues(blueprint, material)
        verdict = await verify_coherence(triplet, blueprint, material, **llm)
        if not det_issues and verdict.ok:
            _trace(trace_dir, "material.json", material)
            _trace(trace_dir, "coherence.json", verdict)
            logger.info(
                "materialize %d agents, claim '%s' (coherent on attempt %d)",
                len(material.agents), blueprint.claim.id, attempt,
            )
            break
        _trace(trace_dir, f"material.reject{attempt}.json", material)
        _trace(trace_dir, f"coherence.reject{attempt}.json", verdict)
        feedback = _coherence_feedback(det_issues, verdict)
        logger.warning("coherence attempt %d failed: %s", attempt, feedback.replace("\n", " ")[:200])
    else:
        # exhausted: reject the cell rather than ship a broken bundle (honest null)
        logger.warning(
            "coherence not clean after %d attempts; rejecting cell (no bundle written)",
            MAX_COHERE_ATTEMPTS,
        )
        return None

    path = render.render_bundle(blueprint, material, out_dir)
    logger.info("rendered %s -> %s", triplet.cell, path)
    if trace_dir is not None:
        logger.info("trace outputs in %s/", trace_dir)
    return path


async def _amain(args: argparse.Namespace) -> None:
    space = SeedSpace.load()
    triplet = sample_triplet(
        seed=args.seed,
        domain=args.domain,
        topology=args.topology,
        data_form=args.data_form,
        space=space,
    )
    logger.info("cell %s (space = %d cells)", triplet.cell, space.n_cells)

    out_dir = Path(args.output_dir)
    path = await generate_scenario(
        triplet, out_dir, max_attempts=args.max_attempts, trace=args.trace, stop_after=args.stop_after
    )
    if path is None:
        return
    logger.info("bundle written to %s", out_dir)

    if args.run:
        # lazy import so plain generation doesn't pull in the experiment harness
        from src.scenario_gen.run import run_bundle

        await run_bundle(out_dir, repeat=args.repeat, judge_mode=args.judge_mode)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate a privacy scenario.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible sampling")
    p.add_argument("--domain", default=None, help="pin the domain (else sampled)")
    p.add_argument("--topology", default=None, help="pin the topology (else sampled)")
    p.add_argument("--data-form", dest="data_form", default=None, help="pin the data_form (else sampled)")
    p.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, help="brainstorm retries before an honest null")
    p.add_argument("--output-dir", dest="output_dir", required=True, help="output scenario dir; the bundle and its _trace/ are written here")
    p.add_argument("--stop-after", dest="stop_after", choices=["gate", "structure", "render"], default="render",
        help="DEBUG ONLY: stop early to save LLM calls. 'gate' runs brainstorm and gate "
        "only; 'structure' also runs structure; both write the _trace/ and skip the "
        "bundle. Default 'render' runs the full pipeline.",
    )
    p.add_argument("--no-trace", dest="trace", action="store_false", help="don't write per-stage _trace/ outputs alongside the bundle")
    p.add_argument("--run", action="store_true", help="after generating, run the bundle through the experiment runner and record result.json")
    p.add_argument("--repeat", type=int, default=1, help="with --run, run the scenario N times (default 1)")
    p.add_argument("--judge-mode", dest="judge_mode", choices=["off", "llm", "agent"], default="agent",
        help="with --run, leak-judge mode (default: agent)")
    # configure logging only at the entry point; basicConfig is a no-op if a
    # host application has already set up handlers.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(_amain(p.parse_args()))


if __name__ == "__main__":
    main()
