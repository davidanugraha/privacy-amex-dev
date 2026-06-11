"""Load the thin seeds and sample a (domain, topology, data_form) triplet."""

from __future__ import annotations

import random
from pathlib import Path
from typing import get_args

import yaml

from src.scenario_gen.ir import DataForm, Topology, Triplet

SEEDS_DIR = Path(__file__).parent / "seeds"
DOMAIN_SEED = SEEDS_DIR / "domain_seed.yaml"
STRUCTURAL_SEED = SEEDS_DIR / "structural_seed.yaml"


def _clean(text: str) -> str:
    """Collapse YAML folded-scalar whitespace into one tidy line."""
    return " ".join(text.split())


class SeedSpace:
    """The sampling space parsed from the two seed files."""

    def __init__(self, domains: dict[str, str], topologies: dict[str, str], data_forms: dict[str, str]):
        self.domains = domains
        self.topologies = topologies
        self.data_forms = data_forms

    @classmethod
    def load(cls, domain_seed: Path = DOMAIN_SEED, structural_seed: Path = STRUCTURAL_SEED) -> "SeedSpace":
        raw_domains = yaml.safe_load(domain_seed.read_text())["domains"]
        raw_struct = yaml.safe_load(structural_seed.read_text())
        domains = {k: _clean(v["description"]) for k, v in raw_domains.items()}
        topologies = {k: _clean(v["description"]) for k, v in raw_struct["topology"].items()}
        data_forms = {k: _clean(v["description"]) for k, v in raw_struct["data_form"].items()}

        # seed keys must match the IR Literals exactly
        _assert_covers("topology", topologies.keys(), get_args(Topology))
        _assert_covers("data_form", data_forms.keys(), get_args(DataForm))
        return cls(domains, topologies, data_forms)

    @property
    def n_cells(self) -> int:
        return len(self.domains) * len(self.topologies) * len(self.data_forms)

    def make_triplet(self, domain: str, topology: Topology, data_form: DataForm) -> Triplet:
        for name, key, table in (
            ("domain", domain, self.domains),
            ("topology", topology, self.topologies),
            ("data_form", data_form, self.data_forms),
        ):
            if key not in table:
                raise KeyError(f"unknown {name} {key!r}; options: {sorted(table)}")
        return Triplet(
            domain=domain,
            topology=topology,  # type: ignore[arg-type]
            data_form=data_form,  # type: ignore[arg-type]
            domain_description=self.domains[domain],
            topology_description=self.topologies[topology],
            data_form_description=self.data_forms[data_form],
        )

    def sample(self, rng: random.Random) -> Triplet:
        return self.make_triplet(
            rng.choice(list(self.domains)),
            rng.choice(list(self.topologies)),  # type: ignore[arg-type]
            rng.choice(list(self.data_forms)),  # type: ignore[arg-type]
        )


def _assert_covers(axis: str, present, expected) -> None:
    missing = set(expected) - set(present)
    extra = set(present) - set(expected)
    if missing or extra:
        raise ValueError(f"{axis} seed mismatch — missing {missing or '∅'}, extra {extra or '∅'}")


def sample_triplet(
    *,
    seed: int | None = None,
    domain: str | None = None,
    topology: Topology | None = None,
    data_form: DataForm | None = None,
    space: SeedSpace | None = None,
) -> Triplet:
    """Pin any of domain/topology/data_form; the rest are drawn from `seed`."""
    space = space or SeedSpace.load()
    rng = random.Random(seed)
    return space.make_triplet(
        domain or rng.choice(list(space.domains)),
        topology or rng.choice(list(space.topologies)),  # type: ignore[arg-type]
        data_form or rng.choice(list(space.data_forms)),  # type: ignore[arg-type]
    )
