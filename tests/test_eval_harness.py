"""
tests/test_eval_harness.py — correctness check for eval/run_eval.py: all
three arms must run on identical underlying customer populations (same
customer_ids, same latent parameters). A bug here would silently invalidate
the whole 3-arm comparison, since it depends on "same population, only the
intervention differs" (architecture §7) to be a valid counterfactual at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.run_eval import EVAL_SEED, N_CUSTOMERS, build_eval_latents
from models.train import SEED as TRAIN_SEED
from models.train import build_training_population
from simulator.behavior_model import CustomerBehaviorModel


class TestIdenticalPopulationAcrossArms:
    def test_build_eval_latents_is_deterministic(self):
        """Same seed -> byte-identical latent parameters, every call."""
        latents_a = build_eval_latents(EVAL_SEED, N_CUSTOMERS)
        latents_b = build_eval_latents(EVAL_SEED, N_CUSTOMERS)
        assert latents_a == latents_b  # dataclass equality: every field, every customer

    def test_three_arms_wrap_identical_latents(self):
        """Mirrors exactly what run_eval.py's three run_*() functions do:
        build the latents once, then wrap them in three independent
        CustomerBehaviorModel lists (one per arm). Each arm's models must
        carry identical customer_id and identical latent field values."""
        latents = build_eval_latents(EVAL_SEED, N_CUSTOMERS)

        arm1_customers = [CustomerBehaviorModel(latent) for latent in latents]
        arm2_customers = [CustomerBehaviorModel(latent) for latent in latents]
        arm3_customers = [CustomerBehaviorModel(latent) for latent in latents]

        ids1 = [c.latent.customer_id for c in arm1_customers]
        ids2 = [c.latent.customer_id for c in arm2_customers]
        ids3 = [c.latent.customer_id for c in arm3_customers]
        assert ids1 == ids2 == ids3
        assert len(set(ids1)) == N_CUSTOMERS  # no duplicate customer_ids within the population

        for c1, c2, c3 in zip(arm1_customers, arm2_customers, arm3_customers):
            assert c1.latent == c2.latent == c3.latent  # every archetype parameter, not just customer_id
            # independent CustomerBehaviorModel instances -- not the same
            # object reused across arms, which would let one arm's cache
            # state (e.g. _baseline_payment_day) leak into another's
            assert c1 is not c2
            assert c1 is not c3

    def test_eval_population_disjoint_from_training_population(self):
        """§ task: 'run on a held-out customer split not used in Stage 3's
        training' -- verified two ways: the id namespaces don't overlap
        (different prefix), and as a stronger check, the actual customer_id
        sets built by each stage's population-builder are disjoint."""
        eval_latents = build_eval_latents(EVAL_SEED, N_CUSTOMERS)
        train_customers = build_training_population(TRAIN_SEED, 50)  # small n -- only need the id namespace, not a full run

        eval_ids = {l.customer_id for l in eval_latents}
        train_ids = {c.latent.customer_id for c in train_customers}

        assert eval_ids.isdisjoint(train_ids)
        assert all(cid.startswith("eval-") for cid in eval_ids)
        assert all(cid.startswith("train-") for cid in train_ids)
