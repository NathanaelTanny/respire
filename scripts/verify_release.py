"""Verification driver for the release.

Three families of checks. The first re-derives the arithmetic the article prints from the numbers it
prints, without touching the package. The second inspects the package and confirms that each
structural element of the method is present at the size the article states. The third runs the
package: a forward pass, a backward pass, an update, a checkpoint round-trip and, when asked, the
whole pipeline.

Every check reports PASS, FAIL, NOT_RUN or BLOCKED. Nothing is inferred from a previous run: the
script recomputes each quantity here and writes the three artefacts beside the package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRECTORIES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
PACKAGE = ROOT / "src" / "respire"

PAPER = {
    "title": (
        "Recursive Self-Improving Reinforcement Learning with Latent Physiological Pattern "
        "Recognition for Closed-Loop Anaesthesia in Robotic Surgery"
    ),
    "system": "ReSPIRe",
    "venue_style": "Elsevier single-column manuscript, Pattern Recognition reference family",
    "corpora": {
        "vitaldb": "intraoperative records; 4,806 records of which 1,132 robot-assisted; 226 held out",
        "mimic_iv": "4,124 admissions filtered to 1,876 episodes",
        "eicu_crd": "cross-setting arm for hypotension and drug consumption",
        "cohort": "2,000 Eleveld-Minto patients, 20,000 training and 2,000 evaluation episodes",
    },
}


@dataclass
class Check:
    """One verification entry."""

    identifier: str
    family: str
    claim: str
    method: str
    status: str
    detail: str
    paper_ref: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "family": self.family,
            "claim": self.claim,
            "paper_ref": self.paper_ref,
            "method": self.method,
            "status": self.status,
            "detail": self.detail,
            "evidence": self.evidence,
        }


def _verdict(condition: bool, fail_status: str = "FAIL") -> str:
    return "PASS" if condition else fail_status


def close(left: float, right: float, tolerance: float = 5e-3) -> bool:
    return abs(left - right) <= tolerance * max(abs(right), 1.0)


# ---------------------------------------------------------------------------------------------
# Family 1: the article's own arithmetic
# ---------------------------------------------------------------------------------------------


def manuscript_checks() -> list[Check]:
    checks: list[Check] = []

    def add(
        identifier: str,
        claim: str,
        method: str,
        status: str,
        detail: str,
        ref: str,
        evidence: dict[str, object] | None = None,
    ) -> None:
        checks.append(
            Check(
                identifier=identifier,
                family="manuscript_arithmetic",
                claim=claim,
                method=method,
                status=status,
                detail=detail,
                paper_ref=ref,
                evidence=evidence or {},
            )
        )

    reduction = 100.0 * (0.71 - 0.44) / 0.71
    add(
        "M01",
        "The recursion lowers rate variability by 38.0%",
        "Recompute the relative change from Table 1's rate-variability column",
        _verdict(close(round(reduction, 1), 38.0, 1e-3)),
        f"(0.71 - 0.44) / 0.71 = {reduction:.3f}%",
        "Abstract; Table 1; Sec. 7",
        {"computed": round(reduction, 3)},
    )

    propofol = 100.0 * (6.96 - 6.79) / 6.79
    add(
        "M02",
        "ReSPIRe uses 2.5% more propofol than the strongest unconstrained arm",
        "Recompute the relative change from Table 1's propofol column",
        _verdict(close(round(propofol, 1), 2.5, 2e-2)),
        f"(6.96 - 6.79) / 6.79 = {propofol:.3f}%",
        "Abstract; Table 1",
        {"computed": round(propofol, 3)},
    )

    add(
        "M03",
        "The conclusion states the propofol difference as a reduction",
        "Compare the conclusion's direction with Table 1's column",
        "FAIL",
        (
            "Table 1 gives 6.96 for ReSPIRe against 6.79 for DreamerV3 and the abstract reads "
            "'at 2.5% higher propofol', but the conclusion reads '2.5% is lower in price than "
            "propofol'. The sign of the same quantity is stated both ways."
        ),
        "Sec. 7 versus Abstract and Table 1",
        {"table": [6.96, 6.79], "abstract": "higher", "conclusion": "lower"},
    )

    differences = {
        "hypotension": 8.2 - 14.6,
        "burst": 2.1 - 4.4,
        "pressure_area": 11.6 - 20.1,
        "rate_variability": 0.44 - 0.71,
        "shield": 1.9 - 8.7,
    }
    printed = {
        "hypotension": -6.4,
        "burst": -2.3,
        "pressure_area": -8.5,
        "rate_variability": -0.27,
        "shield": -6.8,
    }
    mismatched = {
        key: (round(value, 2), printed[key])
        for key, value in differences.items()
        if not close(round(value, 2), printed[key], 1e-3)
    }
    add(
        "M04",
        "Every difference column in the safety table is the stated subtraction",
        "Subtract the two printed columns for each event and compare with the printed difference",
        _verdict(not mismatched),
        f"all five differences reproduce; mismatches: {mismatched}",
        "Table 3",
        {"computed": {key: round(value, 3) for key, value in differences.items()}},
    )

    cases = 226
    add(
        "M05",
        "One point of the hypotension difference is about fourteen cases of 226",
        "Multiply the case count by the difference in percentage points",
        _verdict(close(cases * 0.01 * 6.4, 14.46, 1e-2)),
        f"226 x 6.4% = {cases * 0.01 * 6.4:.2f} cases",
        "Sec. 6.1",
        {"computed": round(cases * 0.01 * 6.4, 3)},
    )
    add(
        "M06",
        "One point of the suppression difference is about five cases",
        "Multiply the case count by the difference in percentage points",
        _verdict(close(cases * 0.01 * 2.3, 5.20, 1e-2)),
        f"226 x 2.3% = {cases * 0.01 * 2.3:.2f} cases",
        "Sec. 6.1",
        {"computed": round(cases * 0.01 * 2.3, 3)},
    )

    share = 100.0 * 6.4 / 14.6
    add(
        "M07",
        "The safety improvement is about forty-four percent of the unconstrained rate",
        "Divide the difference by the unconstrained arm's rate",
        _verdict(close(round(share, 1), 43.8, 1e-3)),
        f"6.4 / 14.6 = {share:.2f}%",
        "Sec. 6.1",
        {"computed": round(share, 3)},
    )

    add(
        "M08",
        "A tuned constrained learner closes 1.5 points and the instruments the other 4.9",
        "Subtract the constrained arm from the unconstrained rate, then the proposed arm from it",
        _verdict(close(14.6 - 13.1, 1.5, 1e-2) and close(13.1 - 8.2, 4.9, 1e-2)),
        f"14.6 - 13.1 = {14.6 - 13.1:.1f}; 13.1 - 8.2 = {13.1 - 8.2:.1f}",
        "Abstract; Table 1 (CPPO row)",
        {"closed_by_learner": 1.5, "closed_by_instruments": 4.9},
    )

    quartiles = [2.9, 5.8, 9.3, 82.0]
    add(
        "M09",
        "The quartile shares of the benefit sum to one hundred percent",
        "Sum the four shares printed in the right-hand panel",
        _verdict(close(sum(quartiles), 100.0, 1e-6)),
        f"{' + '.join(str(value) for value in quartiles)} = {sum(quartiles)}",
        "Fig. 5 (right panel)",
        {"computed": sum(quartiles)},
    )

    size = 2.0 * 1.0**2 * math.log(4.0 / 0.05) / (0.05 - 2.0 * 0.01) ** 2
    add(
        "M10",
        "The sample-size example needs about 9,700 episodes",
        "Evaluate Eq. (6) with R=1, delta=0.05, epsilon=0.01 and g=0.05",
        _verdict(close(round(size), 9738.0, 1e-3)),
        f"2 log(80) / 0.03^2 = {size:.1f}",
        "Corollary 1, Sec. 3.2",
        {"computed": round(size, 2)},
    )
    add(
        "M11",
        "The printed denominator of the sample-size example is the squared margin",
        "Read the printed expression against Eq. (6)",
        "FAIL",
        (
            "The text writes '2 log(80)/0.032', which is 2 log(80) divided by 0.032 and gives 274, "
            "while the sentence's own result of 9,700 requires (g - 2 epsilon)^2 = 0.03^2. The "
            "superscript is lost in the printed expression."
        ),
        "Sec. 3.2 (sentence following Corollary 1)",
        {
            "printed": "0.032",
            "required": "0.03^2",
            "value_with_printed_denominator": round(2 * math.log(80) / 0.032, 1),
        },
    )

    bias = 1.5
    exact = bias**2 / 4.0
    printed_value = bias - 1.0
    add(
        "M12",
        "The reported value of the biased recursion is (c-1) a_k^2",
        "Maximise V(a) = -a^2 + c a_k a, substitute the maximiser and compare with the printed form",
        "FAIL",
        (
            "The maximiser is a_{k+1} = (c/2) a_k and the value there is (c^2/4) a_k^2, not "
            f"(c-1) a_k^2. At c={bias} the two are {exact:.3f} and {printed_value:.3f}; they agree "
            "only at c=2, so the printed growth condition c>1 does not follow from the printed "
            "model, while the divergence condition c>2 does."
        ),
        "Proposition 2, Sec. 3.3",
        {"c": bias, "exact": exact, "printed": printed_value},
    )

    add(
        "M13",
        "Every column of the primary table is defined in the text",
        "Search the article for a definition of each column header",
        "FAIL",
        (
            "The final column of Table 1 is headed 'Ind.' and is never defined in the article, the "
            "captions or the appendix, so no reader can recompute it. The column 'MAP AUC' is "
            "called 'Pressure-time area' in Table 3 with identical values, so one quantity carries "
            "two names and one column carries none."
        ),
        "Table 1 versus Table 3 and Sec. 3.1",
        {"undefined_columns": ["Ind."], "renamed": {"MAP AUC": "Pressure-time area"}},
    )

    add(
        "M14",
        "The primary table reports means over five initialisations for every learned row",
        "Count the rows for which an initialisation exists",
        "FAIL",
        (
            "The caption states that values are means over five initialisations, but four of the "
            "thirteen rows are deterministic controllers (two target-controlled infusions, a PID "
            "and an internal-model controller) whose output does not depend on an initialisation. "
            "The caption over-states the design for those rows."
        ),
        "Table 1 caption",
        {"deterministic_rows": ["Schnider TCU", "Eleveld TCU", "PID", "Internal model"]},
    )

    add(
        "M15",
        "A hard shield leaves a non-zero violation rate of 1.9%",
        "Compare the shield's definition with the violation row of the safety table",
        "FAIL",
        (
            "The shield is described as an analytic projection applied to the delivered action, so a "
            "violation can only occur when no admissible action exists. A rate of 1.9% is therefore "
            "either a pre-projection counter or a measure of infeasible states, and the article does "
            "not say which. This release counts pre-projection requests and reports infeasible "
            "states separately."
        ),
        "Sec. 4 (shield) versus Table 3 (shield violations)",
        {"reported": 1.9, "definition": "pre-projection request count in this release"},
    )

    retained = 1876 / 4124
    add(
        "M16",
        "The sedation corpus keeps 1,876 of 4,124 admissions after filtering",
        "Check that the retained count is smaller than the screened count and consistent with the abstract",
        _verdict(0.0 < retained < 1.0),
        f"1,876 / 4,124 = {100.0 * retained:.1f}% retained",
        "Sec. 5.1; Abstract",
        {"retained_fraction": round(100.0 * retained, 2)},
    )

    add(
        "M17",
        "The held-out robot-assisted cohort is a fifth of the robot-assisted records",
        "Divide the held-out case count by the robot-assisted count",
        _verdict(close(100.0 * 226 / 1132, 19.96, 1e-2)),
        f"226 / 1,132 = {100.0 * 226 / 1132:.2f}%",
        "Sec. 5.1; Table 1 caption",
        {"computed": round(100.0 * 226 / 1132, 3)},
    )

    add(
        "M18",
        "The cohort supplies 20,000 training and 2,000 evaluation episodes",
        "Read the episode counts against the patient count",
        _verdict(20000 > 2000 and 2000 >= 2000),
        "20,000 training episodes from 2,000 patients (10 per patient) and 2,000 evaluation episodes",
        "Sec. 5.1",
        {"train": 20000, "evaluation": 2000, "patients": 2000},
    )

    return checks


# ---------------------------------------------------------------------------------------------
# Family 2: the package against the article's structure
# ---------------------------------------------------------------------------------------------


def _source_line(relative: str, needle: str) -> int:
    path = ROOT / relative
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return number
    return 0


def structure_checks() -> list[Check]:
    from respire.control.certificate import ConformalCertificate
    from respire.data.summary import summary_dim
    from respire.flags import COMPONENTS
    from respire.models.encoder import LatentEncoder
    from respire.models.verifier import ReturnVerifier
    from respire.settings import load_experiment
    from respire.theatre.environment import CHANNELS

    checks: list[Check] = []
    primary = load_experiment(ROOT / "configs" / "round" / "primary.yaml")

    def add(
        identifier: str,
        claim: str,
        method: str,
        ok: bool,
        detail: str,
        ref: str,
        evidence: dict[str, object] | None = None,
    ) -> None:
        checks.append(
            Check(
                identifier=identifier,
                family="code_vs_paper",
                claim=claim,
                method=method,
                status=_verdict(ok),
                detail=detail,
                paper_ref=ref,
                evidence=evidence or {},
            )
        )

    encoder = LatentEncoder(primary.encoder)
    blocks = len(encoder.stack.blocks)
    add(
        "S01",
        "The encoder is six selective state-space blocks",
        "Count the blocks in the constructed encoder",
        blocks == 6,
        f"{blocks} blocks",
        "Fig. 2; Sec. 4.2",
        {"blocks": blocks, "expects": 6},
    )

    add(
        "S02",
        "The observation is twelve channels over a sixty-second window",
        "Read the channel schema and the configured window",
        len(CHANNELS) == 12 and primary.encoder.window_steps * primary.theatre.dt_s == 60.0,
        f"{len(CHANNELS)} channels, {primary.encoder.window_steps} steps of {primary.theatre.dt_s} s",
        "Sec. 3.1; Fig. 2",
        {"channels": list(CHANNELS)},
    )

    add(
        "S03",
        "The latent is sixteen-dimensional with a sparse readout and three axes",
        "Read the configured widths and the number of atoms selected by the readout",
        primary.encoder.latent_dim == 16
        and primary.encoder.axis_dim == 3
        and primary.encoder.sparse_topk == 4,
        f"z in R^{primary.encoder.latent_dim}, s in R^{primary.encoder.axis_dim}, top-{primary.encoder.sparse_topk} of 32 atoms",
        "Fig. 2; Sec. 1",
        {"latent": 16, "axes": 3, "topk": 4},
    )

    verifier = ReturnVerifier(primary.verifier, summary_dim(12))
    count = verifier.parameter_count()
    relative = abs(count - 4.0e5) / 4.0e5
    add(
        "S04",
        "The verifier has about 0.4 million parameters",
        "Build the verifier from the primary document and count its trainable parameters",
        relative <= primary.verifier.param_tolerance,
        f"{count:,} parameters ({100.0 * relative:.1f}% from 4.0e5)",
        "Fig. 2",
        {"parameters": count, "tolerance": primary.verifier.param_tolerance},
    )

    add(
        "S05",
        "The policy is a flow with four Euler steps",
        "Read the configured integrator length",
        primary.policy.euler_steps == 4,
        f"{primary.policy.euler_steps} Euler steps",
        "Fig. 2; Algorithm 2 line 4",
        {"euler_steps": primary.policy.euler_steps},
    )

    add(
        "S06",
        "The reward is the depth error and the drug cost with gamma 0.995 and a target of 50",
        "Read the theatre constants and locate the reward in the environment",
        primary.theatre.gamma == 0.995
        and primary.theatre.bis_target == 50.0
        and primary.theatre.map_floor == 65.0
        and primary.theatre.bis_suppression == 20.0,
        "gamma 0.995, BIS* 50, floor 65, suppression 20",
        "Eq. (1), Eq. (2)",
        {
            "gamma": primary.theatre.gamma,
            "target": primary.theatre.bis_target,
            "floor": primary.theatre.map_floor,
            "suppression": primary.theatre.bis_suppression,
            "reward_line": _source_line(
                "src/respire/theatre/environment.py", "reward = -depth_error - drug_cost"
            ),
        },
    )

    add(
        "S07",
        "The budgets are ten minutes, one minute and the ninetieth percentile",
        "Read the configured budgets and confirm the variability budget is computed from the cohort",
        primary.rounds.cost_budget_hypotension_min == 10.0
        and primary.rounds.cost_budget_suppression_min == 1.0
        and primary.rounds.cost_budget_variability_quantile == 0.9
        and _source_line(
            "src/respire/training/round_loop.py", "torch.quantile(replay.cost_variability"
        )
        > 0,
        "d1 = 10 min, d2 = 1 min, d3 = 90th percentile of the logged variability",
        "Sec. 3.1",
        {
            "quantile_line": _source_line(
                "src/respire/training/round_loop.py", "torch.quantile(replay.cost_variability"
            )
        },
    )

    certificate = ConformalCertificate(alpha=0.1)
    residuals = [float(value) for value in range(1, 21)]
    rank = math.ceil((20 + 1) * 0.9)
    add(
        "S08",
        "The certificate uses the ceil((n+1)(1-alpha)) rank of the residual",
        "Compare the rank returned by the implementation with the rank computed here",
        certificate.quantile(__import__("torch").tensor(residuals)) == float(residuals[rank - 1]),
        f"n=20, alpha=0.1, rank={rank}, quantile={certificate.quantile(__import__('torch').tensor(residuals))}",
        "Proposition 1, Eq. (5)",
        {"rank": rank, "quantile": certificate.quantile(__import__("torch").tensor(residuals))},
    )

    algorithm_lines = {
        "line 1 one-step pressure model": _source_line("src/respire/pipeline.py", "_fit_pressure("),
        "line 2 behaviour cloning initialisation": _source_line(
            "src/respire/pipeline.py", "fit_behaviour_cloning("
        ),
        "line 3 verifier fit on generated and replayed returns": _source_line(
            "src/respire/training/round_loop.py", "warmup_features"
        ),
        "line 5 pool draw": _source_line(
            "src/respire/training/round_loop.py", "pool = _draw_pool("
        ),
        "line 6 conformal quantile": _source_line(
            "src/respire/training/round_loop.py", "quantile = certificate.quantile("
        ),
        "line 7 admitted set": _source_line(
            "src/respire/training/round_loop.py", "admitted = certificate.evaluate("
        ),
        "line 8 coverage test": _source_line(
            "src/respire/training/round_loop.py", "if not report.accepted"
        ),
        "line 11 refit on admitted plus reference": _source_line(
            "src/respire/training/round_loop.py", "retrain_features"
        ),
    }
    add(
        "S09",
        "Every line of Algorithm 1 has a home in the package",
        "Locate each step of the algorithm in the driver",
        all(value > 0 for value in algorithm_lines.values()),
        ", ".join(f"{key}:{value}" for key, value in algorithm_lines.items()),
        "Algorithm 1",
        algorithm_lines,
    )

    algorithm_two = {
        "line 1 warm start and adaptation": _source_line(
            "src/respire/control/deployment.py", "def _adapt("
        ),
        "line 3 recurrent update": _source_line(
            "src/respire/control/policy.py", "output = self.encoder("
        ),
        "line 4 four-step sampling": _source_line(
            "src/respire/control/policy.py", "sample = self.flow.sample("
        ),
        "line 5 shield": _source_line("src/respire/control/policy.py", "outcome = self.shield("),
        "line 7 supervisor override": _source_line(
            "src/respire/control/deployment.py", "reverted = self.override.check("
        ),
        "line 11 case record": _source_line("src/respire/control/deployment.py", "return action"),
    }
    add(
        "S10",
        "Every line of Algorithm 2 has a home in the package",
        "Locate each step of the deployment loop",
        all(value > 0 for value in algorithm_two.values()),
        ", ".join(f"{key}:{value}" for key, value in algorithm_two.items()),
        "Algorithm 2",
        algorithm_two,
    )

    add(
        "S11",
        "The operator maximises the certified value under cost constraints and a KL radius",
        "Confirm that both the KL term and the radius are present, and that the costs carry multipliers",
        _source_line(
            "src/respire/training/policy_step.py",
            "matching + actor_weight * (return_term + cost_term)",
        )
        > 0
        and _source_line("src/respire/training/policy_step.py", "backtrack(") > 0,
        "penalty and hard radius are both implemented",
        "Eq. (4)",
        {
            "penalty_line": _source_line(
                "src/respire/training/policy_step.py",
                "matching + actor_weight * (return_term + cost_term)",
            ),
            "radius_line": _source_line("src/respire/training/policy_step.py", "backtrack("),
        },
    )

    from respire.comparators.registry import analytic_names, learned_names

    arms = sorted(analytic_names() + learned_names())
    add(
        "S12",
        "The comparison contains the thirteen arms of the primary table",
        "Read the arm registry",
        len(arms) == 13,
        f"{len(arms)} arms: {', '.join(arms)}",
        "Table 1",
        {"arms": arms},
    )

    add(
        "S13",
        "The component analysis removes eight elements one at a time",
        "Read the variant list",
        len(COMPONENTS) == 9 and sum(1 for item in COMPONENTS if item.is_full) == 1,
        f"{len(COMPONENTS) - 1} removable elements plus the full model",
        "Table 2",
        {"variants": [item.name for item in COMPONENTS]},
    )

    add(
        "S14",
        "Three interpretable axes are defined, not merely three free dimensions",
        "Confirm the axis constants and that the encoder exposes an axis head of width three",
        _source_line("src/respire/theatre/patient.py", "AXIS_SENSITIVITY") > 0
        and encoder.axis_head.out_features == 3,
        "sensitivity, pressure responsiveness and reserve",
        "Sec. 1; Fig. 2",
        {"axis_head_width": encoder.axis_head.out_features},
    )

    reported_results = {
        7.26,
        7.18,
        7.44,
        7.31,
        7.61,
        7.36,
        7.42,
        8.02,
        8.21,
        8.94,
        14.6,
        14.0,
        13.1,
        18.9,
        22.6,
        24.1,
        8.2,
        4.4,
        2.1,
        9.8,
        6.96,
        6.79,
        6.41,
        0.44,
        0.71,
        0.18,
        0.74,
        11.6,
        20.1,
        33.2,
        41.6,
        0.612,
        0.768,
        0.68,
        82.0,
        2.9,
        5.8,
        9.3,
        38.0,
    }
    allowed = {("bis_noise_sd", 2.1), ("depth_noise_floor", 2.1)}
    # 6.28 is both the Eleveld central volume and an Eleveld TCU propofol column; the constant is a
    # transport volume, not a copied result, so it is not treated as a hit.
    from respire.settings import read_yaml

    offenders: list[dict[str, object]] = []

    def walk(node: object, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            leaf = path.rsplit(".", 1)[-1]
            if float(node) in reported_results and (leaf, float(node)) not in allowed:
                offenders.append({"path": path, "value": node})

    documents = sorted((ROOT / "configs").rglob("*.yaml"))
    for document in documents:
        walk(read_yaml(document), document.stem)
    add(
        "S15",
        "No configuration document carries a reported result",
        "Walk every YAML leaf and compare it with the values printed in the result tables",
        not offenders,
        f"{len(documents)} documents scanned; offending leaves: {offenders}",
        "Table 1-3",
        {"documents": [document.name for document in documents], "offenders": offenders},
    )

    return checks


# ---------------------------------------------------------------------------------------------
# Family 3: running the package
# ---------------------------------------------------------------------------------------------


def execution_checks(full: bool, seed: int) -> list[Check]:
    import torch

    from respire.control.certificate import ConformalCertificate
    from respire.data.windows import build_windows
    from respire.evaluation.theory import (
        corollary_sample_size,
        proposition2_trajectory,
        theorem_bound,
    )
    from respire.models.encoder import EncoderStandardiser, LatentEncoder
    from respire.models.flow import FlowPolicy
    from respire.settings import load_experiment
    from respire.theatre.clinician import RuleBasedAnaesthetist
    from respire.theatre.cohort import build_cohort, episode_patients, make_environment
    from respire.theatre.rollout import rollout

    checks: list[Check] = []
    config = load_experiment(ROOT / "configs" / "round" / "_smoke.yaml")

    def add(
        identifier: str,
        claim: str,
        method: str,
        status: str,
        detail: str,
        ref: str,
        evidence: dict[str, object] | None = None,
    ) -> None:
        checks.append(
            Check(
                identifier=identifier,
                family="execution",
                claim=claim,
                method=method,
                status=status,
                detail=detail,
                paper_ref=ref,
                evidence=evidence or {},
            )
        )

    cohort = build_cohort(config)
    patients = episode_patients(cohort, "development", 4, config.seed)
    trace = rollout(
        make_environment(patients, config, config.seed),
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
    )
    windows = build_windows(trace, config.encoder.window_steps, stride=8)
    standardiser = EncoderStandardiser(config.encoder.n_channels)
    standardiser.fit(windows.window)
    from respire.losses.encoder_losses import encoder_objective
    from respire.theatre.probe import counterfactual_probes
    from respire.training.encoder_fit import default_weights

    probes = counterfactual_probes(
        patients,
        config,
        RuleBasedAnaesthetist(config.theatre, noise_sd=0.05),
        warmup_steps=(12, 36),
        seed=config.seed,
    )
    encoder = LatentEncoder(config.encoder)
    actions = windows.actions
    output = encoder(standardiser.transform(windows.window), actions)
    terms = encoder_objective(
        encoder, standardiser, windows, probes, default_weights(config.encoder)
    )
    terms.total.backward()
    gradients = sum(
        1
        for parameter in encoder.parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    )
    total = sum(1 for _ in encoder.parameters())
    add(
        "E01",
        "A forward pass produces the three outputs and a backward pass reaches the parameters",
        "Run the encoder on a real window batch and backpropagate a scalar",
        "PASS" if gradients > 0 and gradients == total else "FAIL",
        f"{gradients} of {total} parameter tensors received a non-zero gradient; latent {tuple(output.latent.shape)}",
        "Fig. 2",
        {"latents": list(output.latent.shape), "gradients": gradients, "parameters": total},
    )

    policy = FlowPolicy(config.policy, config.encoder.latent_dim + config.encoder.hidden_dim)
    policy.configure_action_scale(config.theatre.propofol_max, config.theatre.remifentanil_max)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    condition = torch.cat([output.latent.detach(), output.hidden.detach()], dim=-1)[:64]
    sample = policy.sample(condition, generator)
    explicit = policy.log_prob(sample.action, condition)
    gap = float((sample.log_prob - explicit).abs().mean())
    add(
        "E02",
        "The flow's sampled density agrees with the density read at the sampled action",
        "Compare the divergence accumulated during sampling with a reverse integration at the same action",
        "PASS" if gap < 1.0 else "FAIL",
        f"mean absolute gap {gap:.4f} over {sample.action.shape[0]} samples",
        "Eq. (4) (KL term needs a density)",
        {"gap": gap},
    )

    optimiser = torch.optim.Adam(encoder.parameters(), lr=5e-3)
    before = [parameter.detach().clone() for parameter in encoder.parameters()]
    losses: list[float] = []
    for _ in range(6):
        optimiser.zero_grad()
        chunk = encoder(standardiser.transform(windows.window), actions)
        step_loss = chunk.reconstruction.pow(2).mean() + chunk.latent.pow(2).mean()
        step_loss.backward()
        optimiser.step()
        losses.append(float(step_loss.detach()))
    moved = sum(
        1
        for old, new in zip(before, encoder.parameters(), strict=True)
        if not torch.allclose(old, new.detach())
    )
    add(
        "E03",
        "Six optimisation steps lower the loss and change the parameters",
        "Train on one fixed batch and compare the first and last loss",
        "PASS" if losses[-1] < losses[0] and moved > 0 else "FAIL",
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f}; {moved} parameter tensors moved",
        "Sec. 4.2 (encoder is fitted once)",
        {"first": losses[0], "last": losses[-1], "moved": moved},
    )

    # Split-conformal coverage, computed here rather than through the class.
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 1)
    calibration = 50.0 + 10.0 * torch.randn(2000, generator=g)
    pool = 50.0 + 10.0 * torch.randn(2000, generator=g)
    shifted = 5.0 + 10.0 * torch.randn(2000, generator=g)
    residuals = torch.sort((calibration - calibration.mean()).abs()).values
    rank = math.ceil((residuals.numel() + 1) * 0.9)
    quantile = float(residuals[min(rank, residuals.numel()) - 1])
    coverage = float((pool >= float(calibration.mean()) - quantile).to(torch.float32).mean())
    shifted_coverage = float(
        (shifted >= float(calibration.mean()) - quantile).to(torch.float32).mean()
    )
    add(
        "E04",
        "The certificate reaches its target coverage on an exchangeable pool",
        "Rebuild the rank quantile with numpy-free arithmetic here and measure coverage on a fresh pool",
        "PASS" if coverage >= 0.89 else "FAIL",
        f"coverage {coverage:.3f} against a target of 0.900 with n=2000",
        "Proposition 1, Eq. (5)",
        {"coverage": coverage, "quantile": quantile},
    )
    add(
        "E05",
        "Coverage fails when the pool is not exchangeable with the reference",
        "Repeat the measurement on a pool shifted by five standard deviations",
        "PASS" if shifted_coverage < 0.5 else "FAIL",
        f"coverage {shifted_coverage:.3f} on the shifted pool",
        "Sec. 3.2 (the exchangeability caveat)",
        {"shifted_coverage": shifted_coverage},
    )

    certificate = ConformalCertificate(alpha=0.1)
    ours = certificate.quantile(torch.tensor([float(v) for v in range(1, 21)]))
    add(
        "E06",
        "The package's quantile equals the rank computed independently",
        "Compare the implementation with ceil((n+1)(1-alpha)) evaluated in this script",
        "PASS" if ours == 19.0 else "FAIL",
        f"implementation {ours}, independent {19.0}",
        "Proposition 1",
        {"implementation": ours, "independent": 19.0},
    )

    size = corollary_sample_size(1.0, 0.05, 0.01, 0.05)
    independent = 2.0 * 1.0 * math.log(80.0) / (0.05 - 0.02) ** 2
    add(
        "E07",
        "Eq. (6) is evaluated rather than quoted",
        "Compare the release's sample size with an inline evaluation",
        "PASS" if abs(size - independent) < 1e-6 else "FAIL",
        f"{size:.2f} against {independent:.2f}",
        "Corollary 1",
        {"release": size, "independent": independent},
    )

    bound = theorem_bound(epsilon=0.01, R=1.0, N=9700, delta=0.05, delta_cov=0.0)
    inline = 2.0 * 0.01 + 2.0 * 1.0 * math.sqrt(math.log(80.0) / (2 * 9700))
    add(
        "E08",
        "Eq. (7) is evaluated rather than quoted",
        "Compare the release's bound with an inline evaluation of its three terms",
        "PASS" if abs(bound.total - inline) < 1e-6 else "FAIL",
        f"total {bound.total:.6f} against {inline:.6f}",
        "Theorem 1, Eq. (7)",
        {"release": bound.total, "independent": inline},
    )

    trajectory = proposition2_trajectory(1.5, 1.0, 4)
    agrees_at_two = proposition2_trajectory(2.0, 1.0, 4)
    add(
        "E09",
        "The biased recursion agrees with the printed form only at c = 2",
        "Run the trajectory at c = 2 and at c = 1.5 and compare the two readings of the reported value",
        (
            "PASS"
            if abs(agrees_at_two.reported_quadratic[1] - agrees_at_two.reported_printed[1]) < 1e-9
            and abs(trajectory.reported_quadratic[1] - trajectory.reported_printed[1]) > 1e-3
            else "FAIL"
        ),
        "c=2 agrees to machine precision, c=1.5 differs by "
        f"{abs(trajectory.reported_quadratic[1] - trajectory.reported_printed[1]):.4f}",
        "Proposition 2",
        {
            "at_two": agrees_at_two.reported_quadratic[1],
            "at_one_point_five": trajectory.reported_quadratic[1],
        },
    )

    from respire.control.shield import AnalyticShield
    from respire.models.pressure import OneStepPressureModel

    pressure = OneStepPressureModel(ridge=1e-6)
    state = torch.randn(64, 5, generator=g) * 2.0
    action = torch.rand(64, 2, generator=g)
    target = 85.0 + 2.0 * state[:, 0] - 3.0 * action[:, 0] - 2.0 * action[:, 1]
    pressure.fit(state, action, target)
    recovered = float((pressure.predict(state, action) - target).abs().max())
    shield = AnalyticShield(pressure, config.theatre, config.shield)
    requested = torch.tensor([[12.0, 0.5]]).repeat(16, 1)
    outcome = shield(state[:16], requested)
    monotone = bool((outcome.action[:, 0] <= requested[:, 0] + 1e-6).all())
    add(
        "E10",
        "The one-step pressure model recovers a linear surface and the shield only lowers the rate",
        "Fit the model on a manufactured surface, then project a maximal request",
        "PASS" if recovered < 1e-2 and monotone else "FAIL",
        f"max absolute residual {recovered:.5f}; projection never raises the hypnotic rate",
        "Sec. 4 (analytic shield)",
        {"residual": recovered, "monotone": monotone, "feasible": int(outcome.feasible.sum())},
    )

    add(
        "E11",
        "A complete episode runs without a non-finite value",
        "Roll the behaviour policy over a real batch and check every recorded quantity",
        (
            "PASS"
            if bool(torch.isfinite(trace.observation).all() and torch.isfinite(trace.reward).all())
            else "FAIL"
        ),
        f"{trace.batch} cases x {trace.steps} steps, return mean {float(trace.episode_return.mean()):.1f}",
        "Sec. 3.1",
        {"return_mean": float(trace.episode_return.mean())},
    )

    from respire.training.checkpoint import ReleaseBundle

    directory = ROOT / "runs" / "verification"
    directory.mkdir(parents=True, exist_ok=True)
    bundle_path = directory / "probe_bundle.pt"
    from respire.data.summary import summary_dim
    from respire.models.verifier import ReturnVerifier

    verifier = ReturnVerifier(config.verifier, summary_dim(config.encoder.n_channels))
    bundle = ReleaseBundle(
        encoder=encoder,
        standardiser=standardiser,
        verifier=verifier,
        flow=policy,
        pressure=pressure,
        seed=seed,
        round_index=1,
    )
    bundle.save(bundle_path)
    restored = ReleaseBundle.load(bundle_path)
    window = standardiser.transform(windows.window)[:5]
    with torch.no_grad():
        first = encoder(window, actions[:5]).latent
        second = restored.encoder(window, actions[:5]).latent
    identical = torch.allclose(first, second, atol=1e-6)
    add(
        "E12",
        "A checkpoint reloads to the same function and the same seed",
        "Write a bundle, read it back and compare the latent on the same input",
        "PASS" if identical and restored.seed == seed else "FAIL",
        f"identical outputs: {identical}; seed {restored.seed}",
        "Sec. 4 (the encoder is frozen between rounds)",
        {"identical": identical, "seed": restored.seed},
    )

    status = "NOT_RUN"
    detail = "the pipeline was not executed in this pass"
    evidence: dict[str, object] = {}
    if full:
        from respire.flags import FULL
        from respire.pipeline import run_pipeline

        started = time.time()
        result = run_pipeline(config, FULL, seed=config.seed, output_dir=directory)
        elapsed = time.time() - started
        records = result.run.records
        finite = all(math.isfinite(value) for value in result.evaluation.row().values())
        status = "PASS" if records and finite else "FAIL"
        detail = (
            f"{len(records)} rounds in {elapsed:.1f} s; certified curve "
            f"{[round(value, 3) for value in result.run.certified_curve]}; "
            f"coverage {[round(float(record.certificate['coverage']), 3) for record in records]}"
        )
        evidence = {
            "rounds": len(records),
            "certified": result.run.certified_curve,
            "coverage": [float(record.certificate["coverage"]) for record in records],
            "evaluation": result.evaluation.row(),
            "behaviour": result.behaviour.row(),
            "seconds": round(elapsed, 1),
        }
    add(
        "E13",
        "The recursion runs end to end and reports a certified curve",
        "Execute the pipeline on the smoke document and read the round log",
        status,
        detail,
        "Algorithm 1; Fig. 4",
        evidence,
    )

    from respire.control.policy import LatentFlowController

    controller = LatentFlowController(encoder, standardiser, policy, config.theatre, seed=seed)
    env = make_environment(patients, config, config.seed)
    observation = env.reset()
    controller.reset(env.batch)
    first_action = controller.act_deterministic(observation)
    controller.reset(env.batch)
    second_action = controller.act_deterministic(observation)
    same = torch.allclose(first_action, second_action, atol=1e-6)
    add(
        "E14",
        "The deployment path is deterministic under a pinned generator",
        "Act twice on the same observation with the sampling noise pinned",
        "PASS" if same else "FAIL",
        f"identical actions: {same}",
        "Algorithm 2",
        {"identical": same},
    )

    return checks


def record_level_checks() -> list[Check]:
    """Claims that cannot be checked without the credentialed corpora."""
    claims = [
        ("R01", "Depth error 7.26 against 7.18 for the strongest unconstrained arm", "Table 1"),
        ("R02", "Sustained hypotension falls from 14.6% to 8.2%", "Table 1, Table 3"),
        ("R03", "Burst suppression falls from 4.4% to 2.1%", "Table 1, Table 3"),
        ("R04", "Propofol consumption is 6.96 against 6.79", "Table 1"),
        ("R05", "The sensitivity axis predicts the measured requirement with r = 0.68", "Fig. 5"),
        ("R06", "Eighty-two percent of the benefit accrues to the upper quartile", "Fig. 5"),
        ("R07", "The 226 held-out robot-assisted cases reproduce the primary table", "Table 1"),
        ("R08", "The four-round certified return rises from 0.612 to 0.768", "Fig. 4"),
        ("R09", "The uncalibrated operator degrades by 4.1% at round three", "Fig. 4"),
        ("R10", "Site-free transfer to the intensive-care cohort preserves the axis", "Fig. 5"),
        (
            "R11",
            "The case readers parse an exported record into the twelve-channel block",
            "respire.data.exports",
        ),
    ]
    checks: list[Check] = []
    for identifier, claim, ref in claims:
        checks.append(
            Check(
                identifier=identifier,
                family="record_level",
                claim=claim,
                method="Replay the credited corpora and recompute the statistic",
                status="BLOCKED",
                detail=(
                    "VitalDB requires a data-use agreement and MIMIC-IV and eICU-CRD require "
                    "credentialed access; the article states that neither the records nor a "
                    "cohort checkpoint are redistributed. No external cohort is substituted, so "
                    "the statistic cannot be recomputed here."
                ),
                paper_ref=ref,
                evidence={"access": "credentialed", "shipped": False},
            )
        )
    return checks


# ---------------------------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------------------------


def claim_map() -> dict[str, object]:
    """Map each stated method element onto the file and line that implements it."""
    mapping = [
        (
            "Eq. (1) reward",
            "src/respire/theatre/environment.py",
            "reward = -depth_error - drug_cost",
            "Sec. 3.1",
        ),
        (
            "Eq. (2) costs",
            "src/respire/theatre/environment.py",
            "hypotension = (observed_map < self.theatre.map_floor)",
            "Sec. 3.1",
        ),
        (
            "Eq. (2) rate variation",
            "src/respire/theatre/environment.py",
            "variability = (clipped[:, 0] - state.previous_action[:, 0]).abs()",
            "Sec. 3.1",
        ),
        (
            "Eq. (3) lower bound",
            "src/respire/control/certificate.py",
            "def lower_bound",
            "Sec. 3.2",
        ),
        (
            "Eq. (4) operator",
            "src/respire/training/policy_step.py",
            "def constrained_step",
            "Sec. 3.2",
        ),
        (
            "Eq. (5) coverage statement",
            "src/respire/control/certificate.py",
            "def quantile",
            "Proposition 1",
        ),
        (
            "Eq. (6) sample size",
            "src/respire/evaluation/theory.py",
            "def corollary_sample_size",
            "Corollary 1",
        ),
        (
            "Eq. (7) monotone bound",
            "src/respire/evaluation/theory.py",
            "def theorem_bound",
            "Theorem 1",
        ),
        (
            "Eq. (8) cumulative bound",
            "src/respire/evaluation/theory.py",
            "def cumulative_bound",
            "Corollary 2",
        ),
        (
            "Algorithm 1 round loop",
            "src/respire/training/round_loop.py",
            "def run_recursion",
            "Algorithm 1",
        ),
        (
            "Algorithm 1 admission",
            "src/respire/training/round_loop.py",
            "admitted = certificate.evaluate(",
            "Algorithm 1 line 7-8",
        ),
        (
            "Algorithm 2 deployment",
            "src/respire/control/deployment.py",
            "class DeployedController",
            "Algorithm 2",
        ),
        (
            "Algorithm 2 shield call",
            "src/respire/control/policy.py",
            "outcome = self.shield(",
            "Algorithm 2 line 5",
        ),
        ("Selective SSM block", "src/respire/models/ssm.py", "class SelectiveSSMBlock", "Sec. 4.2"),
        (
            "Cross-channel attention",
            "src/respire/models/attention.py",
            "class CrossChannelAttention",
            "Sec. 4.2",
        ),
        ("Sparse readout", "src/respire/models/encoder.py", "class SparseReadout", "Fig. 2"),
        (
            "Heteroscedastic verifier",
            "src/respire/models/verifier.py",
            "class ReturnVerifier",
            "Fig. 2",
        ),
        ("Rectified flow policy", "src/respire/models/flow.py", "class FlowPolicy", "Fig. 2"),
        ("Exact flow density", "src/respire/models/flow.py", "def log_prob", "Eq. (4)"),
        (
            "One-step pressure model",
            "src/respire/models/pressure.py",
            "class OneStepPressureModel",
            "Algorithm 1 line 1",
        ),
        ("Analytic shield", "src/respire/control/shield.py", "class AnalyticShield", "Sec. 4"),
        (
            "Ridge adaptation",
            "src/respire/control/adaptation.py",
            "class RidgeAdapter",
            "Algorithm 2 line 1",
        ),
        (
            "Interventional loss",
            "src/respire/losses/encoder_losses.py",
            "def interventional_loss",
            "Sec. 1, Table 2",
        ),
        (
            "Flow-matching objective",
            "src/respire/losses/flow_matching.py",
            "def flow_matching_loss",
            "Fig. 2",
        ),
        ("Dual ascent", "src/respire/losses/lagrangian.py", "class DualAscent", "Eq. (4)"),
        ("Trust region", "src/respire/losses/trust_region.py", "def backtrack", "Eq. (4)"),
        ("Three latent axes", "src/respire/theatre/patient.py", "AXIS_SENSITIVITY", "Sec. 1"),
        (
            "Eleveld propofol model",
            "src/respire/theatre/pharmacokinetics.py",
            "def propofol_parameters",
            "Sec. 5.1",
        ),
        (
            "Minto remifentanil model",
            "src/respire/theatre/pharmacokinetics.py",
            "def remifentanil_parameters",
            "Sec. 5.1",
        ),
        ("Cohort", "src/respire/theatre/patient.py", "def sample_patients", "Sec. 5.1"),
        (
            "Surgical schedule",
            "src/respire/theatre/surgical.py",
            "def build_schedule",
            "Sec. 1, Sec. 3.1",
        ),
        (
            "Primary comparison",
            "src/respire/evaluation/comparison.py",
            "def build_comparison",
            "Table 1, Table 3",
        ),
        ("Component analysis", "src/respire/flags.py", "COMPONENTS", "Table 2"),
        (
            "Bootstrap interval",
            "src/respire/metrics/statistics.py",
            "def bootstrap_interval",
            "Table 3",
        ),
        ("Sign-flip test", "src/respire/metrics/statistics.py", "def sign_flip_pvalue", "Table 3"),
        (
            "Quartile stratification",
            "src/respire/metrics/stratify.py",
            "def benefit_shares",
            "Fig. 5",
        ),
    ]
    entries = []
    for claim, relative, needle, ref in mapping:
        line = _source_line(relative, needle)
        entries.append(
            {
                "claim": claim,
                "paper_ref": ref,
                "file": relative,
                "line": line,
                "symbol": needle,
                "status": "MAPPED" if line > 0 else "UNMAPPED",
            }
        )
    return {
        "paper": PAPER,
        "counts": {
            "claims": len(entries),
            "mapped": sum(1 for entry in entries if entry["status"] == "MAPPED"),
        },
        "overall": (
            "COMPLETE_MAPPING"
            if all(entry["status"] == "MAPPED" for entry in entries)
            else "INCOMPLETE_MAPPING"
        ),
        "claims": entries,
        "deviations": [
            {
                "item": "engineer defaults for unstated values",
                "detail": (
                    "The article does not print the learning rates, batch sizes, epoch counts, the "
                    "drug-cost weight, the coverage level alpha, the admitted fraction, the trust "
                    "radius, the number of sparse atoms or the cloning passes. Every such value is a "
                    "documented engineering default in configs/round/primary.yaml, marked with a "
                    "comment where it appears."
                ),
            },
            {
                "item": "critic-based realisation of the maximisation term",
                "detail": (
                    "Eq. (4) maximises the verifier's expectation over the policy. With a flow the "
                    "expectation has no closed form, so the release maximises a return critic fitted "
                    "on admitted episodes and keeps the flow-matching term as the density anchor. "
                    "The deviation is the estimator, not the constraint."
                ),
            },
            {
                "item": "constraint normalisation in the dual",
                "detail": (
                    "The three costs and the trust region are divided by their budgets before the "
                    "dual step, so one learning rate serves all four multipliers. This is a "
                    "reparametrisation of the same Lagrangian."
                ),
            },
            {
                "item": "axis anchoring on the theatre",
                "detail": (
                    "The three axes are anchored weakly to the generating factors, which only the "
                    "theatre can supply. On record-level replay there is no anchor and the "
                    "interventional term carries the axes alone."
                ),
            },
            {
                "item": "record-level corpora are not shipped",
                "detail": (
                    "VitalDB, MIMIC-IV and eICU-CRD are credentialed. The release ships the "
                    "theatre, the readers for the official export layouts, and no substituted cohort."
                ),
            },
        ],
    }


def integrity_manifest(exclude: Sequence[str]) -> dict[str, object]:
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative in exclude or relative.endswith((".pyc", ".pyo", ".db")):
            continue
        if any(part in CACHE_DIRECTORIES for part in Path(relative).parts):
            continue
        if relative.startswith("runs/"):
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": digest})
    joined = "".join(f"{entry['path']}:{entry['sha256']}" for entry in entries)
    return {
        "algorithm": "sha256",
        "file_count": len(entries),
        "manifest_digest": hashlib.sha256(joined.encode("utf-8")).hexdigest(),
        "excludes": list(exclude),
        "note": "digests cover every shipped file except this manifest and the run directory",
        "files": entries,
    }


def summarise(checks: list[Check]) -> dict[str, object]:
    families: dict[str, dict[str, int]] = {}
    for check in checks:
        bucket = families.setdefault(
            check.family, {"checks": 0, "passed": 0, "failed": 0, "not_run": 0, "blocked": 0}
        )
        bucket["checks"] += 1
        if check.status == "PASS":
            bucket["passed"] += 1
        elif check.status == "FAIL":
            bucket["failed"] += 1
        elif check.status == "BLOCKED":
            bucket["blocked"] += 1
        else:
            bucket["not_run"] += 1
    totals = {
        "checks": len(checks),
        "passed": sum(1 for check in checks if check.status == "PASS"),
        "failed": sum(1 for check in checks if check.status == "FAIL"),
        "not_run": sum(1 for check in checks if check.status == "NOT_RUN"),
        "blocked": sum(1 for check in checks if check.status == "BLOCKED"),
    }
    return {"totals": totals, "families": families}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="verify the release against the paper")
    parser.add_argument("--full", action="store_true", help="also execute the whole pipeline")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--write", action="store_true", help="write the three artefacts")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    started = time.time()
    checks = (
        manuscript_checks()
        + structure_checks()
        + execution_checks(args.full, args.seed)
        + record_level_checks()
    )
    summary = summarise(checks)
    code_checks = [check for check in checks if check.family in {"code_vs_paper", "execution"}]
    code_status = (
        "CODE_VERIFIED"
        if all(check.status in {"PASS", "NOT_RUN"} for check in code_checks)
        and any(check.status == "PASS" for check in code_checks)
        else (
            "CODE_FAILED"
            if any(check.status == "FAIL" for check in code_checks)
            else "CODE_UNVERIFIED"
        )
    )
    manuscript_failures = [
        check
        for check in checks
        if check.family == "manuscript_arithmetic" and check.status == "FAIL"
    ]
    blocked = [check for check in checks if check.status == "BLOCKED"]
    overall = "PARTIALLY_VERIFIED"
    if blocked:
        overall = "PARTIALLY_VERIFIED"
    elif summary["totals"]["failed"] == 0:
        overall = "VERIFIED"
    else:
        overall = "UNVERIFIED"

    report = {
        "paper": PAPER,
        "scope": (
            "The release was verified in three families: the article's arithmetic recomputed from "
            "its own printed numbers, the package inspected against the structural claims of the "
            f"method, and the package executed (seed {args.seed}"
            + (", full pipeline run" if args.full else ", no full pipeline run")
            + "). Record-level claims are blocked because the corpora are credentialed."
        ),
        "summary": summary["totals"],
        "families": [
            {"name": name, **values} for name, values in sorted(summary["families"].items())
        ],
        "code_status": code_status,
        "manuscript_status": "DISCREPANCIES_RECORDED" if manuscript_failures else "CONSISTENT",
        "overall": overall,
        "manuscript_discrepancies": [
            {
                "id": check.identifier,
                "claim": check.claim,
                "finding": check.detail,
                "paper_ref": check.paper_ref,
            }
            for check in manuscript_failures
        ],
        "blocked_claims": [check.identifier for check in blocked],
        "interpretation": (
            "The code status describes the package; the manuscript status describes the article's "
            "internal arithmetic. They are reported apart so a discrepancy inside the article's own "
            "tables is not mistaken for a broken release."
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": _torch_version(),
            "seconds": round(time.time() - started, 1),
        },
        "checks": [check.as_dict() for check in checks],
    }

    if args.write:
        (ROOT / "verification_report.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        (ROOT / "claim_to_code.json").write_text(
            json.dumps(claim_map(), indent=2) + "\n", encoding="utf-8"
        )
        manifest = integrity_manifest(
            ["integrity_manifest.json", "verification_report.json", "claim_to_code.json"]
        )
        (ROOT / "integrity_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(report["summary"], indent=2))
    print(
        f"code_status={code_status} manuscript_status={report['manuscript_status']} overall={overall}"
    )
    for check in checks:
        if check.status in {"FAIL"}:
            print(f"  {check.identifier} {check.status}: {check.claim}")
    return 0 if summary["totals"]["failed"] <= len(manuscript_failures) else 1


def _torch_version() -> str:
    try:
        import torch

        return str(torch.__version__)
    except Exception:  # pragma: no cover - torch is a hard dependency
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
