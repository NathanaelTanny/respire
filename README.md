# ReSPIRe: a closed-loop anaesthesia controller that rewrites itself

ReSPIRe couples three things that are usually built apart. A **state-space encoder** reads the
twelve intraoperative channels over a sixty-second window and returns a sixteen-dimensional latent
plus three named axes: how strongly this patient responds to the hypnotic, how strongly the
pressure loop responds to depth, and how much circulatory reserve is left. A **verifier** scores a
whole episode and is calibrated on a fresh reference split every round, so the score it returns is
a one-sided lower bound rather than a point estimate. A **flow-matching policy** over the two
infusion rates is updated inside a KL trust region under three safety budgets, and an **analytic
shield** sits between the policy and the pump. The recursion is the loop that ties them together: a
pool of episodes is drawn from the current policy, certified, partly admitted, and used to refit the
verifier and to reweight the policy's own objective.

The release is the code that accompanies the manuscript. It contains the theatre the article's
cohort is drawn from, the full recursion, the deployment wrapper, thirteen comparison arms, the
evaluation that produces the two tables, and the arithmetic of every closed-form statement the
article makes a claim about.

## What runs, and what it runs on

Two paths exist and they are deliberately separate.

The **cohort** path needs nothing but this repository. Two thousand patients are drawn from
the covariate distributions of the published propofol and remifentanil models, each with its own
latent factors, and the theatre around them is generated from the surgical schedule: insufflation
to twelve millimetres of mercury, reverse Trendelenburg, stimulation trains, blood loss and the
relaxant boluses. This path exercises every component end to end.

The **record** path needs the corpora, which are credentialed and cannot be redistributed. The
readers in `respire.data.exports` take one case per file and map its columns onto the twelve
channels through the schemas at the top of that module; the access routes are in
`dataset_urls.txt`. Nothing in the release fabricates a record to stand in for them, so a missing
corpus is reported as blocked rather than filled.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

Conda and Docker are equivalent:

```bash
conda env create -f environment.yml && conda activate respire
docker build -t respire .
```

The package is not published; `pip install -e .` from the repository root is the intended route.
All commands below assume `src` is importable, either through the editable install or with
`PYTHONPATH=src`.

## The two documents

`configs/round/primary.yaml` carries every value the article prints: the five-second cadence, the
hundred-and-twenty-minute episodes, the target index of fifty, the pressure floor of sixty-five, the
suppression threshold of twenty, the discount of 0.995, the twelve channels, the six selective
blocks, the sixteen-dimensional latent, the three axes, the four Euler steps, the two thousand
patients, the inflation of individual variability by 1.25, the twenty thousand training episodes,
the two thousand evaluation episodes, the two hundred and twenty-six held-out cases, the five
initialisations, the ten-minute and one-minute budgets and the ninetieth percentile.

`configs/round/_smoke.yaml` exists so the test suite and a first run finish in seconds. It changes
the scale, never the shape: same twelve channels, same window, same six blocks, same four Euler
steps. It is marked in the file and must not be used for reporting.

Every value that the article does not print, namely learning rates, batch sizes, epoch counts, the drug
cost weight, the coverage level, the admitted fraction and the trust radius, is a documented
engineering default, marked with a comment at the place it appears, and listed in
`verification_report.json` under `claim_to_code.json`'s deviations.

## Running the recursion

```bash
PYTHONPATH=src python -m respire.cli.cohort --config configs/round/primary.yaml
PYTHONPATH=src python -m respire.cli.train    --config configs/round/primary.yaml
```

`cohort` writes the cohort and the behaviour pool to `runs/cohort/`. `train` fits the encoder,
initialises the policy by cloning, fits the verifier on the first pool, runs the rounds and writes
`runs/rounds/bundle.pt` together with `rounds.json`, which holds the certificate of every round, the
accepted and rejected flag, the measured cost of each constraint and the policy step's own
diagnostics.

To see the mechanism at the scale of a first run, use the smoke document:

```bash
PYTHONPATH=src python -m respire.cli.train --config configs/round/_smoke.yaml --output /tmp/respire
```

That run takes about a minute on a laptop. A representative log at that scale reads

```
round  accepted  coverage  certified  pool
0      True      1.000     -1.012     -3587.8
1      True      0.917     -0.944     -3465.6
```

which says the certificate admitted both rounds and the certified value rose between them. The run
is deterministic: the same document and seed reproduce that table exactly. Those numbers are a smoke
run at reduced scale; they are not the article's results and are not reported as such anywhere in
this release.

## Reading the certificate

`respire.cli.certify` isolates the statistical claim from the controller. It draws a calibration set,
builds the rank quantile, and measures coverage twice: once on a pool from the same distribution and
once on a pool from a policy that ran shallower.

```bash
PYTHONPATH=src python -m respire.cli.certify --config configs/round/_smoke.yaml --repetitions 200
```

At the smoke scale the exchangeable arm covers at or above the target and the shifted arm falls far
below it, which is the exchangeability condition of Proposition 1 doing the work. When the target is
not reached the round loop rejects the round, keeps the previous policy and records the rejection;
`runs/rounds/rounds.json` keeps that decision per round, so a rejected round is visible rather than
silently dropped.

## Comparison and ablations

```bash
PYTHONPATH=src python -m respire.cli.compare --config configs/round/primary.yaml \
    --bundle runs/rounds/bundle.pt --arms schnider_tcu,eleveld_tcu,pid,internal_model,respire,clinician
PYTHONPATH=src python -m respire.cli.ablate  --config configs/round/primary.yaml
```

`compare` writes the primary table and the paired safety table: for each event, the two rates, the
difference, a percentile bootstrap interval over cases and a sign-flip p-value. `ablate` runs the
same pipeline once per removal and writes the component table. The learned arms need
`--iterations`; without it they are skipped rather than reported as zeros.

## Tests and verification

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/verify_release.py --full --write
ruff check . && black --check . && isort --check-only . && mypy --strict src/respire
```

The suite covers the transport model, the theatre, every model head, the certificate, the three
closed-form statements, every comparison arm, a single-batch overfit for both objectives, a
checkpoint round-trip and one end-to-end pipeline run.

`scripts/verify_release.py` writes three files. `claim_to_code.json` maps each stated method element
onto a file and line. `verification_report.json` holds every check with its status, split into the
article's own arithmetic, the package against the article's structure, and what actually executed;
record-level claims are marked blocked with the reason. `integrity_manifest.json` digests every
shipped file except itself.

## Compute

The theatre is the only expensive part and it is vectorised: a batch of patients advances in
lockstep through the same decision clock, so a two-thousand-patient cohort costs a few minutes per
hundred and twenty-minute episode on a single CPU core. The encoder is small by design: six blocks
of width sixty-four, and the verifier is deliberately held near four hundred thousand parameters.
A full five-round recursion at the article's scale is a GPU-scale job only because of the number of
episodes, not because of the model sizes. The release runs on CPU; no result in this repository was
produced on a GPU.

## Layout

```
configs/round/          experiment documents, primary and smoke
configs/theatre/        transport and depth surfaces, in one place
configs/encoder/        window, width and readout of the encoder
configs/controller/     flow head and shield
src/respire/theatre/    transport, surfaces, theatre, cohort, probes, clinician proxy
src/respire/data/       twelve-channel schema, windows, episode summaries, export readers
src/respire/models/     selective block, cross-channel attention, encoder, verifier, flow, critic
src/respire/losses/     flow matching, the four encoder terms, heteroscedastic NLL, duals, radius
src/respire/control/    certificate, shield, adaptation, deployed controller
src/respire/training/   the round loop and every fit
src/respire/comparators/ the thirteen arms
src/respire/evaluation/ the two tables, the recursion curve and the closed-form checks
src/respire/metrics/    depth, safety, resampling statistics, stratification
scripts/                launch helpers and the verification driver
tests/                  the suite
```

## Licence and provenance

The code is MIT. The transport and depth surfaces follow the published propofol and remifentanil
models named in `NOTICE`, whose parameter sets are cited there; the corpora are governed by their
own access agreements and are not redistributed here.
