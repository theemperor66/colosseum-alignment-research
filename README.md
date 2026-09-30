# Colosseum Alignment Research

**From embodied behaviour to evidence for ethical and safety arguments.**

Research software and evidence accompanying *Simulation-Based Alignment Research for Embodied Agents: A Colosseum Testbed for Autonomous UAVs in Civilian Defence*, by Zaid Marzguioui, Universität Regensburg. The manuscript develops a general simulation-based evaluation approach and examines it through a completed civilian UAV inspection study.

The central question is what a system's observed behaviour warrants us in claiming about it. This implementation separates the agent's observations, the guard's judgement, independently measured consequences, and what a later auditor can reconstruct. The separation makes disagreements inspectable: a monitor can endorse an unsafe episode, restraint can remove useful service, and a brief recovery can precede a later violation.

[Public data](data/README.md) · [Research design](docs/research-design.md) · [Reproducibility](docs/reproducibility.md) · [Citation](CITATION.cff) · [Contributing](CONTRIBUTING.md)

## Study at a glance

The completed design contains **792 selected episodes in 216 environments**, with a distinct **180-group primary comparison**. It uses a fixed, hand-written perception-based controller, three main guard configurations, and three additional configurations in the focused permission-constraint comparison. All 826 retained attempt records, including incomplete attempts, are accounted for in the manuscript's evidence package.

| Observation | Scope |
|---|---|
| The assumption-aware configuration has 7 physical violations, versus 76 for the policy-only configuration. | Descriptive totals over 216 episodes per main configuration. |
| Safe useful completions fall from 83 to 34. | Same 216-episode populations; restraint and service must be assessed together. |
| The policy-only guard accepts 161 episodes, including 48 with an independently established physical or procedural violation. | Acceptance and independent conformance are different measurements. |
| The assumption-aware guard accepts no episodes. | Its conditional false-assurance rate is undefined. |
| All seven assumption-aware physical-violation episodes first satisfy the short recovery endpoint. | Recovery within a specified horizon does not establish continued mission safety. |

These are findings within the selected simulated design. The primary physical contrast, uncertainty assumptions, follow-up selection and original sensitivity analysis are explained in the manuscript and [research design](docs/research-design.md). The results do not establish real-world incident rates, moral acceptability, learned-policy alignment or superiority across tasks.

## Architecture

```mermaid
flowchart TB
    P["Frozen scenario and protocol"] --> S["Colosseum 3D world"]
    S --> O["Delivered RGB, depth and state"]
    O --> C["Fixed controller"]
    O --> G["Runtime guard"]
    C --> G
    H["Synthetic authority and supervision"] --> G
    G --> D["Executed commands"]
    D --> S
    S --> T["Privileged sampled truth"]
    T --> E["Independent endpoint evaluator"]
    D --> R["Retained records"]
    O --> R
    G --> R
    R --> A["Restricted-view reconstruction"]
    E --> F["Conformance, service and evidence profile"]
    A --> F
```

Privileged truth is available to evaluation, not to the online controller or guard. Record ablations change the auditor's evidence while leaving the recorded flight unchanged. “Independent” describes this separation of execution and scoring; it does not imply an external laboratory or independent human review.

## Quick start: check the reported arithmetic

Release **v1.1.0** includes the compact evidence archive and browsable computed records for **all 792 selected episodes**. This check needs only Python's standard library; no download beyond the repository, GPU or simulator is required.

```bash
git clone https://github.com/theemperor66/colosseum-alignment-research.git
cd colosseum-alignment-research
git checkout v1.1.0
python3 scripts/verify_paper.py data/compact/evidence.zip
```

The verifier also accepts a freshly extracted evidence directory. It first verifies archive integrity, then checks design accounting, preservation of the 757 originally complete records, primary and focused contrasts, original sensitivity results, arm totals and derived-file hashes. It checks saved endpoint labels and their arithmetic; it does not reconstruct physical measurements from images or trajectories. A successful run reports `all_checks_passed: true`.

## Inspect an authentic recorded episode

The [v1.1.0 data release](https://github.com/theemperor66/colosseum-alignment-research/releases/tag/v1.1.0) also supplies **one complete A2 episode from F4, realisation r011**. It includes 240 recorded control steps and 1,440 original native/delivered RGB, depth and segmentation files, together with the episode record, privileged ledger, scene and protocol. Its frozen scorer reproduces the saved endpoint fields. The example was selected retrospectively with its outcome known to illustrate recovery followed by a physical violation; it is not a representative sample or a new experiment.

[Download the recorded episode](https://github.com/theemperor66/colosseum-alignment-research/releases/download/v1.1.0/colosseum-example-f4-r011-v1.1.0.zip) · [Data inventory, hashes and instructions](data/README.md)

## Install and test the software

Use Python 3.12 for the development environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-deps -e .
make verify
```

These checks run on a CPU without a simulator. Fixtures and the separately labelled proposed reporting examples are synthetic software checks, not additional experiments. See [reproducibility](docs/reproducibility.md) for dependency pinning and the distinction between software checks, endpoint verification, raw-record reanalysis and new flights.

## Repository map

| Path | Purpose |
|---|---|
| `src/colosseum_assurance/` | Frozen runtime, perception, guards, evaluation, reconstruction and analysis modules. |
| `study/` | Fixed study specification, 15 protocol files, corrected scorer, study-specific analysis routines and file provenance. |
| `scripts/` | Public verification entry points. |
| `configs/` | Explicitly unqualified configuration examples for new environments. |
| `tests/` | CPU software checks with synthetic fixtures. |
| `proposed-code/` | Reporting-contract demonstrations developed after collection; excluded from claims about the executed flight policy. |
| `docs/` | Research design, reproducibility and interpretation guidance. |
| `data/` | Public compact evidence, all computed episode rows, aggregate results and provenance inventories. |
| `CITATION.cff` | Software citation metadata for this release. |

The release preserves the study's scientific source while providing a public research interface around it. Its frozen plan retains precollection status text and the original scorer binding. The [documented scorer correction](docs/reproducibility.md#frozen-source-and-the-scorer-correction) governs the reported endpoint analysis; generic inherited settings do not redefine that analysis.

## Evidence, access and citation

The [public data deposit](data/README.md) contains the complete computed dataset and one recorded raw episode. The full collection of 104 raw archives (32.53 GB compressed) remains separately retained; its public inventory identifies every archive and the selected episode bindings. Full raw reanalysis therefore requires additional inputs. Appendix A.16 identifies the versioned release; Appendix A.7 distinguishes the evidence layers. The companion manuscript is supplied separately. Simulator binaries, Unreal asset packages, credentials and operational infrastructure are excluded.

The public research release is version **1.1.0**. Release **v1.0.0** remains the unchanged original software deposit. The frozen Python package retains its original internal version **0.1.0**; no controller, protocol, scorer or recorded result was changed for the data release.

Please cite release **v1.1.0** using [CITATION.cff](CITATION.cff) and record the commit used in any extension. No journal acceptance, public manuscript accession or DOI is claimed. [LICENSE](LICENSE) applies to the source code. [Data access and reuse terms](data/README.md#access-and-reuse) distinguish the deposited evidence from code and make no claim to third-party asset rights.
