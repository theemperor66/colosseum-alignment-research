# Reproducibility guide

Choose a reproduction level before interpreting a successful command. Each level answers a different question.

| Level | Required inputs | What success establishes |
|---|---|---|
| Software checks | This repository and Python dependencies | Specified software behaviour on synthetic fixtures. |
| Reported arithmetic | This repository, including `data/compact/evidence.zip` | Accounting, contrasts and hashes computed from all 792 saved endpoint labels. |
| Published raw example | One complete recorded episode from the v1.1.0 release and the frozen scorer | Endpoint recalculation for that episode and inspection of its original observations. |
| Raw-record reanalysis | Sealed raw archives, bound source and dependencies | Recalculation of measurements from retained physical and image records. |
| New simulated flights | Qualified simulator, scene, settings, protocols and runtime | New observations under the declared apparatus and execution conditions. |

Software checks, arithmetic verification and raw-record recalculation need no GPU or running simulator. Recalculating the published example does not reproduce the measurements for the full collection.

## Check the compact evidence package

The unchanged compact archive is included at [`data/compact/evidence.zip`](../data/compact/evidence.zip). Its 16 `review-data/` members are also available as byte-identical browsable files under [`data/derived/`](../data/derived/). The manuscript's PDF attachment contains the same archive. See the [data guide](../data/README.md) for the public deposit and the boundaries of the raw example.

From the repository root:

```bash
python3 scripts/verify_paper.py data/compact/evidence.zip
```

Alternatively, pass a freshly extracted directory containing the archive's `review-data/` and `frozen-analysis/` directories:

```bash
python3 scripts/verify_paper.py /path/to/extracted-evidence
```

The repository wrapper checks the exact archive digest or the complete extracted file inventory against the release's trusted manifest. Keep the extracted directory pristine: additional files, including generated `__pycache__` directories, are rejected. The wrapper runs the repository's retained verifier on validated JSON inputs; it does not execute Python supplied by an arbitrary archive.

The archived verifier remains directly runnable from an extracted archive:

```bash
cd /path/to/extracted-evidence
python3 review-data/verify_reported_results.py
```

The standard-library check verifies 792 selected complete episodes, preservation of 757 original complete outcomes, 35 selected follow-up completions and accounting for 826 retained attempts. It recomputes the three primary-group contrasts, six focused contrasts, original sensitivity comparisons and arm totals, and checks derived-file hashes. Only the physical contrast was prespecified as primary. No labels are generated anew from simulator records by this check.

The compact archive supplied with this manuscript has SHA-256:

```text
8fb48df5102ba4b06c1adcae83c7434152bdadff31009bc9f0d4d066d44ad894
```

A matching digest establishes byte identity to this reference, not independent authentication of the scientific provenance. The archive's historical README uses an earlier working title and describes the prepublication access state. Its bytes remain unchanged for provenance. The author publicly deposits that same archive in v1.1.0; the [current data guide](../data/README.md) describes its access and reuse terms.

## Install and run CPU checks

Use Python 3.12 and an isolated environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements-lock.txt
python -m pip install --no-deps -e .
make verify
```

The lock file pins dependencies and their package hashes. The editable installation then exposes this checkout without resolving a new dependency set. The individual checks can also be run directly:

```bash
python scripts/verify_source.py
python -m pytest tests -q
python -m unittest discover -s proposed-code -v
```

`requirements-lock.txt` supplies the recorded pinned dependencies, including development tools. Record the Python version and resolved package versions when reporting an extension.

Tests use synthetic inputs. Their passing status is evidence about software behaviour, not additional support for the paper's physical results. The eleven tests in `proposed-code/` concern post-collection reporting demonstrations; those modules had no control authority during collection.

## Frozen source and the scorer correction

The package under `src/colosseum_assurance/` preserves the study source snapshot. `study/` retains the fixed plan, 15 protocols and study-specific analysis files. `study/release-file-provenance.json` maps released files to their archived counterparts and hashes. `python scripts/verify_source.py` checks the released bytes against that mapping. Release packaging and verification interfaces are distinguished from those frozen scientific files.

`study/secondary_functions.py` and `study/guard_linkage_audit.py` are historical analysis helpers retained with their original bytes and path assumptions. They refer to the original analysis directory layout and are not standalone public quick-start commands. The portable public entry points are under `scripts/`; raw reanalysis additionally requires the bound inputs described below. Public release v1.1.0 adds evidence access while preserving v1.0.0's scientific files and the package's internal version 0.1.0.

The disclosed corrected endpoint scorer has SHA-256:

```text
aaa4087d0c32d844dbfb74aca9f7813d762ce27cfdd821ecd6ff0793b85ff62c
```

It corrects a comparison between uppercase verdict labels and lowercase schema values. The defect left some complete PASS records unresolved. The correction concerns analysis only: it does not change the controller, guard, protocol or recorded flights. Established positive violations already used the independent Boolean.

The frozen plan intentionally retains its original scorer binding. Authenticated reanalysis explicitly bound the corrected scorer as an erratum; the original freeze-checking entry point refuses changed bytes. Do not alter the plan to make that check pass or silently substitute the generic evaluator for the reported integrated-study analysis. Precollection status labels likewise describe the time of freezing, not the completed outcome.

## Recalculate endpoints from raw records

The compact evidence contains all selected endpoint rows and verification summaries. The release additionally provides one complete A2 F4 r011 episode as [colosseum-example-f4-r011-v1.1.0.zip](https://github.com/theemperor66/colosseum-alignment-research/releases/download/v1.1.0/colosseum-example-f4-r011-v1.1.0.zip). Its README and included `recompute.py` describe recalculation with this repository's frozen scorer. The example contains original sensor files, retained states, the privileged ledger, scenario and protocol; any infrastructure-path substitutions are explicitly recorded in its provenance. It was selected with its outcome known. Its result establishes no representative rate.

The full set of 104 sealed raw archives is not part of the public download. Full reanalysis requires those RGB/depth and trajectory archives, their inventories, original input bindings and the appropriate execution environment. The [public provenance inventory](../data/provenance/raw-inventory.json) identifies their sizes and hashes, while the [episode index](../data/provenance/episode-index.json) identifies all selected episode, ledger and scenario members. The author retains the complete collection for an agreed examiner handover; public access to the inventory does not grant server access or establish receipt of the raw collection.

Appendix A.6 describes completed separate-host recomputation, including the original bounded analysis, follow-up batches and first-complete combination. Appendix A.7 records provenance, and A.9 describes deviations. Those executions compare saved endpoints, aggregates and accessed-member bindings. They support computational reproduction, not external human adjudication or independent validation of the normative specification.

The archived Python package is a reusable runtime and analysis implementation. Its generic CLI is not a one-command substitute for the integrated study's bound inputs, completion selection and historical analysis procedures. Reconstructing the full study requires the inputs and procedures at this level, not just a successful package installation.

## Execute new simulated flights

Fresh execution requires a separately obtained, qualified Colosseum environment and the declared scene construction, camera, clock and controller settings. The experiment used the third-party ESAR dapeng build declared to use Unreal Engine 5.6.1. The upstream Colosseum commit compiled into that binary is unknown; the manuscript identifies the archive and selected map files by content hashes. The simulator binary and Unreal assets are not redistributed here.

Before treating a run as experimental evidence, establish simulator identity, motion, RGB/depth content, segmentation, clock behaviour, camera geometry and scene qualification. Use a private, single-owner RPC connection and retain qualification outcomes and every attempted episode. Test fixtures must remain explicitly synthetic.

Reusing seeds does not ensure identical trajectories across sessions. A fresh run is a new empirical observation. Report changes to source, configuration, apparatus and selection rules, and preserve inconclusive outcomes rather than silently treating them as passes.

## Report a reproduction result

Include the release tag and commit, Python and dependency versions, reproduction level, input digests, exact command, output and deviations. For a numerical discrepancy, identify the endpoint, group or table involved. Remove local paths, access tokens, infrastructure identifiers and restricted records before opening a public issue. See [CONTRIBUTING.md](../CONTRIBUTING.md) and [SECURITY.md](../SECURITY.md).
