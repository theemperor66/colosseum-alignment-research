# Frozen study materials

These files preserve the scientific implementation used for the companion manuscript. Their hashes are checked by `python scripts/verify_source.py` from the repository root.

| File or directory | Role |
|---|---|
| `protocols/` | Fifteen frozen protocol configurations. |
| `study-plan.json` | Original plan, including its precollection status and original scorer binding. |
| `scorer.py` | Disclosed corrected endpoint scorer for sealed raw archives. |
| `secondary_functions.py` | Historical aggregation functions with original analysis-path constants. |
| `guard_linkage_audit.py` | Historical command-linkage analysis with original archive-path assumptions. |
| `verify_reported_results.py` | Unchanged arithmetic verifier from the compact evidence archive. |
| `original-bytes-sha256.json` | Original frozen-file manifest. |
| `release-file-provenance.json` | Mapping from public paths to original archive members and hashes. |

Use `python scripts/verify_paper.py data/compact/evidence.zip` to check the compact evidence now included in release v1.1.0. The wrapper prepares validated inputs for the unchanged arithmetic verifier. Running the archived verifier directly in this directory does not supply those inputs.

The two historical helpers retain path assumptions for provenance. The data release provides all computed rows, public archive bindings and one complete raw episode. Reanalysis of the full collection additionally requires the separately retained sealed archives. See the [reproducibility guide](../docs/reproducibility.md) for the scorer erratum, reproduction levels and limits.
