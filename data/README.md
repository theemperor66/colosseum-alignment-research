# Public research evidence

Release **v1.1.0** deposits the computed evidence for all **792 selected episodes** and one authentic complete raw episode. It adds access to the completed study; no new flight was performed, and the study’s endpoint values and 792-episode selection remain unchanged.

## Download and inspect

| Material | Location | Scope |
|---|---|---|
| Compact evidence archive | [`compact/evidence.zip`](compact/evidence.zip) | All selected endpoint rows, aggregate results, original sensitivity results, reconstruction/response analyses, provenance, frozen source and 15 protocols. |
| Browsable computed records | [`derived/`](derived/) | Byte-identical copies of all 16 archived `review-data/` files. |
| Authentic raw example | [Download F4/A2/r011](https://github.com/theemperor66/colosseum-alignment-research/releases/download/v1.1.0/colosseum-example-f4-r011-v1.1.0.zip) | One complete recorded episode, original sensor files, sampled state/action records and a raw-evidence recalculation. |
| Full collection inventory | [`provenance/`](provenance/) | Archive hashes and member bindings for all selected episodes; the inventory does not contain the full raw collection. |
| Release bindings | [`release-manifest.json`](release-manifest.json) | File sizes and SHA-256 values for the deposited data and the release asset. |

The compact ZIP is **952,403 bytes**; its SHA-256 is:

```text
8fb48df5102ba4b06c1adcae83c7434152bdadff31009bc9f0d4d066d44ad894
```

The raw example ZIP is **49,268,340 bytes** (46.99 MiB), expanding to 145,772,291 bytes. Its SHA-256 is:

```text
e550a2ed609a7441d478180166626fff42161372e68ae524b6b8e9ec5ad0d92d
```

The compact ZIP is also available as a [release download](https://github.com/theemperor66/colosseum-alignment-research/releases/download/v1.1.0/evidence.zip). [`SHA256SUMS.txt`](https://github.com/theemperor66/colosseum-alignment-research/releases/download/v1.1.0/SHA256SUMS.txt) records both downloadable archive hashes.

## Check all reported episode results

From the repository root, using Python 3.10 or later:

```sh
python3 scripts/verify_paper.py data/compact/evidence.zip
```

This standard-library check verifies the archive and its 126 manifested files, then checks all 792 selected complete episodes, preservation of 757 original complete outcomes, 35 selected first-complete follow-ups and accounting for all 826 retained attempts. It recomputes the reported contrasts and aggregate counts from saved endpoint labels. It does not derive those labels again from images and trajectories.

The primary comparison contains 180 matched environment groups. The 216-environment totals additionally include the enlarged F5 comparison. Keep these denominators distinct. Group independence is an assumption of the reported interval; shared simulator sessions do not establish it. These numerical results concern the finite simulated design and do not estimate deployment incident rates.

The browsable endpoint table is [`derived/endpoint-rows.json`](derived/endpoint-rows.json). Other files retain the initial bounded analysis, follow-up transitions, recovery comparisons, reconstruction counts and guard/request linkage. [`derived/derivation-provenance.json`](derived/derivation-provenance.json) explains the source-to-review transformation. Infrastructure identifiers were removed when these review records were prepared; their numerical outcomes are unchanged.

## Inspect the authentic raw example

The asset contains **A2, F4, realisation r011**, episode `heldout-b367c734b5a3-simulated_manipulation-r011__A2_assumption_aware`. It was originally complete and remains one of the 757 preserved original episodes. It is a single configuration's flight, not a complete matched multi-arm scenario.

The package provides all **240 retained control steps** and **1,440 native/delivered RGB, depth and segmentation files**, with the episode record, privileged ledger, state transformations, camera poses, scenario, protocol and recorded qualification evidence. Sensor files retain their original bytes. Operational paths in structured records were rewritten to package-relative references; shared streams were restricted to the selected episode or explicitly excluded. [`example/provenance.json`](example/provenance.json) records source and public hashes and each transformation. The complete original archive remains unchanged.

This episode was selected retrospectively with its outcome known. It illustrates short sampled recovery followed by a later contact-rule violation. The selection is illustrative and cannot establish a frequency. The contact rule measures neither injury nor damage. The full 792-episode computed dataset supplies the study's denominators.

After downloading and extracting the asset, use the repository's documented Python 3.12 environment and run:

```sh
python /path/to/colosseum-example-f4-r011/recompute.py --repository /path/to/colosseum-alignment-research
```

The included program verifies 1,457 package-file hashes and 87 frozen evaluator/source files, recalculates the episode from its retained evidence, and compares all 16 computed fields with the saved calculation and reported row. It needs no running simulator or GPU. The original source records and the public copy produced identical fields during release preparation. See the asset's README for field definitions and exclusions; [`example/release-receipt.json`](example/release-receipt.json) records the release check.

## What remains separately retained

The complete raw collection consists of **104 ZIP archives**, totalling **32,531,349,292 bytes** compressed (32.53 GB). It is not included in the public download. [`provenance/raw-inventory.json`](provenance/raw-inventory.json) lists its archives; [`provenance/episode-index.json`](provenance/episode-index.json) binds all 792 selected episodes to exact episode, ledger and scenario members. [`provenance/RAW-SHA256SUMS.json`](provenance/RAW-SHA256SUMS.json) also binds the original seal and inventory sidecars. Repeated archival copies do not create extra trials.

The author retains the full collection for an agreed examiner handover. A recipient and transfer channel must be arranged through the existing course or submission contact. An inventory is not evidence that a recipient has received the raw files. The public example permits raw-record inspection and recalculation of that one flight; full measurement reproduction still requires the complete retained inputs.

## Provenance and historical wording

The compact archive is byte-identical to the manuscript attachment. Its original README names an earlier working title and states that no public deposit or unrestricted reuse licence is implied. Those sentences describe the archive's earlier distribution state. The author now publicly deposits the unchanged archive in v1.1.0. The old files remain intact to preserve their hashes; this guide describes current availability.

Release v1.0.0 and all frozen scientific source files remain unchanged. The package's internal version remains 0.1.0. Precollection status labels and the original scorer binding stay in the frozen plan; the [documented analysis erratum](../docs/reproducibility.md#frozen-source-and-the-scorer-correction) identifies the corrected scorer used for the reported results.

## Access and reuse

The source code is distributed under the repository's [Apache-2.0 licence](../LICENSE). Public access to these evidence files provides material for inspection and reproduction; this deposit does not declare an additional blanket data reuse licence. The software licence does not automatically license the data or third-party rendered content.

The simulator executable, Unreal maps, textures and asset packages are excluded. No rights to those third-party materials are claimed or transferred. Cite the versioned research release and retain the source/public provenance when reporting an analysis of the deposited evidence. See [`CITATION.cff`](../CITATION.cff).
