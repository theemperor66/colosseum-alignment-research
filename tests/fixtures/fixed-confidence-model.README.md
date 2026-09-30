# Fixed confidence model parameter fixture

`fixed-confidence-model.json` is the byte-identical, 4,900-byte model envelope already prespecified
for the prospective confidence study. Source: `output/analysis/heldout-main144-v1/vision-model.json`.
Model identity: `sha256:0ce57a0611ccfc41601eba8c2e7ae9b341ac2fafe2cb3c1047646ca8f3c4f625`.

It contains fitted coefficients, feature normalizers, calibration parameters, hashes, counts and
split-group identifiers. It contains no training/evaluation rows, raw pixels, target labels, test
predictions, outcomes, credentials or local paths. Preserve its full envelope: stripping the group
identifiers would change the model identity. Analytic test inputs remain synthetic; using this fixed
parameter artifact never turns fixture observations into empirical data or validates calibration.
No parameter was selected or refit by these tests.
