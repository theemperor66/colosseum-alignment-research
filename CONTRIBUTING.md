# Contributing

Contributions that improve inspectability, reproducibility or the clarity of evidence are welcome. Start with the [research design](docs/research-design.md) and [reproducibility guide](docs/reproducibility.md) so that a software change's scientific consequences are explicit.

## Preserve the study record

Release `v1.0.0` identifies the original public software deposit; `v1.1.0` adds the authorised computed dataset and one authentic raw episode. Frozen source, protocols and the disclosed corrected scorer preserve the study's measurement definitions. Changes to scientific behaviour belong in a new, clearly versioned implementation with an explanation of the changed assumptions. They must not be presented as code that produced the original results.

Distinguish a defect report from a proposed methodological extension. For a defect that affects reported measurements, state the affected inputs and expected effect and preserve the original output. A corrected analysis needs an explicit erratum and new provenance; changing a hash manifest alone is not a scientific correction.

## Propose a change

Open an issue describing the concrete problem, affected release and expected behaviour. A useful reproduction includes the smallest shareable example, command and observed result. Keep credentials, private records and restricted third-party assets out of issues and pull requests. Reference the versioned public data when reporting its results; additional research data require explicit publication authorisation, provenance and a content review.

For a pull request:

1. State the problem, resulting behaviour and any impact on scientific interpretation.
2. Add a focused regression check when the change affects behaviour.
3. Run the relevant CPU checks and identify any checks you could not run.
4. Keep synthetic examples labelled and separate from experimental evidence.
5. Update the documentation when an interface or its interpretation changes.

Use the local setup and commands in the [reproducibility guide](docs/reproducibility.md). Passing fixture tests must not be described as simulator qualification, new experimental data or validation of real-world safety.

For ethical or methodological proposals, identify the normative premise, its operational measure and the inference supported by the proposed evidence. Adding a policy condition alone does not establish that the policy is morally sufficient.
