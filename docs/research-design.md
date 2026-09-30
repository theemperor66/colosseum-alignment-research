# Research design and interpretation

## What the method contributes

The method uses a physical simulation to examine the connection between an agent's information, its actions, the resulting consequences and the records available for review. It supplies empirical premises for ethical and safety arguments. It cannot determine whether a mission is legitimate or whether an imposed risk is justified solely by checking a configured predicate.

The implemented case concerns abstract civilian UAV inspection. Infrastructure restoration and emergency service are motivating civilian-defence applications, not additional simulated events. No civilians, disaster damage, human supervisors or real communications systems were studied. The ethical family labels and general conceptual framework were developed retrospectively around the unchanged experimental conditions.

Four distinctions organise the analysis:

1. A guard's affirmative verdict is distinct from independently scored conformance.
2. Avoiding a violation is distinct from completing useful authorised service.
3. Reaching a required state is distinct from maintaining it for the required duration.
4. A retained record is distinct from sufficient evidence for a particular reconstruction question.

## Implemented ethical conflicts

| Family | Ethical question | Controlled representation |
|---|---|---|
| F1 — Duty of care | When precaution withholds useful service, who should bear the cost of uncertainty? | Delivered-image obscuration and depth loss. |
| F2 — Fidelity to purpose | Does rewarding a measurable proxy preserve the benefit that justifies the task? | A finite capture/return optimiser with changed proxy weight. |
| F3 — Delegated authority | What discretion remains legitimate after supervisory contact is lost? | Synthetic authorisation delay and supervision outage. |
| F4 — Evidential responsibility | What grounds justify imposing risk when the available evidence is reassuring but unreliable? | Delivered position and depth biases. |
| F5 — Limits of permission | Does a beneficial objective justify acting beyond the spatial or temporal scope of permission? | Tighter spatial boundary and shorter token lifetime. |

The measured obligations operationalise selected parts of these conflicts. They do not measure the social cost of delayed service, emergency exception legitimacy, stakeholder consent, privacy rights or the moral adequacy of the mission. Those conclusions require additional normative premises and evidence.

## Configurations and design

All configurations share a fixed, hand-written perception-based controller. The flight policy does not query a language model. The controller consumes delivered RGB/depth and state information; privileged geometry, segmentation identity and sampled truth support qualification and evaluation.

| Configuration | Role | Selected episodes |
|---|---|---:|
| A0 | Unguarded controller | 216 |
| A1 | Policy-only diagnostic guard | 216 |
| A2 | Assumption-aware guard with bundled validity checks | 216 |
| B1 | A1 with predictive boundary handling | 48 |
| B2 | Immediate-abort reference | 48 |
| B3 | A2 with authorisation record-age handling changed | 48 |

The primary mixture has 15 cells: two nominal conditions, five family treatments and eight additional delay combinations. Each cell contains 12 matched groups spanning three layouts, two requested visibility strata and two realisations. Its 180 groups each receive A0, A1 and A2.

F5 expands from 12 to 48 environments and includes all six configurations. The complete design is therefore `14 × 12 × 3 + 48 × 6 = 792` episodes in 216 environments. Descriptive totals over 216 environments give F5 more weight than the primary mixture. They must not be substituted for the 180-group primary estimand.

The sample size was chosen for coverage within a compute budget, without a power-based minimum. A1 is a deliberately limited diagnostic baseline; A2 changes several checks together. Their contrast does not isolate an individual check or reproduce a published state-of-the-art monitor.

## Selection and uncertainty

Initial collection retained 789 attempts, including 757 complete episodes. A fixed follow-up queue targeted the 32 incomplete and three missing design slots after interim outcomes were known. The first technically complete follow-up per slot was selected regardless of outcome. Thirty-seven additional attempts supplied 35 completions, yielding 792 selected complete episodes and 826 retained attempts. All originally complete records were preserved.

Follow-ups retained the scientific inputs but used later, qualified simulator sessions. They are fresh physical realisations, not continuations of interrupted trajectories. Outcome-independent queue construction does not establish random missingness, independence across sessions or prospective confirmation. The original bounded analysis is retained as a sensitivity result.

The primary physical-violation contrast is A1 minus A2 over the 180 primary groups: **0.3389**, with conservative marginal 95% Hoeffding interval **[0.1364, 0.5413]**. The interval assumes independent groups and valid endpoint measurement. It does not correct shared-session dependence. Procedural and safe-useful contrasts are exploratory; secondary intervals have no multiplicity adjustment.

## Outcome definitions

- **Physical and procedural conformance:** independently scored sampled obligations. Established violations remain positive; insufficient negative evidence is represented by bounds rather than a pass. Contacts reflect a location rule with a declared home exemption, not measured impact damage.
- **Safe useful completion:** qualifying authorised captures, dwell, return and a quiet landed home state, with complete evidence and no established physical or procedural violation. This is an inspection proxy, not a measure of image utility to a real operator.
- **Affirmative acceptance:** terminal completion, PASS at every evaluated monitor step, no UNKNOWN or VIOLATION and sufficient control-step coverage. Its native retained horizon differs from the independent common-window endpoint.
- **Conditional false assurance:** violation among accepted episodes, reported alongside acceptance coverage. An empty acceptance set makes the conditional rate undefined. Different guards accept different sets, so their conditional rates do not alone identify a causal safety effect.
- **Recovery:** a declared short sampled response endpoint. Later violations establish that this endpoint cannot warrant continued mission safety.
- **Reconstruction:** answers to four fixed question types under restricted views of an unchanged record. The final conservative procedure improves error avoidance partly by withholding answers; answer coverage remains necessary for interpretation.

## Scope of the evidence

The main empirical contribution is an observed separation of these quantities in one closed-loop 3D case. Headless rendering preserves the simulated physics and visual observations. It does not establish sensor or dynamics fidelity to deployment.

The separately fitted historical visual predictor did not control the main flights. The record audit is retrospective. Proposed reporting contracts are software demonstrations with synthetic tests. No learned flight policy, stakeholder study, hardware-in-the-loop evaluation, real-world flight, validated repair or external human peer review is reported.

New agents, morphologies and missions require fresh qualification, measurement validation and evaluation. Their numerical rates cannot be inferred from this study merely because they use the same simulation interface.
