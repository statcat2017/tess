# Product Requirements Document

## Multi-Epoch TESS Transit Event Association

## Problem Statement

Long-period transiting exoplanets are difficult to recover from TESS because a star may be observed for only one approximately 27-day sector during an observing epoch. A planet with a period of tens to hundreds of days may therefore produce only one transit-like event, leaving its period unconstrained and making the event difficult to distinguish from stellar variability, eclipsing binaries, or instrumental artefacts.

The extended TESS mission provides a new opportunity. Stars observed during early mission sectors are increasingly revisited years later. A later transit-like event around the same TIC can convert an isolated monotransit into a duotransit or multiepoch candidate, but only if the two events are correctly associated.

The researcher needs a rigorous, reproducible way to answer whether two individually detected events around the same star are likely to have been produced by the same transit-producing object. The researcher also needs to know whether that association reduces orbital-period aliases when the full TESS observing window is considered.

The repository currently contains the validated research proposal but no implementation, data model, pipeline, or test suite. The work must therefore be phased around observable scientific checkpoints rather than around an assumed neural-network outcome.

## Solution

Build a reproducible batch pipeline for multi-epoch TESS transit-event association.

The pipeline will use a high-recall upstream event proposal process, represent each candidate as an event record, construct positive and realistic hard-negative candidate pairs, and compare two association strategies:

1. A deterministic baseline using depth, duration, morphology, and timing compatibility.
2. A small learned association model using normalized local transit morphology, with optional scalar features evaluated through ablations.

Associated events will be passed to a deterministic orbital-inference layer. The layer will generate period aliases from the event separation and reject aliases that predict confidently observable but absent transits in the actual TESS observing window.

The principal evaluation will be data-blinded and temporal. Observations through Sector 79 will be used for development. Sectors 80–105 will remain sealed until preprocessing, event-selection rules, model weights, thresholds, ranking rules, and alias logic are frozen. The primary result will compare repeat-event retrieval and candidate burden for the learned and deterministic methods on previously unseen TICs.

The learned method will remain central only if it provides a practically meaningful improvement over the deterministic baseline. Otherwise, the project will pivot to a deterministic morphology-aware framework without invalidating the dataset, orbital-inference layer, or temporal evaluation.

## User Stories

1. As an exoplanet researcher, I want to identify stars with observations in multiple TESS epochs, so that isolated transit-like events can be compared across time.
2. As an exoplanet researcher, I want to retain high-recall event proposals, so that the association stage is not limited by an overly selective upstream detector.
3. As a data engineer, I want every candidate event represented by a stable event record, so that extraction, pairing, modelling, and evaluation use one consistent contract.
4. As a data engineer, I want event records to include sector, event time, local time and flux windows, event depth, duration, S/N, stellar metadata, and quality information, so that both deterministic and learned methods have the necessary evidence.
5. As a data engineer, I want optional centroid, background, and crowding data to be represented separately from the core morphology, so that they can be added through controlled ablations or candidate vetting.
6. As a researcher, I want TIC identifiers to be used only for grouping and partitioning, so that the association model cannot memorize star-specific identity.
7. As a researcher, I want known planets with events in multiple TESS epochs to generate positive candidate pairs, so that the association task has physically motivated labels.
8. As a researcher, I want realistic same-TIC hard negatives, so that reported performance reflects the false associations the deployed workflow will actually encounter.
9. As a researcher, I want random, morphology-matched, and same-TIC hard-negative performance reported separately, so that easy negative pairs cannot conceal failure on realistic candidates.
10. As a researcher, I want all events from a TIC assigned to only one data partition, so that validation and test results are not inflated by star-level leakage.
11. As a researcher, I want development data restricted to observations through Sector 79, so that later-sector measurements cannot influence model design.
12. As a researcher, I want a deterministic association baseline before any neural model is trained, so that the project can measure whether machine learning adds value.
13. As a researcher, I want the deterministic baseline to compare depth, duration, morphology, and timing, so that it represents a credible scientific alternative rather than a straw-man benchmark.
14. As a researcher, I want the learned association model to compare normalized local transit morphology, so that it can use shape information beyond summary statistics.
15. As a model developer, I want to start with a modest Siamese or contrastive one-dimensional CNN, so that the experiment tests the association hypothesis without requiring large GPU infrastructure.
16. As a model developer, I want optional stellar and contamination metadata controlled through ablation studies, so that any improvement can be attributed to meaningful information rather than unexplained feature accumulation.
17. As a researcher, I want the association model to output the probability that two events arise from the same transit-producing object, so that its output is not misinterpreted as a planet probability or orbital-period estimate.
18. As an exoplanet researcher, I want candidate periods generated as aliases of the observed event separation, so that sparse events produce an explicit finite hypothesis set.
19. As an exoplanet researcher, I want each period alias tested against the actual TIC observing window, so that aliases predicting observable but absent transits can be rejected deterministically.
20. As a researcher, I want the number of viable aliases reported before and after window filtering, so that the orbital-inference contribution is measurable independently of association quality.
21. As a researcher, I want the correct later event ranked among competing candidate pairs, so that evaluation reflects the operational task rather than generic binary classification.
22. As a human reviewer, I want candidate burden measured at fixed repeat-event recall, so that I can determine how many associations require inspection in practice.
23. As a researcher, I want top-k retrieval, mean reciprocal rank, precision-recall, and fixed-burden metrics, so that the result is not dependent on generic accuracy.
24. As a researcher, I want injection/recovery experiments to vary signal strength, event morphology, and epoch separation, so that the method's operating regime is known.
25. As a researcher, I want injections performed into real TESS light curves, so that association performance is tested against realistic noise and instrumental context.
26. As a researcher, I want preprocessing rules, model weights, thresholds, manifests, seeds, and predictions frozen before holdout unblinding, so that the temporal evaluation is auditable.
27. As a reviewer, I want an independent check that no Sectors 80–105 measurements entered development, so that the pseudo-prospective claim is credible.
28. As a researcher, I want failure analysis separated into event-detection failure and event-association failure, so that a weak result identifies the actual limiting stage.
29. As a researcher, I want the project to pivot to deterministic matching if machine learning does not materially improve the baseline, so that the study does not become hostage to a neural architecture.
30. As a researcher, I want known long-period systems reserved for meaningful evaluation where feasible, so that training performance does not substitute for scientific recovery performance.
31. As a researcher, I want alternative TESS reductions used for final candidate verification rather than core training, so that the primary benchmark remains controlled.
32. As a researcher, I want Sector 106 treated as a conditional discovery cohort, so that discovery candidates do not determine whether the core methodology succeeds.
33. As a candidate vetter, I want high-ranked candidates checked for eclipsing-binary signatures, secondary eclipses, odd/even differences, crowding, centroid movement, Gaia neighbours, and difference-image evidence, so that associations are not presented as confirmed planets.
34. As a researcher, I want external photometry queried only for high-ranked candidates and specific aliases, so that follow-up work does not expand into an archive-scale dependency.
35. As a maintainer, I want every derived dataset and result tied to source-product versions and download dates, so that later reruns can explain differences.
36. As a maintainer, I want the pipeline runnable locally on compact event data with archive-side preprocessing available through TIKE, so that development does not require years of raw TESS files on the local machine.
37. As a paper author, I want the final report to support either a positive ML result, a null ML result, or an association-limit result, so that the study remains publishable regardless of the model comparison outcome.
38. As a future researcher, I want the event and association outputs to be suitable for later graph-based multievent analysis, so that future work can extend the core result without changing the foundational data contract.

## Implementation Decisions

- The first release is a reproducible batch research pipeline, not a continuously deployed service or polished candidate-report application.
- The core scientific unit is a candidate event pair around one TIC, followed by deterministic period-alias inference.
- The primary photometric product will be selected during the protocol phase and then held fixed for the core benchmark. TESS-SPOC FFI light curves are the default candidate because they align with the proposal's related-work baseline. SPOC, QLP, TGLC, and other reductions are reserved for robustness checks and final candidate vetting.
- Individual-event detection is upstream infrastructure. The project will prioritize recall and will not claim novelty for a new monotransit detector.
- Event proposals must produce records containing TIC identifier, sector, event time, local time array, local flux array, depth, duration, S/N, stellar metadata, and quality information.
- TIC identifiers are grouping and partitioning keys only. They must not be model features.
- Positive pairs will be generated from known planets with compatible events in multiple TESS epochs. Label construction must record the catalogue version and the information available at label-generation time.
- Negative pairs will prioritize same-TIC hard negatives, including unrelated artefacts, stellar variability, eclipsing-binary-like events, morphology-matched events, and timing-incompatible events.
- All events from a TIC must belong exclusively to training, validation, or test. The temporal holdout adds a second boundary: development uses Sectors 1–79 and the primary sealed evaluation uses Sectors 80–105.
- The deterministic baseline will compare relative depth difference, relative duration difference, normalized morphology correlation, and timing plausibility. Thresholds must be learned only from development data.
- The learned association model will initially use a small Siamese or contrastive one-dimensional CNN with shared event encoding, compact embeddings, absolute embedding difference, and a small comparison head.
- The first learned-model input is normalized local transit morphology. Stellar metadata, depth, duration, S/N, and contamination information are optional inputs evaluated through explicit ablations.
- Timing information will initially remain in the deterministic orbital layer rather than being supplied broadly to the neural model, reducing the risk that the model memorizes common period distributions.
- The learned output is `P(same transit-producing object)`. It is not `P(star contains a planet)` and it is not an orbital-period prediction.
- Candidate periods will be generated from `P_n = Delta T / n` for permitted positive integers, with an initial long-period lower boundary near 27 days.
- The observing-window filter will use the actual cadence availability and quality for each TIC. It will predict transit epochs, establish whether usable TESS data exist, check for corresponding events, and reject aliases with confidently observable but absent events.
- The primary association metric is true-repeat retrieval at a fixed candidate burden. Supporting metrics include precision-recall AUC, top-1 and top-5 retrieval, mean reciprocal rank, recall at fixed false-association rate, and precision at fixed human-vetting budget.
- Full-pipeline reporting will include recovered known long-period systems, retrieved true repeat events, candidate pairs per recovered system, aliases before and after window filtering, and false associations per 1,000 targets.
- The project will begin with a tracer-bullet dataset containing a small set of known multiepoch systems and realistic competing events. This validates the data contract and end-to-end flow before broad archive processing.
- The pipeline will expose event-detection recall separately from event-association performance. A missed event must not be reported as an association failure without evidence that the event was retained upstream.
- The project will follow these phased checkpoints: protocol freeze, data smoke test, event-recall check, label/leakage audit, deterministic baseline, learned-model comparison, freeze/seal audit, temporal holdout, and injection/operating-regime report.
- The ML stop/go decision is operational rather than purely statistical. ML continues as the principal method only if it materially improves repeat recall at fixed false-association rate, candidate burden at fixed recall, or top-k correct-match retrieval on unseen TICs.
- If ML does not materially improve the baseline, the frozen dataset, deterministic matcher, orbital layer, and temporal evaluation remain the core contribution.
- The sealed holdout is an operational data boundary, not a claim that public data are inaccessible. Holdout manifests, measurements, and labels must be kept outside development inputs until the freeze record is complete, with hashes and access logs sufficient for audit.
- Reproducibility metadata will include exact TIC lists, sector manifests, source-product versions, catalogue versions and download dates, preprocessing parameters, event-selection rules, pair construction, TIC assignments, random seeds, model checkpoints, thresholds, injection parameters, and holdout-unblinding date.
- Local execution will handle model development, training, plotting, debugging, ablations, analysis, and manuscript generation. TIKE may handle archive-side manifests, large-scale preprocessing, compact event extraction, and selected transit-search workloads.
- Candidate-specific false-positive vetting occurs only after the core temporal experiment. It must not alter the primary holdout scores.
- Sector 106 is a conditional discovery extension. It proceeds only after the core temporal result and is reported separately from the primary methodology result.

## Testing Decisions

- Good tests will verify externally observable scientific behavior: records are valid, partitions are leak-free, event pairs receive the expected labels, aliases are generated and filtered correctly, and frozen runs produce reproducible rankings and metrics.
- The highest test seam is one end-to-end batch run from a versioned compact event dataset and TIC observing manifest to ranked candidate associations, viable period aliases, and evaluation metrics. This seam should be exercised with a small golden fixture before archive-scale runs.
- No implementation detail of the CNN, data loader, or internal helper structure should be the primary test target. Model internals may be tested only where they affect the public scoring contract or reproducibility.
- Event-record validation will test required fields, units, finite values, time ordering, cadence alignment, quality flags, and serialization round trips.
- Partition tests will assert that a TIC cannot occur in multiple partitions and that no holdout measurement or label is consumed by development workflows.
- Pair-construction tests will verify positive pairs from known repeated events and hard negatives from realistic same-TIC contexts. Tests will also confirm that random unrelated-star pairs cannot dominate a dataset by accident.
- Deterministic matcher tests will use controlled events with known depth, duration, morphology, and timing differences to verify compatibility decisions and threshold behavior.
- Orbital-inference tests will verify alias arithmetic, integer-cycle handling, period boundaries, transit-epoch prediction, cadence coverage, and rejection of aliases contradicted by observed data.
- The orbital layer should include property-style tests for invariants such as positive periods, increasing predicted epochs, and preservation of the observed event times under a generated alias.
- Association evaluation tests will verify ranking, top-k retrieval, candidate-burden curves, fixed-recall comparisons, and metric behavior on known toy outcomes.
- Reproducibility tests will rerun the same frozen fixture with the same seed and assert equivalent event records, scores, rankings, and aggregate metrics within declared numerical tolerances.
- Injection tests will verify that injected events preserve the source light curve outside the intended window, carry correct provenance, and can be recovered under the expected signal and separation regimes.
- Temporal-holdout tests will run against a deliberately miniature sealed fixture and verify that unblinding is impossible through normal development inputs before the freeze checkpoint.
- The tracer-bullet fixture is the primary prior art for this repository because no existing implementation or test suite is currently present. The fixture should include known repeat events, unrelated same-TIC dips, missing-cadence intervals, multiple aliases, and at least one rejected alias.
- Archive-scale execution is an integration and reproducibility test, not a substitute for deterministic fixture tests. It must emit manifests, logs, version metadata, and compact derived outputs.
- Human vetting remains a scientific validation activity rather than an automated test. The workflow should preserve plots and candidate context so a reviewer can inspect why an association was ranked.

## Out of Scope

- Developing a novel large Transformer for individual transit-event detection.
- Treating event detection as the principal methodological novelty.
- Exhaustive TGLC or TARS archive searches.
- Pixel-level deep-learning models.
- Graph neural networks or full event-graph clustering in the first release.
- Systematic archive-scale processing of ZTF, ATLAS, or ASAS-SN.
- Occurrence-rate measurement.
- A full multidimensional TESS survey selection function.
- Automated follow-up observation scheduling.
- A polished candidate-reporting product.
- Claiming that an association probability is a planet probability.
- Claiming that a pair of events uniquely determines an orbital period without alias analysis.
- Using current catalogue information in a way that leaks later observations into development labels or features.
- Using Sector 106 discovery candidates to tune the core model or thresholds.
- Adding architecture complexity without an empirical failure mode and a predeclared evaluation.
- Replacing the deterministic baseline with a weak benchmark designed to make ML appear better.
- Treating a statistically detectable improvement as sufficient without an operational improvement in recall, candidate burden, or ranking.
- Full candidate-specific follow-up before the sealed temporal evaluation is complete.
- Comprehensive completeness claims beyond the limited injection/recovery operating-regime study.
- A requirement to discover a new planet for the project to succeed.

## Further Notes

- The research proposal and citation audit are the scientific source documents for this PRD. The cited literature was validated as real, and the proposal's attributed numerical and methodological claims were found to be materially accurate.
- The first implementation milestone should be the tracer-bullet experiment, not the neural association model. It should demonstrate that a known multiepoch system can move through event extraction, pairing, scoring, alias generation, and window filtering.
- The main scientific risk is not model architecture. It is realistic event and negative-pair construction under strict temporal and TIC-level leakage controls.
- The event detector and association model must be evaluated as separate stages. This distinction is required to explain whether poor recovery comes from failure to retain an event or failure to link two retained events.
- The expected primary outcomes are intentionally symmetric: ML may improve association, deterministic morphology-aware matching may be sufficient, or even known repeat systems may be unrecoverable at acceptable candidate burden. Each outcome is scientifically reportable.
- The primary result should be written before any optional discovery extension. Sector 106 is an opportunity to apply a validated method to recent data, not a dependency of the core research.
- The project should preserve enough provenance and compact derived data to permit later extensions such as full archive event graphs, cross-reduction learning, smaller-planet association, and continuous new-sector monitoring.
