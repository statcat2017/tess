# Research Proposal

## Linking the Dips: Machine-Learning Association of Multi-Epoch TESS Transit Events for Long-Period Exoplanet Recovery

**Version 2 — Revised Scope**

---

## 1. Abstract

The Transiting Exoplanet Survey Satellite (TESS) archive now spans approximately eight years of observations. This is an archive-wide baseline, not continuous eight-year coverage for most individual stars. It creates a growing opportunity to recover long-period transiting exoplanets that were poorly constrained when first observed.

For much of the sky, a single TESS observing sector provides approximately 27 days of continuous coverage. Planets with orbital periods of tens to hundreds of days may therefore produce only one observable transit during a particular observing epoch. Such monotransits are difficult to distinguish from stellar variability, eclipsing binaries and instrumental artefacts, and they do not normally provide a unique orbital period.

The extended TESS mission changes this problem. Stars observed during the primary mission are increasingly revisited in later cycles. A transit-like event observed in an early sector may therefore have a second compatible event several years later. Identifying these cross-epoch associations can convert poorly constrained monotransits into duotransit or multiepoch systems and restrict the orbital period to a finite set of aliases.

Existing research has established methods for detecting individual non-periodic transit events. [Salinas et al. (2025)](https://doi.org/10.1093/mnras/staf347) demonstrated Transformer-based detection of single-transit candidates directly from complete TESS light curves, identifying 88 single-transit candidates in Sectors 1–26. More recently, [EXOVEIL](https://arxiv.org/abs/2606.02778) introduced a self-supervised Transformer world model for individual transit-event detection without phase folding or periodicity assumptions. EXOVEIL predicts expected stellar behaviour and identifies transit-like residuals, providing an increasingly capable solution to the event-detection problem. (Priyanshu 2026)

The proposed research therefore does not seek primarily to develop another monotransit detector.

Instead, it addresses the subsequent problem:

> Given individual transit-like events observed around the same star during widely separated TESS epochs, can machine learning determine which events are likely to have been produced by the same long-period planetary object?

The principal methodological contribution will be a learned cross-epoch event-association model, benchmarked against deterministic matching based on transit depth, duration and timing. Associated event pairs will then be passed through a deterministic orbital-inference layer that generates candidate period aliases and rejects aliases inconsistent with the actual TESS observing window.

Evaluation will use a deliberately data-blinded temporal design. Model development will use observations through Sector 79. Sectors 80–105 will remain sealed until preprocessing rules, model weights and thresholds have been frozen. The system will then be tested on its ability to recover and correctly associate later transit events around previously unseen stars.

If the core methodology succeeds, the frozen pipeline will subsequently be applied to [Sector 106](https://heasarc.gsfc.nasa.gov/docs/tess/sector106_summary.html), observed from 11 July to 9 August 2026, as a recent discovery cohort.

The primary scientific output is therefore a validated methodology for long-period transit recovery rather than dependence on the discovery of a new planet.

---

## 2. Motivation

### 2.1 TESS is strongly biased toward short-period planets

Transit detection benefits strongly from repeated events.

A short-period planet may produce:

```text
────\_/────\_/────\_/────\_/
     ↑      ↑      ↑      ↑
```

Repeated transits allow signals to be phase-folded and stacked, substantially increasing signal-to-noise.

A long-period planet may instead produce a single event:

```text
─────────────────\____/────────────────
                      ↑
                 one transit
```

For most targets, a single TESS sector is too short to observe multiple events from planets with orbital periods substantially longer than approximately 27 days.

This creates two related problems:

1. long-period planets are intrinsically less likely to transit during an observing window;
2. when they do transit, there may be insufficient repetition to establish periodicity.

Injection/recovery work with [TIaRA](https://doi.org/10.1093/mnras/stae474) predicts that monotransits become a substantial part of the detectable TESS population at long periods. In its simulated TESS Year 1 and Year 3 population, approximately half of detectable planets with periods greater than 25 days and roughly three quarters of those beyond 100 days were predicted to appear as monotransits.

The long-period TESS population is therefore a scientifically meaningful incompleteness problem rather than merely an unusual edge case.

---

## 3. Why the Extended Mission Changes the Problem

The primary TESS mission frequently provided only one observing epoch for a star.

The extended mission increasingly provides:

```text
2019          2021          2023          2025/26
████          ████          ████            ████
```

A system that originally looked like:

```text
2019
─────\____/────────
```

may later become:

```text
2019                                      2026
─────\____/───────────────────────────────\____/────
```

This provides qualitatively new information.

Two events separated by ΔT imply a family of possible orbital periods:

```text
Pₙ = ΔT / n
```

where *n* is an integer number of orbital cycles between observations.

For example, two events separated by 900 days could potentially correspond to periods of:

```text
900 d
450 d
300 d
225 d
180 d
...
```

The actual TESS observing history can then reject many of these aliases.

If an alias predicts an additional transit at a time when TESS was observing the star and suitable data show no event, that alias becomes inconsistent with the observations.

Thus every TESS revisit has the potential to convert an old monotransit into a much better constrained planetary candidate.

---

## 4. Relevant Existing Work

### 4.1 Duotransit searches

[Hawthorn et al. (2024)](https://doi.org/10.1093/mnras/stad3783) systematically searched TESS Cycle 1 and Cycle 3 observations for stars displaying one transit in each observing epoch.

They identified 85 duotransit candidates, of which 60 were previously unreported.

This demonstrates two important facts.

First, useful long-period candidates remain recoverable through retrospective cross-epoch analysis.

Second, the candidate-vetting burden is substantial. Hawthorn et al.’s methodology began with 9,718 initial duotransit candidates that required visual inspection.

The proposed research aims to automate part of the association and prioritisation stage rather than simply repeat the same search over later sectors.

---

### 4.2 Transformer detection of non-periodic TESS transits

[Salinas et al. (2025)](https://doi.org/10.1093/mnras/staf347) developed a Transformer architecture that analysed complete unfolded TESS SPOC FFI light curves together with centroid and background time series.

Unlike classical Astronet-style architectures, the model did not require an existing orbital period or phase-folded transit.

Applied to 4.1 million light curves in Sectors 1–26, the system identified:

* 214 new candidate systems;
* 122 multitransit candidates;
* 88 single-transit candidates;
* 4 multiplanet candidates.

The authors explicitly identify analysis of sectors beyond 26 and improved sensitivity to smaller planets as directions for future work.

The study demonstrates that machine learning can identify non-periodic transit-like signals at TESS scale.

It does not, however, solve the general problem of automatically associating individual events observed in separated mission epochs and converting those associations into long-period orbital constraints.

---

### 4.3 EXOVEIL

[Priyanshu (2026)](https://arxiv.org/abs/2606.02778) introduced EXOVEIL, a self-supervised Transformer-based system designed specifically to identify transit events without requiring periodicity.

EXOVEIL trains a world model to predict expected stellar flux. Transit-like deviations are identified in the prediction residuals using matched filtering and subsequently classified.

The reported results include:

* AUC 0.938 on Kepler DR25;
* 179 transit-like signals in a blind Kepler search that were not present in the DR25 TCE catalogue;
* 46 monotransit candidates;
* recovery of 47/47 tested confirmed TESS planets in a zero-shot cross-mission application to the PLATO LOPS2 field.

The paper states that the software is released as `pip install exoveil` with pretrained weights and a candidate catalogue. The [arXiv project record](https://arxiv.org/abs/2606.02778) is the source for this release statement.

EXOVEIL substantially overlaps with the individual-event detection component envisaged in earlier versions of this proposal.

This is useful rather than problematic.

The present project will therefore treat event detection as an upstream input problem, using existing methods such as EXOVEIL, matched filtering or simple event proposal algorithms rather than attempting to make another transit detector the principal research contribution.

---

### 4.4 RAVEN and the remaining period gap

[Lafarga et al. (2026)](https://doi.org/10.1093/mnras/stag512) applied RAVEN to approximately 2.26 million TESS-SPOC FFI stars across Sectors 1–55.

The search combines BLS detection with machine-learning classification and statistical validation.

Its principal search range is explicitly restricted to periods between 0.5 and 16 days.

This illustrates an important division in the literature.

Large-scale automated systems increasingly perform well for conventional periodic planets, while the sparse-event long-period regime requires a different treatment.

---

## 5. Research Gap

The core research gap is therefore defined as:

> Automated association of individually detected transit events across non-contiguous TESS observing epochs.

Existing approaches increasingly address:

> Does this light curve contain a transit?

or:

> Is this periodic signal a planet or false positive?

The proposed project instead addresses:

> Are these two transit events, observed around the same star months or years apart, likely to have been generated by the same planetary body?

The distinction is important.

A good event detector can produce thousands of isolated candidate events.

For long-period discovery, the next challenge becomes organizing those events into physically consistent sequences.

To the best of the literature review conducted for this proposal, no published large-scale system has yet combined:

1. non-periodic individual transit-event detection;
2. learned cross-epoch event association;
3. multiple TESS mission cycles;
4. deterministic TESS-window period-alias elimination; and
5. a sealed temporal validation in which future sectors are hidden until the method is frozen.

The novelty claim is therefore not:

> machine learning can detect TESS transits.

That is established.

The proposed novelty is:

> machine learning can be used to associate independently observed transit events across the multi-year TESS archive.

---

## 6. Primary Research Question

Can a learned cross-epoch event-association model improve the recovery of long-period transiting exoplanets from multi-cycle TESS observations relative to simple deterministic event matching?

---

## 7. Secondary Research Questions

**RQ1 — Association**

Can a learned representation distinguish genuine repeated planetary transits from superficially similar but unrelated events observed around the same star?

**RQ2 — Blinded temporal recovery**

Using only earlier TESS observations, can the system identify systems whose later observations contain a compatible repeat event?

**RQ3 — Period inference**

Once two events are associated, how effectively can the actual TESS observing window eliminate orbital-period aliases?

**RQ4 — Practical efficiency**

Can learned association materially reduce the number of candidate event pairs requiring manual vetting?

**RQ5 — Discovery**

When applied to recent TESS observations, does the method identify credible previously uncatalogued long-period candidates?

RQ5 is explicitly secondary and is not required for methodological success.

---

## 8. Hypotheses

### H1

A learned event-association model will outperform deterministic depth-, duration- and timing-based matching on previously unseen TICs.

### H2

Association performance will improve when the model is supplied with transit morphology rather than summary statistics alone.

### H3

Combining learned event association with deterministic TESS observing-window constraints will recover repeat events from known long-period planets at a useful precision/recall trade-off.

### H4

The model will reduce the human-vetting burden required to recover a fixed proportion of known long-period systems.

---

## 9. Deliberate Scope Reduction

The project is divided into core, contingent, and future components.

### 9.1 Core research

The minimum publishable unit consists of:

1. constructing a multi-sector transit-event dataset;
2. implementing a deterministic event-association baseline;
3. training and evaluating a learned event-association model;
4. implementing deterministic orbital-period alias inference;
5. executing the sealed Sectors 80–105 temporal validation;
6. running a limited injection/recovery experiment.

If these six components are completed rigorously, the project has achieved its primary scientific objective.

---

### 9.2 Contingent extension

Only after the core methodology performs satisfactorily:

7. apply the frozen pipeline to Sector 106;
8. identify uncatalogued candidates;
9. conduct detailed candidate-specific vetting.

---

### 9.3 Explicitly out of scope for v1

The following are deferred:

* developing a novel large Transformer for single-event detection;
* exhaustive TGLC searching;
* exhaustive TARS searching;
* pixel-level deep-learning models;
* graph neural networks for multi-event clustering;
* systematic ZTF/ATLAS/ASAS-SN processing of the entire target population;
* occurrence-rate measurement;
* a full multidimensional survey selection function;
* automated follow-up observation scheduling;
* polished candidate-report infrastructure.

These remain potential subsequent studies.

---

## 10. Target Population

The primary population of interest is planets with:

```text
P > 27 days
```

with particular emphasis on approximately:

```text
30–500 days
```

The upper boundary is not a hard astrophysical cut.

Rather, as periods become very long, the probability of observing two events declines and the period-alias degeneracy becomes increasingly severe.

Targets will initially be restricted to stars with:

* observations in at least two distinct TESS epochs;
* sufficient photometric quality;
* reliable source identification;
* reasonable stellar metadata;
* at least one plausible transit-like event.

This dramatically reduces the search space relative to an indiscriminate all-star archive search.

---

## 11. Data Sources

### 11.1 TESS light curves

The primary dataset will consist of publicly available TESS light curves from MAST.

Candidate products include:

* SPOC;
* TESS-SPOC FFI;
* QLP;
* potentially TGLC for specific robustness tests.

For the core methodological study, one primary photometric product will be chosen and held fixed.

Alternative reductions will initially be used only for candidate verification.

---

### 11.2 Event detections

Event proposals may be produced using one or more of:

* EXOVEIL;
* matched-filter transit templates;
* simple robust local-dip detection;
* published candidate-event catalogues;
* known planet ephemerides for labelled examples.

The objective is high recall, not novelty.

The event detector should deliberately generate a somewhat impure candidate population because Model B is intended to perform much of the subsequent ranking.

---

## 12. Data Representation

Every candidate event will be represented by an event record containing at minimum:

```text
TIC_ID
sector
local_time_array
local_flux_array
stellar_metadata
quality_information
```

Optional additions include:

```text
centroid_X
centroid_Y
background_flux
Gaia_crowding_metrics
```

Importantly:

> TIC ID itself will never be supplied as a predictive feature.

It is an identifier used to construct valid candidate pairs and enforce data partitions.

---

## 13. Constructing the Association Dataset

### 13.1 Positive pairs

Known planets observed in multiple TESS epochs provide natural positive examples.

For example:

```text
Planet X
  │
  ├── transit in Sector 12
  ├── transit in Sector 39
  └── transit in Sector 66
```

generates:

```text
(S12, S39) → positive
(S12, S66) → positive
(S39, S66) → positive
```

---

### 13.2 Negative pairs

Negative-pair design is likely to be one of the most important methodological decisions.

Trivial negatives would make the task unrealistically easy.

The priority will therefore be hard negatives, such as:

* two unrelated transit-like artefacts around the same star;
* a planetary transit paired with a stellar-variability event;
* two events with similar depths but incompatible durations;
* plausible-looking eclipsing-binary events;
* events that are morphologically similar but physically incompatible;
* events whose timing cannot be reconciled with the observing window.

For evaluation, negatives should primarily come from the same target context in which the association model will actually operate.

Randomly pairing completely unrelated stars will not be allowed to dominate the reported performance.

---

## 14. Preventing Leakage

### 14.1 TIC-level splitting

All events from a TIC must belong exclusively to one partition.

If:

```text
TIC 123456
```

appears in training, no event from that star may appear in validation or test.

This prevents the network from learning star-specific behaviour.

---

### 14.2 Temporal splitting

The principal benchmark will additionally enforce chronology.

Observations through Sector 79 will constitute the development period.

Sectors 80–105 will form the sealed temporal holdout.

---

### 14.3 Catalogue leakage

Modern exoplanet catalogues contain information derived from later observations.

Using current catalogue information carelessly could leak the answer into the experiment.

Mitigation will include:

* TIC-level exclusion of holdout systems from training;
* freezing training labels before holdout inspection;
* recording catalogue versions and download dates;
* where practical, reconstructing historically available candidate information;
* avoiding claims of a truly prospective historical experiment where that reconstruction is incomplete.

The experiment will therefore be described as:

> data-blinded pseudo-prospective validation.

---

## 15. Deterministic Baseline

Before any neural association model is trained, a simple baseline will be constructed.

For two events A and B around the same star, calculate quantities such as:

**Relative depth difference**

```text
|dA - dB| / mean(dA,dB)
```

**Relative duration difference**

```text
|TA - TB| / mean(TA,TB)
```

**Morphological correlation**

Correlation between normalized local transit windows.

**Timing plausibility**

Whether ΔT permits a physically reasonable set of orbital aliases.

The simplest matcher may resemble:

```text
compatible depth
AND
compatible duration
AND
compatible morphology
AND
valid timing
```

with thresholds selected using training data.

This baseline is scientifically important.

The project only needs a neural association model if it adds value beyond these simple rules.

---

## 16. Association Model

The proposed ML model will use a Siamese or contrastive architecture.

```text
Event A
   ↓
shared encoder
   ↓
embedding A
       \\
        ───→ comparison network → P(same planet)
       /
embedding B
   ↑
shared encoder
   ↑
Event B
```

The event encoder may initially be a small one-dimensional CNN.

Complex Transformers are not required unless simpler architectures clearly fail.

A candidate architecture may contain:

* 3–5 1D convolutional blocks;
* normalization;
* pooling;
* a compact event embedding;
* absolute embedding difference;
* optional summary features;
* a small dense association head.

Total parameter count should remain modest.

---

## 17. Inputs to the Association Model

### Primary input

Normalized local transit morphology.

This allows the model to compare:

* overall shape;
* ingress;
* bottom curvature;
* egress;
* asymmetry;
* local noise.

### Optional scalar features

* transit depth;
* duration;
* S/N;
* stellar radius;
* stellar density;
* crowding;
* magnitude.

Timing information should initially be handled largely by the deterministic orbital layer rather than allowing the neural model to memorize common period distributions.

---

## 18. Association Model Output

The model predicts:

> P(events A and B arise from the same transit-producing object)

It does not predict:

> P(the star contains a planet).

It does not directly determine the orbital period.

It also does not replace astrophysical false-positive vetting.

These distinctions will be explicit throughout the study.

---

## 19. Orbital-Alias Inference

For an associated event pair occurring at:

```text
t₁ and t₂
```

calculate:

```text
ΔT = t₂ - t₁.
```

Candidate periods are:

```text
Pₙ = ΔT / n
```

for permitted positive integers *n*.

A lower period boundary of approximately 27 days will initially define the long-period search regime.

---

## 20. TESS Window-Function Filtering

Every candidate alias will be checked against the actual observation times for that TIC.

For each proposed period:

1. predict all transit epochs;
2. determine whether TESS was observing at each predicted epoch;
3. determine whether sufficient valid cadences exist;
4. determine whether a corresponding event was observed;
5. reject aliases predicting confidently observable but absent events.

Example:

```text
Alias A predicts:
2019     2020     2021     2022
  V        V        V        V
TESS:     DATA              DATA
           │                 │
        no transit        transit
→ alias inconsistent
```

This stage is deterministic.

There is little scientific benefit in asking the neural network to learn arithmetic that can be calculated exactly.

---

## 21. Blinded Temporal Validation

This is the primary evaluation.

### Development period

Sectors 1–79 will be used for:

* preprocessing decisions;
* event-dataset construction;
* association-model development;
* hyperparameter selection;
* deterministic-baseline tuning;
* threshold selection.

No Sectors 80–105 light-curve measurements will be used during model development.

---

### Sealed holdout

Sectors 80–105 will remain hidden until the pipeline is frozen.

Before unblinding, the following must be versioned and recorded:

* preprocessing code;
* event-selection rules;
* model weights;
* association threshold;
* deterministic baseline thresholds;
* candidate-ranking rules;
* orbital-alias algorithm.

---

## 22. Pseudo-Prospective Experiment

For eligible targets with observations before Sector 80:

```text
AVAILABLE
Sector 32
──────\____/────────
          ↑
      event A
HIDDEN
Sector 93
────────────\____/────
                ↑
            event B
```

The analysis will ask:

1. Was event A recognized as worthy of retention?
2. After unblinding, was event B detected?
3. Did Model B correctly associate A and B?
4. Where did the true association rank among competing event pairs?
5. Did the association improve the period constraints?
6. How did ML performance compare with the deterministic matcher?

This approximates deployment before the later observations existed.

---

## 23. Headline Evaluation Metrics

Generic accuracy will not be the headline result.

### Primary metric

True-repeat retrieval rate at fixed candidate burden.

For example:

> What proportion of real later transits are correctly recovered if a reviewer is willing to inspect the top 100 candidate associations?

### Additional association metrics

* Precision-Recall AUC;
* top-1 association accuracy;
* top-5 retrieval rate;
* Mean Reciprocal Rank;
* recall at fixed false-association rate;
* precision at fixed human-vetting budget.

### Full-pipeline metrics

* fraction of known long-period systems recovered;
* fraction of true repeat events retrieved;
* number of candidate pairs per recovered planet;
* number of viable period aliases before and after window filtering;
* false associations per 1,000 targets.

These metrics connect more directly to the scientific workflow than classification accuracy alone.

---

## 24. Minimum Injection-Recovery Study

A full occurrence-rate-quality completeness experiment is explicitly out of scope.

However, a limited injection study is necessary to determine where the method works.

The initial experiment will vary approximately three dimensions:

### Transit signal strength

For example:

* low S/N;
* medium S/N;
* high S/N.

### Transit morphology/depth

Representative Neptune/Jupiter-scale events.

### Temporal separation

For example:

* months;
* approximately 1–2 years;
* multiple years.

Synthetic transit events will be injected into real TESS light curves.

The objective is to measure:

> At what signal strengths and epoch separations can Model B reliably recognize two manifestations of the same underlying transit shape?

This is an association-completeness study, not a full TESS occurrence-rate calculation.

---

## 25. Ablation Studies

If the learned model outperforms the baseline, ablation studies will identify why.

Compare:

* **A:** Depth + duration only.
* **B:** Depth + duration + morphology correlation.
* **C:** Learned flux embedding.
* **D:** Learned flux embedding + stellar metadata.
* **E:** Learned flux embedding + contamination information.

This determines whether the neural network is finding meaningful additional structure.

---

## 26. Critical Stop/Go Decision

At the end of the association-model development phase, compare Model B against the deterministic baseline on completely unseen TICs.

### GO criterion

Continue with ML as the principal methodology only if it produces a meaningful improvement in at least one operationally important measure such as:

* true-repeat recall at fixed false-association rate;
* candidate burden at fixed recall;
* top-k retrieval of the correct match.

The improvement should be practically meaningful, not merely statistically detectable.

### STOP/PIVOT criterion

If simple depth/duration/morphology rules perform equally well:

> stop tuning the neural model.

The project becomes:

> A systematic deterministic framework for multiepoch TESS transit association and blinded long-period recovery.

That remains scientifically useful.

This prevents the project becoming hostage to the assumption that ML must win.

---

## 27. Candidate-Specific False-Positive Vetting

Only after the core holdout experiment will detailed vetting be implemented for discovery candidates.

For promising systems, inspect:

* odd/even differences where multiple events exist;
* secondary eclipses;
* event shape;
* Gaia neighbours;
* crowding;
* centroid movement;
* TESS difference images;
* inferred companion radius;
* known eclipsing binaries;
* known TOIs and planets.

TESS’s approximately 21-arcsecond pixels make contamination particularly important. ([TESS Instrument Handbook](https://heasarc.gsfc.nasa.gov/docs/tess/documentation.html))

---

## 28. Alternative TESS Reductions

For final candidates only, check whether the event appears in alternative reductions such as:

* SPOC;
* QLP;
* TGLC;
* other accessible FFI reductions.

A genuine astrophysical event surviving different extraction pipelines is more convincing than one appearing in only a single reduction.

This is a vetting tool in v1, not another model input pipeline.

---

## 29. External Photometry

ZTF, ATLAS and ASAS-SN will be treated as candidate-specific resources rather than archive-scale inputs.

For a high-ranked candidate with multiple period aliases:

```text
TESS events
     ↓
period aliases
     ↓
predicted historical transit dates
     ↓
query external survey coverage
```

A ground-based non-detection may reject an alias only where:

* the survey observed the target;
* the observation covered the predicted transit;
* the expected transit depth was detectable.

This work is contingent and will not delay the core experiment.

---

## 30. Sector 106 Discovery Extension

Sector 106 was observed from 11 July to 9 August 2026. ([TESS sector information](https://heasarc.gsfc.nasa.gov/docs/tess/sector.html))

It will serve as a recent search cohort only if the association methodology passes the core evaluation.

For each eligible Sector 106 target:

1. identify all historical TESS observations;
2. detect transit-like events in Sector 106;
3. retrieve historical event candidates;
4. score all plausible cross-epoch associations;
5. generate period aliases;
6. reject aliases using the full observing history;
7. cross-match known catalogues;
8. manually vet the highest-ranked previously uncatalogued candidates.

The discovery search therefore asks two symmetric questions:

> Does Sector 106 contain the repeat transit of an unresolved historical event?

and:

> Does a new Sector 106 event match something hidden in earlier TESS observations?

---

## 31. Computational Strategy

The core research does not require large-scale GPU infrastructure.

Event windows and Siamese CNN models are small enough for local development.

The primary computational challenges are data access and any large transit-search operations.

The proposed environment split is:

### Local machine

Use for:

* model development;
* association training;
* plotting;
* debugging;
* ablations;
* analysis;
* manuscript generation.

### STScI TIKE

Use for:

* archive-side TESS data access;
* construction of TIC/sector manifests;
* preprocessing large numbers of light curves;
* extraction of compact event windows;
* selected transit-search workloads.

TIKE is a free STScI JupyterHub environment hosted alongside MAST’s AWS datasets, avoiding large local downloads. It currently provides four CPU cores per user; a MyST account is required to access the service. ([STScI TIKE](https://timeseries.science.stsci.edu/))

The intention is to move compact derived event data, rather than years of raw TESS files, into the local ML workflow.

---

## 32. Work Packages

### WP1 — Multi-sector event dataset

**Tasks**

* select primary TESS photometric product;
* create TIC-to-sector coverage table;
* retrieve known long-period systems;
* generate/extract individual event windows;
* construct positive and negative event pairs;
* enforce TIC-level partitions.

**Deliverable**

Reproducible event-pair dataset.

---

### WP2 — Association model

**Tasks**

* implement deterministic baseline;
* implement small Siamese CNN;
* construct hard negatives;
* evaluate on unseen TICs;
* perform basic ablations.

**Deliverable**

Association-model benchmark.

**Decision gate**

ML must demonstrate meaningful advantage over the deterministic baseline to remain central to the paper.

---

### WP3 — Orbital inference

**Tasks**

* calculate period aliases;
* reconstruct target observing windows;
* eliminate incompatible aliases;
* quantify alias reduction.

**Deliverable**

Deterministic period-inference module.

---

### WP4 — Sealed temporal validation

**Tasks**

* freeze code and thresholds;
* finalize pre-unblinding predictions;
* reveal Sectors 80–105;
* measure repeat-event retrieval;
* compare Model B against deterministic association.

**Deliverable**

Primary scientific experiment.

---

### WP5 — Limited injection/recovery

**Tasks**

* inject representative repeated transit events;
* vary signal strength and epoch separation;
* measure association recovery.

**Deliverable**

Defined operating regime for the method.

---

### WP6 — Discovery extension

Conditional on WP1–5 success.

**Tasks**

* search Sector 106;
* associate with historical events;
* catalogue cross-match;
* candidate-specific vetting.

**Deliverable**

Optional candidate catalogue.

---

## 33. Timeline

### Month 1 — Dataset and baseline

* reproduce known duotransit examples;
* build TESS sector-coverage tooling;
* test EXOVEIL/simple event proposals;
* construct first labelled event pairs;
* implement deterministic association baseline.

**Required milestone**

Known multiepoch planets can be reliably represented as event pairs.

---

### Month 2 — Model B

* implement event encoder;
* train Siamese classifier;
* design hard negatives;
* establish unseen-TIC evaluation.

**Required milestone**

Clear comparison between learned and deterministic association.

---

### Month 3 — Decide whether ML survives

* ablation studies;
* error analysis;
* improve only problems identified empirically.

**Decision**

If Model B does not materially outperform the deterministic baseline, freeze it and pivot the paper accordingly.

No open-ended architecture search.

---

### Month 4 — Period inference and injections

* implement observing-window reconstruction;
* period aliases;
* limited injection/recovery;
* freeze methodology.

---

### Month 5 — Blinded holdout

* register/freeze predictions;
* unblind Sectors 80–105;
* evaluate;
* error analysis;
* main paper figures and tables.

---

### Month 6 — Paper first, discovery second

**Primary priority:**

* complete methodological manuscript.

If results justify continuation:

* Sector 106 search;
* manual vetting;
* candidate appendix or follow-up paper.

---

## 34. Success Criteria

The project succeeds if it produces a rigorous answer to:

> Does learned cross-epoch transit association improve long-period TESS recovery?

A new planet is not required.

### Outcome A — ML succeeds

Model B materially outperforms simple event matching and performs well in the temporal holdout.

**Primary result:**

learned event association is useful for long-period TESS recovery.

---

### Outcome B — ML adds little

Simple deterministic event association performs equally well.

**Primary result:**

morphology-aware deterministic association is sufficient for this regime.

The negative ML result is scientifically informative because the benchmark and blinded test remain valid.

---

### Outcome C — Association itself fails

Even known repeat systems cannot be recovered at an acceptable candidate burden.

The research then quantifies why:

* transit S/N;
* morphology instability;
* instrumental differences;
* false-event density;
* crowding.

This provides a scientifically useful limit on archive-based long-period recovery.

---

## 35. Expected Paper

### Proposed title

Learning to Link Transit Events Across the TESS Extended Mission

### Alternative

Multi-Epoch Transit Association for Long-Period Planet Recovery in TESS

### Proposed structure

1. Introduction
2. Long-period TESS detection problem
3. Related work
4. Multi-sector event dataset
5. Deterministic association baseline
6. Learned event-association model
7. Orbital-period alias inference
8. Data-blinded temporal experiment
9. Injection/recovery
10. Discussion
11. Optional Sector 106 search
12. Conclusions

---

## 36. Desired Abstract-Level Result

A strong final result would read approximately:

> We present a framework for associating isolated transit events observed during separate TESS observing epochs. Using observations through Sector 79 for development, we freeze the association model and evaluate it against the previously unseen Sectors 80–105. At fixed recall, the learned association model reduces the number of false event pairings requiring manual inspection by X% relative to matching based only on transit depth, duration and morphology. Combining event association with the TESS observing window reduces the median number of viable orbital aliases from Y to Z. The method recovers N% of eligible known long-period systems in the sealed temporal test.

If Sector 106 produces interesting candidates, an additional sentence can be added.

It is not required.

---

## 37. Reproducibility

The research repository will record:

* exact TIC lists;
* sector manifests;
* catalogue versions;
* observation-product versions;
* preprocessing parameters;
* event-selection rules;
* positive/negative-pair construction;
* train/validation/test TIC assignments;
* random seeds;
* model checkpoints;
* deterministic thresholds;
* injection parameters;
* holdout-unblinding date.

The Sectors 80–105 evaluation protocol should ideally be committed or pre-registered before the holdout is examined.

This is intended to make the pseudo-prospective claim auditable.

---

## 38. Risks

### Risk 1 — EXOVEIL or another existing detector already solves too much of the problem

This does not undermine the study.

The project deliberately treats individual-event detection as upstream infrastructure.

The claimed contribution begins at cross-epoch association.

---

### Risk 2 — Too few real positive event pairs

**Mitigation**

* use all known multiepoch long-period planets;
* use valid repeated events from shorter-period planets where appropriate for representation pretraining;
* use synthetic injections for augmentation;
* reserve real long-period systems for final evaluation.

---

### Risk 3 — Negative-pair construction is unrealistic

**Mitigation**

Prioritize hard negatives from realistic within-target event populations.

Report performance separately for:

* random negatives;
* morphology-matched negatives;
* same-TIC hard negatives.

The last category should drive scientific conclusions.

---

### Risk 4 — ML overfits transit depth and duration

**Mitigation**

Compare against explicit depth/duration baselines and remove scalar features in ablation experiments.

If the neural network adds nothing beyond them, prefer the simpler method.

---

### Risk 5 — Later catalogue information leaks into training

**Mitigation**

Strict TIC holdout, catalogue versioning and separation between labels used to construct development data and information used to evaluate the temporal test.

---

### Risk 6 — Scope expansion

This is now treated as a primary methodological risk.

**Mitigation**

The following are not required before the blinded test:

* TARS;
* full TGLC survey;
* ZTF automation;
* pixel CNNs;
* graph models;
* exhaustive completeness;
* occurrence rates;
* automated candidate reports.

Any of these added before WP4 must replace existing scope rather than simply expanding it.

---

### Risk 7 — Rapidly changing literature

EXOVEIL demonstrates that relevant methods can appear during the project.

**Mitigation**

Repeat the focused literature search:

* before model design is frozen;
* before manuscript drafting;
* immediately before submission.

The paper’s novelty claim will remain restricted to the specific cross-epoch association problem unless newer research requires further narrowing.

---

## 39. Future Work

If the core study succeeds, several larger projects follow naturally.

### Full archive event graph

Represent every transit-like event as a node and event-association probabilities as graph edges.

Infer multievent planetary sequences jointly.

### Smaller long-period planets

Develop event models specifically for shallow sub-Neptune or super-Earth signals.

### TGLC search

Extend the method to Gaia-deblended TGLC photometry and crowded fields.

### Cross-reduction learning

Train models using SPOC, QLP and TGLC as separate views of the same physical event.

### External photometric alias elimination

Systematically use ZTF, ATLAS and ASAS-SN observations to eliminate long-period aliases.

### Continuous deployment

For every newly released TESS sector:

```text
detect new events
      ↓
compare with all historical events
      ↓
update associations
      ↓
update period aliases
      ↓
surface newly resolved systems
```

This would turn the method into an ongoing archive-monitoring pipeline.

---

## 40. Central Contribution

The revised project is intentionally narrower than its original formulation.

It is not:

> Develop an end-to-end deep-learning exoplanet-discovery pipeline.

It is not:

> Invent a better single-transit detector.

It is not:

> Search every TESS star for every possible long-period planet.

The central contribution is:

> Develop and rigorously evaluate a method for associating isolated transit events observed during different TESS mission epochs, allowing historical monotransits to be converted into constrained long-period candidate systems as new observations become available.

The critical experiment is:

> Train and design using Sectors 1–79, freeze the system, then ask whether it can correctly recover repeat events in sealed Sectors 80–105.

Machine learning earns its place only if it demonstrably improves that task over simple deterministic matching.

Sector 106 then provides an opportunity to apply the validated method to recent data, rather than being necessary for the research to succeed.

That makes the project substantially narrower, more falsifiable and more achievable while preserving the part of the original proposal with the strongest potential methodological novelty.

---

## References

<a id="ref-hawthorn-2024"></a>Hawthorn, F. et al. (2024). TESS duotransit candidates from the Southern Ecliptic Hemisphere. *Monthly Notices of the Royal Astronomical Society*, 528(2), 1841–1862. [doi:10.1093/mnras/stad3783](https://doi.org/10.1093/mnras/stad3783).

<a id="ref-lafarga-2026"></a>Lafarga, M. et al. (2026). Automatic search for transiting planets in TESS–SPOC FFIs with RAVEN: over 100 newly validated planets and over 2000 vetted candidates. *Monthly Notices of the Royal Astronomical Society*, 548(3). [doi:10.1093/mnras/stag512](https://doi.org/10.1093/mnras/stag512).

<a id="ref-priyanshu-2026"></a>Priyanshu, P. (2026). One Transit Is All You Need: Detecting Exoplanets Through Learned Stellar Behaviour with EXOVEIL. [arXiv:2606.02778](https://arxiv.org/abs/2606.02778).

<a id="ref-rodel-2024"></a>Rodel, T. et al. (2024). TIaRA TESS 1: estimating exoplanet yields from Years 1 and 3 SPOC light curves. *Monthly Notices of the Royal Astronomical Society*, 529(1), 715–731. [doi:10.1093/mnras/stae474](https://doi.org/10.1093/mnras/stae474).

<a id="ref-salinas-2025"></a>Salinas, H. et al. (2025). Exoplanet transit candidate identification in TESS full-frame images via a transformer-based algorithm. *Monthly Notices of the Royal Astronomical Society*, 538(3), 2031–2049. [doi:10.1093/mnras/staf347](https://doi.org/10.1093/mnras/staf347).

<a id="ref-shallue-2018"></a>Shallue, C. J. & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. *The Astronomical Journal*, 155(2), article 94. [doi:10.3847/1538-3881/aa9e09](https://doi.org/10.3847/1538-3881/aa9e09).

<a id="ref-villanueva-2019"></a>Villanueva, S., Dragomir, D. & Gaudi, B. S. (2019). An Estimate of the Yield of Single-transit Planetary Events from the Transiting Exoplanet Survey Satellite. *The Astronomical Journal*, 157(2), article 84. [doi:10.3847/1538-3881/aaf85e](https://doi.org/10.3847/1538-3881/aaf85e).

---

## Citation and Claim Validation

Validated 2 September 2026 against publisher or archive records, arXiv full text/abstracts, and NASA/STScI documentation.

All seven references are real and correctly attributed. The cited results were checked as follows:

* Hawthorn et al.: 85 duotransit candidates, 60 new, and 9,718 initial candidates inspected.
* Rodel et al. (TIaRA): 50% of simulated detectable planets with periods over 25 days and 76% over 100 days were monotransits.
* Salinas et al.: 4.1 million Sector 1–26 light curves and the reported 214, 122, 88 and 4 candidate counts.
* Priyanshu (EXOVEIL): AUC 0.938, 179 transit-like signals, 46 monotransit candidates, and 100% recovery of 47 confirmed TESS planets in the tested PLATO LOPS2 sample.
* Lafarga et al. (RAVEN): approximately 2.26 million stars from Sectors 1–55 and a 0.5–16 day search range.
* TESS and TIKE operational facts: approximately 27-day sectors, Sector 106 dates, approximately 21 arcsecond pixels, free access, MAST AWS availability and four CPU cores.

The blockquotes in this proposal are project questions or framing statements, not quotations from the cited literature. No cited result was found to be materially misrepresented. The eight-year statement refers to the archive-wide time span; individual stars generally have intermittent sector coverage rather than continuous observations.
