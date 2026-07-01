# Railway Sleeper Crack Detection: Reducing Ballast-Gap False Positives in a YOLOv8-OBB + SAHI Pipeline

A research-and-implementation design document. The central thesis is stated up front because it should drive every design decision below:

> **The dominant failure is context collapse, not weak feature extraction.** A crack and a ballast gap are both "thin, dark, elongated structures." What separates them is *where they sit* — a crack is on a smooth concrete surface, a ballast gap is in a rough stone field. SAHI slicing discards exactly the spatial context that makes them separable, and the standard training recipe never forced the model to learn a decision boundary against ballast. Therefore the highest-leverage fixes restore context and reshape the negative distribution; backbone/neck/attention upgrades are secondary refinements, not the cure.

Keep this hierarchy in mind: the cheapest, highest-impact interventions (Sections 8, 9, 10) attack the root cause directly. The architecture changes (Sections 2–7) improve the ceiling. Treat anyone who reverses that ordering with suspicion, including this document if read out of order.

---

## 1. Why YOLOv8-OBB + SAHI confuses ballast gaps with cracks

Six concrete mechanisms, roughly in order of contribution:

**(a) Context collapse from slicing.** SAHI crops a large image into independent tiles (e.g. 640×640). Inside a single tile near the sleeper-ballast boundary, the network's effective receptive field never spans enough area to answer "am I on a sleeper or in the ballast?" The only cue left is local texture, and locally a ballast gap and a crack are near-identical low-intensity elongated regions. This is why your false positives cluster at boundaries — that is precisely where a tile contains a mix of smooth-concrete and rough-stone texture but lacks the global frame to disambiguate.

**(b) Geometric prior is non-discriminative.** The OBB head is explicitly rewarded for producing high-aspect-ratio oriented boxes (that is the entire point of OBB for cracks). Ballast gaps are *also* high-aspect-ratio elongated dark regions, so they match the geometric prior that you tuned to catch cracks. Orientation and aspect ratio cannot separate the two classes — they are confounders, not discriminators.

**(c) The negative distribution was never shaped.** In most crack datasets, cracks are annotated and everything else is "background." If ballast regions were not deliberately sampled as *hard negatives*, the classifier's "crack" prototype is defined almost entirely against smooth concrete. The decision boundary against ballast texture was never trained, so at inference any ballast structure that resembles a crack lands on the wrong side of an undefined boundary.

**(d) Overlap voting amplifies persistent FPs.** SAHI's slice-and-merge step (NMS / NMM / greedy) treats detections that recur across overlapping tiles as more reliable. A real crack benefits from this. But a *stable* ballast gap that appears consistently across several overlapping tiles gets its confidence reinforced the same way — overlap consistency rewards the wrong thing when the wrong thing is spatially stable.

**(e) Texture statistics are never explicitly encoded.** Concrete is low-variance and smooth; ballast is high-variance, multi-scale stone texture. A model that classifies a candidate purely from its own pixels, with no representation of the *surrounding texture statistics*, throws away the single most physically reliable cue. Standard YOLOv8 features encode this only implicitly and weakly at tile scale.

**(f) Resolution/receptive-field tension.** Thin cracks (often 1–3 px wide) need high-resolution, high-frequency-sensitive features → favors small tiles and shallow strides. Crack-vs-ballast discrimination needs long-range context → favors large receptive fields and global features. SAHI improves the first (good for recall) by worsening the second (bad for precision). The pipeline is currently sitting on the wrong side of this trade-off for precision.

The takeaway for the rest of the document: **(a), (c), (d) are the FP drivers and are fixable cheaply; (b), (e), (f) set the accuracy ceiling and need architecture work.**

---

## 2. Architectural modifications to YOLOv8-OBB

These are framework-level changes to the detector itself, ordered by expected impact.

**2.1 Global-context conditioning branch (the key change).** Run a lightweight encoder on the *downsampled full frame* in parallel with tile inference. It produces (i) a coarse sleeper-region segmentation and (ii) a global context embedding. Inject this into each tile's neck features via FiLM (feature-wise linear modulation): for tile feature map `F`, compute per-channel scale/shift `(γ, β)` from the cropped global embedding at that tile's location and apply `F' = γ ⊙ F + β`. This restores the context SAHI removed, at the cost of one cheap forward pass on a small image. Directly attacks failure (a).

**2.2 Sleeper-region gating head.** An auxiliary semantic head predicting `{sleeper surface, ballast, background}`. At inference, project this mask onto each detection and re-weight confidence by sleeper-membership probability of the box footprint: `conf' = conf · p(sleeper | box)^α`. Off-sleeper detections are suppressed without a hard threshold. This is Section 8 expressed as an architectural component. Attacks (a) and (d).

**2.3 Add a P2 detection level.** YOLOv8 starts detection at P3 (stride 8). For 1–3 px cracks, P3 has already pooled away the structure. Add a P2 level (stride 4) to the FPN/neck and the head. Improves thin-crack recall substantially; costs memory and latency because P2 is large. Pair with the context branch so you do not have to shrink tiles to recover recall.

**2.4 Asymmetric (strip) convolutions in the head.** Replace one `3×3` in the head stem with parallel `1×k` and `k×1` depthwise branches (k = 7 or 9), summed. This matches the geometry of elongated cracks and increases directional sensitivity at near-zero parameter cost. Helps separate "thin line on smooth surface" responses from isotropic ballast texture.

**2.5 Texture-contrast side feature.** A tiny branch computing local variance / high-frequency energy (e.g. a fixed Laplacian + learned `1×1`, or a few learnable Gabor-like filters) pooled around each candidate, concatenated into the classification head. Gives the model the explicit concrete-vs-ballast texture cue from failure (e). Cheap and physically motivated; good ablation candidate and a publishable micro-contribution.

---

## 3. Backbone modifications

Requirements pull in two directions: **fine high-frequency detail** for thin cracks (favors convolutional, high-resolution early stages) and **long-range context** for crack-vs-ballast (favors attention / large kernels later). The right answer is a hybrid, not a wholesale swap to a transformer.

| Backbone | Strength here | Weakness here | Verdict |
|---|---|---|---|
| ConvNeXt (7×7 depthwise) | Large effective RF with conv efficiency; preserves detail | Heavier than CSPDarknet; no explicit global attention | Strong "balanced" choice |
| RepViT / RepVGG-style | Reparameterized → fast single-branch inference, edge-deployable | Less global context on its own | Best for deployment-constrained variant |
| EfficientViT | Linear (ReLU/softmax-linear) attention → global context at near-linear cost on high-res tiles | Slightly weaker on the finest detail than pure conv stems | Best context/cost trade-off |
| Swin (shifted windows) | Multi-scale window attention, strong accuracy | Window partition can fragment a thin crack that spans a window boundary; slow | "Max accuracy" variant only |

**Recommendation:** a hybrid backbone — **convolutional early stages** (a RepViT or ConvNeXt-style stem at stride 2/4 to preserve thin-crack detail) followed by **lightweight linear-attention later stages** (EfficientViT-style) for global context. Rationale: detail must be captured *before* downsampling destroys it (conv early), context can be aggregated *after* (attention late). This is the backbone drawn in the architecture figure. For the deployment paper, also report a pure-RepViT variant for an honest accuracy-vs-latency curve. Avoid plain Swin as the primary backbone specifically because window boundaries cut thin structures — if you use it, use a variant with overlapping windows or add a conv stem to repair boundary continuity.

---

## 4. Neck modifications

| Neck | What it buys | Cost |
|---|---|---|
| BiFPN | Learnable weighted bidirectional fusion of P2–P5; balances fine vs coarse | Low-moderate |
| AFPN (asymptotic FPN) | Progressive adjacent-scale fusion, reduces semantic gap, preserves fine detail | Moderate |
| DyHead | Unifies scale-, spatial-, and task-aware attention at the head | Moderate |
| Attention/transformer top-block | Scene-level context aggregation at P5 (sleeper-vs-ballast) | Moderate |

**Recommendation:** **AFPN (or BiFPN) spanning P2–P5 + DyHead on top.** AFPN's progressive fusion is well-suited to keeping the fine P2 detail alive while still importing P4/P5 context. DyHead is particularly apt: its *spatial-aware* attention concentrates on thin structures, its *scale-aware* attention handles cracks of varying width, and its *task-aware* attention lets the OBB and classification sub-tasks specialize. Add a small global-context block at the top of the neck (or rely on the FiLM injection from 2.1) so scene context reaches every level. Concretely: `backbone → AFPN(P2–P5) → FiLM(context) → CoordAttn/StripPool → DyHead → heads`.

---

## 5. Attention mechanisms for thin cracks

The geometry to exploit is elongation. Isotropic attention (vanilla spatial attention) wastes capacity; *directional/axial* attention is the right family.

- **Coordinate Attention (CA)** — *top pick.* Factorizes attention into two 1-D encodings along H and W, capturing long-range dependence along the elongation axis while keeping precise positional information. Cheap, drop-in, ideal for elongated cracks.
- **Strip pooling** — pools along horizontal/vertical strips (designed for long thin structures). Pairs naturally with CA in the neck/head.
- **Deformable conv (DCNv2/DCNv3) or deformable attention** — sampling points follow curved, branching cracks; use in the crack head where geometry is irregular.
- **CBAM** — channel + (isotropic) spatial attention; cheaper, weaker on elongation. Useful as a baseline ablation, not the final choice.
- **Axial / strip self-attention** — full long-range along a row/column; heavier but maximal directional context.

**Recommendation:** Coordinate Attention + strip pooling in the neck, and deformable conv in the crack-specific head. Run CBAM and vanilla spatial attention as ablation baselines so reviewers see the directional gain is real, not generic "attention helps."

---

## 6. Loss-function modifications to reduce false positives

**6.1 Rotated-box regression — switch to a Gaussian-distribution loss.** Thin cracks are extreme-aspect-ratio oriented boxes, the exact regime where IoU-based rotated losses become unstable (a tiny angle error collapses IoU). Replace ProbIoU/rotated-IoU with **KLD (Kullback–Leibler Divergence)** or **GWD (Gaussian Wasserstein Distance)**, which model the box as a 2-D Gaussian and are provably more stable and scale-invariant for high-aspect-ratio objects. This is well-motivated, citable, and directly relevant — likely a real mAP gain on thin cracks and a clean ablation.

**6.2 Classification — IoU-aware focal reshaping.** Replace plain BCE with **Varifocal Loss (VFL)** or **Quality Focal Loss (QFL)**. VFL down-weights easy negatives and emphasizes high-quality positives via IoU-aware targets; this concentrates learning on the hard ballast-gap negatives instead of the trivial smooth-concrete background. Direct FP-rate lever.

**6.3 Explicit crack-vs-ballast contrastive term.** Add a supervised contrastive loss on a projection head that pulls crack embeddings together and pushes them away from mined ballast-gap embeddings. This carves a decision boundary in feature space precisely where failure (c) said none existed. Treat mined ballast gaps as an explicit "hard background" set (see Section 10).

**6.4 Region-conditioned penalty.** Add a term that penalizes positive detections falling outside the predicted sleeper mask: `L_gate = λ · Σ p(crack) · (1 − p(sleeper | box))`. This trains the gating from 2.2/8 end-to-end rather than only applying it post-hoc.

**6.5 Auxiliary segmentation losses.** For the crack-mask head use **Focal-Tversky** (tune β > 0.5 to favor recall on thin structures, or < 0.5 to favor precision) and for the sleeper-mask head use Dice + BCE. Tversky's recall/precision knob is valuable for 1–3 px structures where Dice alone is dominated by background.

**6.6 Multi-task balancing.** Use uncertainty-based weighting (Kendall et al., homoscedastic task uncertainty) so detection, segmentation, and classification losses self-balance instead of hand-tuned weights.

---

## 7. Multi-task approach (detection + segmentation + classification)

A single shared backbone with branched heads:

```
shared hybrid backbone + AFPN/DyHead neck
      │
      ├── OBB detection head        (KLD/GWD reg + VFL cls)
      ├── crack segmentation head   (Focal-Tversky)   ← pixel-precise thin-structure supervision
      ├── sleeper-region seg head   (Dice + BCE)       ← context gating signal (Sec. 8)
      └── crack-type classification (longitudinal / transverse / map-cracking / spalling)
```

Why this helps FPs specifically:
- **Crack segmentation** forces pixel-precise localization of thin structures; a ballast gap, when segmented, falls into the *non-sleeper* region, providing a strong implicit negative signal that pure detection never sees.
- **Sleeper segmentation** is the context-gating backbone (Section 8) and shares features for free.
- **Crack-type classification** adds fine-grained supervision (regularizes features) and is itself a publishable axis — engineering severity differs by crack type, so a type-aware detector has practical value beyond the paper.

Balance with uncertainty weighting (6.6). Architecturally this is a YOLOv8-seg-OBB hybrid plus a semantic context head — all standard components, novel in combination and framing for this domain.

---

## 8. Incorporating sleeper context to suppress off-sleeper detections

Four mechanisms, from cheapest baseline to fully learned. Report all of them — the cheap one is your strongest single FP reduction and the honest baseline reviewers will expect.

1. **Post-hoc mask filtering (baseline).** Segment the sleeper on the downsampled full frame, threshold, and drop any detection whose center (or majority footprint) lies outside the sleeper mask. Trivial, near-zero cost, and directly removes the boundary FPs. This alone likely eliminates a large fraction of off-sleeper false positives. Use it as the lower bound that everything else must beat.
2. **Soft confidence gating.** Instead of a hard drop, multiply confidence by sleeper-membership probability (2.2). Preserves recall on cracks that physically reach the sleeper edge.
3. **End-to-end region-conditioned training.** Train the gate with `L_gate` (6.4) and FiLM conditioning (2.1) so the detector *learns* to not fire off-sleeper, rather than being corrected afterward.
4. **Structural / geometric prior.** Sleepers are regular rectangular structures at a known orientation and periodic spacing along the track. Exploit this — either a learned sleeper detector defining ROIs, or a periodicity/Hough prior — to restrict the entire pipeline to sleeper ROIs (this also feeds Section 9's ROI-gated slicing). This is the most domain-specific and the most defensible against reviewers asking "why not just crop to the sleeper?"

---

## 9. SAHI slicing-strategy improvements

SAHI is the source of both your recall (good) and your context loss (bad). Tune it to keep the recall and inject back the context.

**9.1 ROI-restricted (context-aware) slicing — the big one.** Use the coarse full-frame sleeper segmentation (Section 8) to slice tiles *only within sleeper ROIs*. Ballast-only tiles are never generated, so the FPs that live there cannot occur. This is the single most direct fix for your stated failure ("FPs mainly near sleeper-ballast boundaries"): you stop feeding the model the regions where it fails.

**9.2 Context-padded tiles.** For each valid tile, feed a slightly larger crop (with margin) but only treat the center region as the valid detection area. Each tile then carries some boundary context, mitigating failure (a) without the full global branch — useful as a lightweight ablation variant.

**9.3 Tile size.** Smaller tiles → higher thin-crack recall but worse context and more FPs; larger tiles → better context, weaker thin-crack recall. Do *not* solve recall by shrinking tiles; instead use a moderate tile size (e.g. 512–768) plus the P2 level (2.3) and context branch (2.1). Sweep `{384, 512, 640, 768}` and report the precision/recall trade-off explicitly.

**9.4 Overlap ratio.** Higher overlap (0.25–0.5) helps cracks that cross tile edges but raises duplicates and feeds overlap-voting FPs (failure d). Sweep `{0.1, 0.2, 0.3, 0.4}`; expect a precision-recall knee. Pair higher overlap with mask-aware fusion (9.6) so the extra votes don't promote stable ballast gaps.

**9.5 Multi-scale slicing.** Run a fine scale (thin-crack recall) and a coarse scale (context) and fuse. Coarse tiles supply approximate sleeper/ballast separation that re-weights fine-tile detections.

**9.6 Fusion.** Replace plain NMS with **NMM (non-max merging)** plus a **mask-weighted re-scoring**: weight each detection by `p(sleeper | box)` and by distance-from-boundary before merging, and explicitly down-weight detections whose support is consistent only within ballast regions. This neutralizes failure (d).

---

## 10. Hard-negative mining targeting ballast gaps

The most important training-side fix, because it builds the decision boundary failure (c) is missing.

**10.1 Bootstrapped FP mining (primary loop).** Train v1 → run on held-out track imagery → collect high-confidence detections that fall *outside* the sleeper mask (auto-verified as ballast FPs, no manual labeling needed) → add them as explicit hard negatives with elevated loss weight → retrain. Iterate 2–3 rounds. This is OHEM/self-training specialized to your exact error mode and is essentially free in labeling cost because the sleeper mask provides automatic verification.

**10.2 Boundary-tile oversampling.** During training, sample tiles straddling the sleeper-ballast boundary at a higher rate than random tiling would. The confusion lives at the boundary; train where the confusion lives.

**10.3 Online hard-example mining (OHEM) / focal weighting.** Within each batch, up-weight the highest-loss negatives (which will be ballast gaps once they are in the sampled distribution). VFL/QFL (6.2) partly does this implicitly; explicit OHEM on mined ballast crops makes it targeted.

**10.4 Embedding-based hardest-negative mining.** Use the contrastive projection head (6.3) to find ballast crops *closest* to the crack manifold and prioritize those — the genuinely confusable cases, not random ballast.

**10.5 Synthetic hard negatives (copy-paste).** Paste real ballast textures adjacent to sleepers and real cracks onto clean sleeper surfaces to controllably balance positives and hard negatives and to stress-test the boundary. Validate that synthetic gains transfer to real test data (report both).

---

## 11. Complete proposed architecture (textual block diagram)

The rendered figure in the chat shows the high-level flow. The expanded block-level spec:

```
                    ┌───────────────────────────────────────────────┐
  full frame  ───►  │ GLOBAL CONTEXT PATH (downsampled, 1 fwd pass)  │
                    │   lightweight encoder                          │
                    │     ├─ sleeper-region seg  ──► sleeper mask     │
                    │     └─ global pooled feats ──► context embed    │
                    └───────────────┬───────────────────┬────────────┘
                                    │ (ROI gate)         │ (FiLM γ,β)
                                    ▼                    │
  full frame ─► ADAPTIVE SAHI SLICER (slice only inside sleeper ROIs, │
                context-padded, multi-scale, overlap-swept)           │
                                    │ tiles                           │
                                    ▼                                 │
                HYBRID BACKBONE                                       │
                  conv stem (RepViT/ConvNeXt) @ stride 2/4  ← detail  │
                  → linear-attn stages (EfficientViT)       ← context │
                  → features P2,P3,P4,P5                              │
                                    │                                 │
                                    ▼                                 │
                AFPN/BiFPN NECK (P2–P5) ◄─────────── FiLM modulation ─┘
                  → Coordinate Attention + strip pooling
                  → DyHead (scale/spatial/task attention)
                                    │
            ┌───────────────┬───────┴────────┬───────────────────┐
            ▼               ▼                ▼                   ▼
      OBB DET HEAD    CRACK SEG HEAD   SLEEPER SEG HEAD   CRACK-TYPE CLS
      KLD/GWD reg     Focal-Tversky    Dice+BCE           CE
      VFL cls         (thin masks)     (gating signal)    (4 classes)
      + strip-conv
      + texture-contrast side feature
      + deformable conv
            │               │                │                   │
            └───────────────┴───────┬────────┴───────────────────┘
                                    ▼
              CONTEXT-GATED FUSION (across tiles)
                conf' = conf · p(sleeper|box)^α · dist-to-boundary weight
                NMM merge (not plain NMS)
                drop ballast-only-consistent detections
                                    │
                                    ▼
                       FINAL CRACK DETECTIONS (OBB + type + mask)
```

Loss (uncertainty-weighted sum): `L = w1·L_OBB(KLD) + w2·L_cls(VFL) + w3·L_crackseg(FTversky) + w4·L_sleeperseg(Dice+BCE) + w5·L_type(CE) + w6·L_contrastive + w7·L_gate`.

---

## 12. Novel research ideas (publication-worthy)

Each is framed as a defensible contribution, not an incremental tweak:

1. **Context-collapse quantification + restoration.** Empirically isolate SAHI's context loss as the *primary* FP driver (a controlled study varying tile size/overlap and measuring off-sleeper FP rate), then show a context-restoration module (FiLM from a global branch) recovers it. The diagnosis itself is a contribution — most SAHI papers treat it as pure recall benefit.
2. **Region-conditioned OBB detection ("defect-on-substrate" framework).** Generalize sleeper-context gating into a reusable paradigm for any "defect confined to a known substrate" problem (cracks on rails/pavement/welds/PCBs). Broadens impact beyond railways → stronger venue fit.
3. **Negative-aware adaptive SAHI.** A slicing strategy that uses a cheap region prior to suppress negative-prone tiles and re-score fusion by region membership. A clean methodological contribution to the SAHI literature.
4. **Texture-statistics-aware discrimination head.** A physically grounded module encoding concrete-vs-ballast texture statistics for fine-grained FP rejection — interpretable and ablatable.
5. **Gaussian rotated losses for extreme-aspect-ratio thin defects.** A focused empirical study of KLD/GWD vs IoU losses in the very-high-aspect-ratio regime that crack OBBs occupy — under-explored; thin cracks are an unusually clean testbed.
6. **OOD framing for ballast gaps.** Treat ballast gaps as out-of-distribution and add an energy-based / Mahalanobis OOD score to reject them — a novel angle that connects defect detection to OOD literature.
7. **A benchmark dataset with annotated ballast-gap hard negatives.** Datasets are publishable in their own right; a railway-sleeper-crack set that explicitly labels ballast-gap confounders would be widely cited and is exactly what the field lacks.
8. **Self-supervised pretraining on unlabeled track imagery** for crack-sensitive features (masked-image-modeling on track video), reducing annotation burden.

Strongest single paper: combine (1) + (2) + (3) — diagnosis, framework, and method — with the multi-task architecture as the vehicle and the dataset (7) as supporting contribution.

---

## 13. Ablation study plan

Build the table cumulatively *and* leave-one-out, on fixed splits, multiple seeds, mean ± std.

**Baseline:** stock YOLOv8-OBB + vanilla SAHI.

**Cumulative additions (each row adds to the previous):**
1. + P2 detection level
2. + hybrid backbone (and separately: each candidate backbone, isolated)
3. + AFPN/BiFPN neck
4. + DyHead
5. + Coordinate Attention + strip pooling (vs CBAM baseline)
6. + KLD/GWD regression loss (vs ProbIoU)
7. + VFL classification (vs BCE)
8. + multi-task seg heads (crack + sleeper)
9. + ROI-restricted SAHI
10. + mask-aware fusion (NMM + re-scoring)
11. + context FiLM conditioning
12. + hard-negative mining loop (rounds 1→3)
13. + contrastive crack/ballast term

**Independent sweeps:** tile size `{384,512,640,768}`; overlap `{0.1,0.2,0.3,0.4}`; gating exponent `α`; Tversky β.

**Metrics — and this is where most crack papers are weak, so make it a strength:**
- Standard: OBB mAP@50, mAP@50:95, precision, recall, PR curves.
- **FP-focused: false-positives-per-image (FPPI) and FROC**, plus a **ballast-gap-specific FP rate** (FPs whose footprint is off-sleeper / in ballast, using the mask as ground-truth region). This metric is the whole point of the paper — report it prominently.
- **FP localization diagnostic:** fraction of FPs that are off-sleeper, before vs after each gating component. This visually proves the mechanism.
- Efficiency: params, GFLOPs, latency/FPS at the deployment resolution, peak memory (P2 is the cost driver — be honest about it).

**Controls:** same train/val/test split with no track-segment leakage between splits (adjacent sleepers are correlated — split by track section, not by frame); ≥3 seeds; identical augmentation across rows except the variable under test.

---

## 14. Expected accuracy gains and computational trade-offs

These are *directional estimates* from the mechanisms and typical literature behavior, not measured results — your ablation (Section 13) must produce the real numbers. They are ordered by expected impact-per-cost so you can prioritize.

| Modification | Expected effect | Compute / cost | Confidence |
|---|---|---|---|
| ROI-restricted SAHI + mask gating (8, 9.1) | **Largest FP reduction** — directly removes the boundary/ballast regions where FPs originate; could cut off-sleeper FPs by a large fraction | Low (one cheap downsampled pass) | High — attacks root cause directly |
| Hard-negative mining loop (10) | Major precision gain; builds the missing crack/ballast boundary | Training-time only (no inference cost); few extra training rounds | High |
| Mask-aware NMM fusion (9.6) | Removes overlap-voting FPs (failure d); modest-to-moderate precision gain | Negligible inference cost | Medium-high |
| Context FiLM conditioning (2.1) | Moderate precision gain on ambiguous boundary tiles | Low (shared global pass) | Medium |
| P2 level (2.3) | Recall gain on thin (1–3 px) cracks | **High** — P2 maps are large; significant latency + memory | High on recall, watch cost |
| KLD/GWD loss (6.1) | Localization/mAP gain on high-aspect-ratio cracks; better angle stability | Zero inference cost (training only) | Medium-high |
| VFL classification (6.2) | Small-moderate precision gain | Zero inference cost | Medium |
| Coordinate Attention + strip pool (5) | Small-moderate recall/precision gain on elongated cracks | Low | Medium |
| AFPN/BiFPN + DyHead (4) | Moderate mAP gain | Moderate latency | Medium |
| Hybrid backbone (3) | Moderate mAP gain; main accuracy ceiling lift | Moderate-high (varies by variant) | Medium |
| Multi-task seg heads (7) | Indirect precision gain + interpretability + crack-type output | Training cost; small inference cost (heads can be dropped at deploy) | Medium |
| Contrastive term (6.3) | Small targeted precision gain on hardest ballast cases | Training only | Lower (more variance) |

**Honest cost note.** The P2 level and any transformer-heavy backbone are your latency/memory risks; if deployment is edge/real-time (likely for track inspection), report a RepViT + no-P2 variant on the Pareto curve. The cheap interventions (ROI slicing, mask gating, fusion, hard-negative mining) plausibly deliver most of the FP reduction at near-zero inference cost — if you are resource-constrained, do those first and treat the architecture upgrades as the accuracy-ceiling phase of the paper.

---

## Suggested build order (so effort tracks impact)

1. **Phase 1 — root-cause fixes, cheap:** sleeper segmentation → ROI-restricted SAHI → post-hoc mask gating → mask-aware fusion → hard-negative mining loop. Measure FPPI and ballast-gap FP rate after each. This should already be a strong result.
2. **Phase 2 — losses, training-only:** KLD/GWD + VFL + Focal-Tversky + multi-task heads + uncertainty weighting.
3. **Phase 3 — architecture ceiling:** P2 level, hybrid backbone, AFPN/DyHead, Coordinate Attention + strip pooling, FiLM context conditioning, contrastive term.
4. **Phase 4 — paper assembly:** full ablation table (Section 13), Pareto efficiency curve, the FP-localization diagnostic, and the dataset/benchmark contribution.

The narrative that writes itself for the paper: *"SAHI's recall benefit comes at a context cost that manifests as substrate-confusion false positives; we diagnose this, restore context cheaply via region-conditioned slicing and gating, and lift the ceiling with thin-structure-specialized architecture and Gaussian rotated losses, achieving SOTA crack detection with a large reduction in ballast-gap false positives at modest cost."*
