## Per-slide changes

### Title slide (1) + every footer
- "Project Proposal" → **"Final Project"** (or "Final Report / Presentation").
- Remove the "· PROPOSAL" suffix from all 13 slide footers.

### Slide 6 — Methodology
- Dashboard line says "Streamlit/Grafana/FastAPI + WebSocket". The shipped UI is a single-file
  **FastAPI + WebSocket + HTML** frontend only — drop Streamlit/Grafana (not used).

### Slide 7 — End-to-End Architecture
- Diagram is still accurate — keep.
- Optional: note the detector can run a **temporal model** instead of YOLO
  (`DETECTOR_MODE=temporal` + `TEMPORAL_WEIGHTS`), tying into the new ablation slides.

### Slide 9 — Dataset
- Clip counts (691 clips, 8 classes, train/test tree) are still correct — keep.
- Footnote "*636/840 frames are unsafe" → **"768/1500 test frames are unsafe (~51%)"**. The class
  balance story changed materially: the test set is now **roughly balanced**, not ~75% unsafe.

### Slide 10 — Evaluation & Validation
- Swap in the corrected metric table above.
- Add the honesty caveat (from README "How to read these"): test set is ~51% unsafe, so the
  trivial "always unsafe" baseline scores only ~0.51 accuracy → **F1 (0.803) and recall (0.841)
  are the honest headline, not accuracy**; the model errs toward false alarms (FP 194 > FN 122),
  the desirable bias for a safety monitor; metrics are frame-level (125 clips × 12), so the
  effective sample size is smaller than 1500.
- Streaming column: keep alerts/min (~14–15) and throughput (~9.8–10.0 msg/s) but tag them
  **demo-specific**. Sliding-window 30% and 2-camera validation are still accurate — keep.
- **Decision needed — the latency bullet** currently says the high end-to-end latency "indicates
  a timestamp-synchronization issue that requires further validation." If that's now fixed, put a
  real number; if not, keep the caveat but soften "requires further validation" for a final deck.

### Slide 8 — Why the Aggregator?
- Keep as-is (the traffic-cam analogy is strong and on-message). Optional one-line tie-in: the
  aggregator + per-model threshold calibration is what makes this work in practice (see new
  Slide B).

---

## New slides to add

### New Slide B — "Key finding: offline quality ≠ streaming quality"
The genuinely novel result — give it its own slide:

1. **Temporal modeling does not beat the per-frame baseline here** — YOLO wins offline AP, F1,
   and 8-class top-1. These violations are mostly single-frame appearance/state, so 16-frame
   aggregation adds little.
2. **Offline AP does not predict streaming false-alert rate** — `head` and `Hiera` tie on offline
   AP (0.862) but have opposite streaming false-alert rates (0.557 vs 0.787). Picking by offline
   F1 would choose one of the *worst* alert streams.
3. **Per-model threshold calibration is the biggest lever** — no model is well-calibrated at the
   default 0.5; chosen enter-thresholds span 0.45–0.70. Hiera *looked* best pre-calibration only
   because it was scored at a recall it never reached.
4. **The shipped per-frame YOLO also streams best and is cheapest** — lowest false-alert rate
   (0.311), highest alert precision (0.732), 6–8× faster than the transformers. → Invest in
   calibration + the aggregator, **not** a heavier backbone.

### New Slide C (optional) — naive vs smart aggregator
If room: head-to-head from `eval_streaming.json` makes the aggregator's value concrete — naive
false-alert 0.508 → smart **0.393**, precision 0.644 → **0.696** (125 clips). Pairs well with
Slide 8.

---

## Slide 13 — Conclusion (reframe proposal → done)
- "Deliverables" promises "ablation over **window length and unsafe-ratio threshold**." What was
  actually delivered is an ablation over **model architecture (5 SOTA temporal models vs YOLO) +
  per-model streaming threshold calibration**. **Reconcile this** — either reword the deliverable
  to match what was done, or explicitly note window length was fixed at 16 (not varied).
- Add the empirical finding as a stated contribution: "offline AP does not predict streaming
  alert quality; calibration + aggregator matter more than backbone."
- Move deliverables to past tense ("Delivered: open-source pipeline, documented topic schemas,
  clip-level + streaming evaluation, SOTA ablation over architecture and calibrated thresholds").

---

## Verification (cross-check before presenting)
- Every number on Slide 10 ↔ `eval_results.json`.
- Every cell on new Slide A ↔ `ablation_results.json` rows (`offline.*`,
  `streaming_smart.false_alert_rate`, `streaming_cal_enter`, `streaming_smart.alert_precision` /
  `alert_recall`, `latency_ms_per_frame`).
- Slide C numbers ↔ `eval_streaming.json`.
- Findings text ↔ README.md "Results & findings" so deck and repo agree.
- Use one consistent latency source (`ablation_results.json` is source of truth; README's table
  is slightly re-rounded — pick one and use it everywhere).
- Resolve the two open decisions: (1) Slide 10 end-to-end latency bullet, (2) Conclusion
  deliverable wording vs what was actually ablated.
