Per-slide change checklist
Title slide (1) + every footer
Change "Project Proposal" → "Final Project" (or "Final Report / Presentation").
Remove the "· PROPOSAL" suffix from all 13 slide footers.
Slide 6 — Methodology
Dashboard line says "Streamlit/Grafana/FastAPI + WebSocket". The shipped UI is a single-file FastAPI + WebSocket + HTML frontend only. Change to "FastAPI + WebSocket + HTML" (drop Streamlit/Grafana — they aren't used).
Slide 7 — End-to-End Architecture
Diagram is still accurate. Optional add: a note that the detector can run a temporal model instead of YOLO (DETECTOR_MODE=temporal + TEMPORAL_WEIGHTS) — ties into the new ablation slides.
Slide 9 — Dataset
Clip counts (691 clips, 8 classes, train/test tree) are still correct — keep.
Footnote "*636/840 frames are unsafe" → "768/1500 test frames are unsafe (~51%)". The class balance story changed materially: the test set is now roughly balanced, not ~75% unsafe.
"trained over 50 epochs, imgsz 224" line is fine.
Slide 10 — Evaluation & Validation
Swap in the corrected metric table above.
Add the honesty caveat (from README "How to read these"): test set is ~51% unsafe, so the trivial "always unsafe" baseline scores only ~0.51 accuracy → F1 (0.803) and recall (0.841) are the honest headline, not accuracy; model errs toward false alarms (FP 194 > FN 122), the desirable bias for a safety monitor; metrics are frame-level (125 clips × 12), so the effective sample size is smaller than 1500.
Streaming column: keep alerts/min (14–15) and throughput (9.8–10.0 msg/s) but tag them demo-specific. Sliding-window 30% and 2-camera validation are still accurate — keep.
Decision needed — the latency bullet currently says the high end-to-end latency "indicates a timestamp-synchronization issue that requires further validation." If that's now fixed, put a real number; if not, keep the caveat but soften "requires further validation" for a final deck.
Slide 8 — Why the Aggregator?
Keep as-is (the traffic-cam analogy is strong and on-message). Optional one-line tie-in to the new finding: the aggregator + per-model threshold calibration is what makes this work in practice (see new Slide B).
New slides to add (the actual new results)
New Slide A — "SOTA Ablation: temporal models vs the per-frame baseline"
Insert after Slide 10. Drop in the 6-row table (values from ablation_results.md/.json; README has the paste-ready version):

Model	Offline AP	F1	8-cls top-1	Smart false-alert	enter-thr (cal)	Alert P / R	Latency ms/frame
YOLOv8 (per-frame)	0.887	0.803	0.641	0.311	0.60	0.732 / 0.812	6.4
head (ResNet18+GRU)	0.862	0.704	0.440	0.557	0.65	0.634 / 0.922	13.6
video (R(2+1)D-18)	0.755	0.695	0.472	0.705	0.65	0.578 / 0.922	26.4
MViTv2-S	0.795	0.707	0.488	0.852	0.65	0.544 / 0.969	48.4
Video Swin-T	0.805	0.702	0.480	0.623	0.70	0.596 / 0.875	39.8
Hiera-B	0.862	0.727	0.536	0.787	0.45	0.571 / 1.000	47.0
Caption: controlled run — same splits / window=16 / 30 epochs; transformers linear-probed (~6K trainable params); streaming thresholds calibrated per model on val to recall ≥ 0.95, reported on test; latency measured identically per row.
New Slide B — "Key finding: offline quality ≠ streaming quality"
The genuinely novel result — give it its own slide. Four points (from README "Results & findings"):

Temporal modeling does not beat the per-frame baseline here — YOLO wins offline AP, F1, and 8-class top-1. These violations are mostly single-frame appearance/state, so 16-frame aggregation adds little.
Offline AP does not predict streaming false-alert rate — head and Hiera tie on offline AP (0.862) but have opposite streaming false-alert rates (0.557 vs 0.787). Picking by offline F1 would choose one of the worst alert streams.
Per-model threshold calibration is the biggest lever — no model is well-calibrated at the default 0.5; chosen enter-thresholds span 0.45–0.70. Hiera looked best pre-calibration only because it was scored at a recall it never reached.
The shipped per-frame YOLO also streams best and is cheapest — lowest false-alert rate (0.311), highest alert precision (0.732), and 6–8× faster than the transformers. → Invest in calibration + the aggregator, not a heavier backbone.
New Slide C (optional) — naive vs smart aggregator
If room: the head-to-head from eval_streaming.json makes the aggregator's value concrete — naive false-alert 0.508 → smart 0.393, precision 0.644 → 0.696 (125 clips). Pairs well with Slide 8.

Slide 13 — Conclusion (reframe proposal → done)
"Deliverables" is future-tense and promises "ablation over window length and unsafe-ratio threshold." What was actually delivered is an ablation over model architecture (5 SOTA temporal models vs YOLO) + per-model streaming threshold calibration. Reconcile this — either reword the deliverable to match what was done, or explicitly note window-length was fixed at 16 (not varied). Don't leave the old promise unfixed.
Add the empirical finding as a stated contribution: "offline AP does not predict streaming alert quality; calibration + aggregator matter more than backbone."
Move deliverables to past tense ("Delivered: open-source pipeline, documented topic schemas, clip-level + streaming evaluation, SOTA ablation over architecture and calibrated thresholds").
Verification (cross-check before presenting)
Every number on Slide 10 ↔ eval_results.json.
Every cell on new Slide A ↔ ablation_results.json rows (offline.*, streaming_smart.false_alert_rate, streaming_cal_enter, streaming_smart.alert_precision/alert_recall, latency_ms_per_frame).
Slide C numbers ↔ eval_streaming.json.
Findings text ↔ README.md "Results & findings" (lines ~339–369) so deck and repo agree.
Confirm one consistent latency source (ablation_results.json is source of truth; README's table is slightly re-rounded — pick one and use it everywhere).
Resolve the two open decisions flagged above: (1) end-to-end latency bullet on Slide 10, (2) the conclusion deliverable wording vs what was actually ablated.