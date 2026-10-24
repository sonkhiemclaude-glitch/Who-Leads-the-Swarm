# Implementation notes

How the paper was translated into code. Section numbers refer to the manuscript PDF.

## Causality of the protocol (§5.1)

`protocol/mapping.py` walks the logical trace in time order. At each plan change it commits, as the paper says, the successor, the handover time and the two decoy UAVs "inside the plan messages"; no extra signalling message is generated for this. Decoys start after their own plan message is *sent* plus an assumed 0.05 s delivery (Algorithm 1; audit M14). Spread decoys, mirrors and constant-rate cover use keyed per-node Poisson or periodic schedules. Cover sizes are drawn from what the UAV has already seen (`SizeMemory`, only sizes with time < t). Positions come from the trajectory; the paper assumes they are shared and does not simulate that traffic.

## Admission (Algorithm 1)

`protocol/cover.py::schedule_cover` sorts all candidates by time and admits each message only if both endpoints stay within the share cap and the total budget, the sender's token bucket has a token, and the directed link has been idle ≥ 0.5 s. The ack reply of a decoy is a cover message from the neighbour, so it is checked the same way against the neighbour's bucket and the reverse link (audit M10). Token refill never goes backwards in time (M11). `CoverLedger` holds estimated energy per node and stage; a test recomputes it from the emitted messages.

## Energy estimate (§5.3)

`RadioModel.tx_cost/rx_cost` = V·(I − I_idle)·A(ℓ + 29). The 29-B security overhead is included because the simulator transmits it (M6). Measured energy comes from ns-3 and is reported separately as a paired difference.

## Adversary (§4.3-4.4)

- Frames are de-duplicated across monitors with key (tx, seq, size, retry, t to 0.1 ms) (M20).
- Role windows are the same for all horizons (ending ≤ T − Δ_max) so horizons compare the same windows (M1).
- One random forest per horizon; the estimate is the argmax score with random tie-breaking (M2).
- Plan detection: local maxima ≥ 0.5, merged within 3 s, plateau keeps its first point; matching in temporal order to the nearest unmatched truth within δ = 5 s; undefined precision/recall/F1 are NaN (M4).
- Folds: contiguous mission groups, at most n_runs/2 folds.

## Proposition 1 (§5.5)

`theory/horizon_bound.py` implements the bound and the exact value; tests check the exact value and the bound by Monte Carlo. `experiments/tables.py::write_markdown_summary` prints a diagnostic comparing measured role accuracy at Δ with the bound, using role_acc_h0 as a stand-in for α0 (the paper's α0 is the accuracy of the horizon-specific estimator for the current orchestrator, which the evaluation does not isolate); it is labelled a diagnostic.

## Backends

- `ns3`: port of the pilot's ns-3 code; each run in its own process (`ProcessPoolExecutor`, spawn, one task per child).
- `toy`: geometric packet replayer for tests only (unit-disk links, shortest paths, per-hop loss with retries, HELLO broadcasts, AODV-like flood). All its outputs are labelled synthetic.

## Plot colours

The three series colours (#2a78d6, #eb6834, #1baf7a) passed a colour-vision-deficiency separation check (worst adjacent ΔE 9.2, deuteranopia); single-point baselines are grey with direct labels, so identity never relies on colour alone.
