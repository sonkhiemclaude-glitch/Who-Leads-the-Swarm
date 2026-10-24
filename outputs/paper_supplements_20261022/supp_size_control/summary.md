# Results summary

**Provenance: simulation output (backend=ns3, traces=llm)**

| config | missions | role_acc_h0 | role_acc_h10 | role_acc_h30 | role_acc_heur | phase_acc | phase_majority | plan_f1 | pdr | lat_ms_median | dE_j_mean | cover_airtime_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full_b0.2_olsr_n20_v15_gm_quad | 12 | 0.883 ± 0.020 | 0.704 ± 0.046 | 0.443 ± 0.072 | 0.107 ± 0.026 | 0.645 ± 0.055 | 0.487 ± 0.043 | 0.668 ± 0.072 | 0.781 ± 0.020 | 0.723 ± 0.000 | 0.297 ± 0.057 | 0.028 ± 0.002 |
| none_b0_olsr_n20_v15_gm_quad | 12 | 0.991 ± 0.009 | 0.991 ± 0.009 | 0.991 ± 0.009 | 0.156 ± 0.033 | 0.693 ± 0.047 | 0.487 ± 0.043 | 0.711 ± 0.052 | 0.781 ± 0.034 | 0.723 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

## Proposition 1 diagnostic (rotating configurations)

alpha0 is approximated by role_acc_h0 of the same configuration; p_E is measured on the evaluation windows; p_max = 1/3 (top-3 successor). This is a diagnostic, not a test of the proposition: the paper's alpha0 is the accuracy of the horizon-specific estimator for the current orchestrator.

| config | horizon | measured role acc | bound | p_E | Pr[>=2 rotations] |
|---|---|---|---|---|---|
| full_b0.2_olsr_n20_v15_gm_quad | 10 s | 0.704 | 0.739 | 0.189 | 0.001 |
| full_b0.2_olsr_n20_v15_gm_quad | 30 s | 0.443 | 0.494 | 0.507 | 0.063 |
