"""Independent arithmetic and pairing checks for the extended analysis."""
from pathlib import Path
import json,sys,hashlib,platform
import numpy as np
import pandas as pd
from scipy.stats import t
import sklearn,scipy
ROOT=Path(__file__).resolve().parents[1];F=ROOT/'outputs/feature_ablation_20261024';E=ROOT/'outputs/extended_analysis_20261024'
raw=pd.read_csv(F/'per_mission.csv');agg=pd.read_csv(F/'metrics.csv');pairs=pd.read_csv(F/'comparisons.csv');primary=pd.read_csv(ROOT/'outputs/paper_primary_qwen25_7b_v3/per_mission.csv')
assert len(raw)==48 and len(agg)==4 and len(pairs)==28
checks=0
for _,a in agg.iterrows():
 g=raw[(raw.config==a.config)&(raw.feature_access==a.feature_access)].set_index('mission').sort_index();assert list(g.index)==list(range(1,13)) and g.fold.nunique()==4
 for key in ['role_acc_h0','role_acc_h10','role_acc_h30','phase_acc','plan_f1','plan_prec','plan_rec']:
  x=g[key].to_numpy();assert np.isclose(a[key],x.mean());assert np.isclose(a[key+'_ci'],t.ppf(.975,11)*x.std(ddof=1)/np.sqrt(12));checks+=1
  p=pairs[(pairs.config==a.config)&(pairs.feature_access==a.feature_access)&(pairs.metric==key)].iloc[0]
  y=primary[primary.config_key==a.config+'_olsr_n20_v15_gm_quad'].set_index('mission').sort_index()[key].to_numpy();d=x-y;b=d[np.random.default_rng(2026).integers(0,12,(10000,12))].mean(1);lo,hi=np.percentile(b,[2.5,97.5]);assert np.allclose([p.mean_diff,p.lo,p.hi],[d.mean(),lo,hi])
# Reconstruct deployment identifiers independently from descriptive labels.
sc={'Default OLSR':'olsr_n20_v15_gm_quad','AODV':'aodv_n20_v15_gm_quad','Group mobility':'olsr_n20_v15_rpgm_quad','Edge monitor':'olsr_n20_v15_gm_edge','10 UAVs':'olsr_n10_v15_gm_quad','30 UAVs':'olsr_n30_v15_gm_quad','5 m/s':'olsr_n20_v5_gm_quad','25 m/s':'olsr_n20_v25_gm_quad'}
pdifs=pd.read_csv(E/'deployment_paired_comparisons.csv');assert len(pdifs)==48
for _,row in pdifs.iterrows():
 suffix=sc[row.scenario];x=primary[primary.config_key=='full_b0.2_'+suffix].set_index('mission').sort_index();y=primary[primary.config_key=='none_b0_'+suffix].set_index('mission').sort_index();assert x.index.equals(y.index);d=(x[row.metric]-y[row.metric]).to_numpy();n=len(d);b=d[np.random.default_rng(2026).integers(0,n,(10000,n))].mean(1);lo,hi=np.percentile(b,[2.5,97.5]);assert np.allclose([row.mean_diff,row.lo,row.hi],[d.mean(),lo,hi]);assert row.negative_missions==(d<0).sum()
report={'feature_means_and_t_intervals_verified':checks,'feature_paired_bootstrap_intervals_verified':len(pairs),'deployment_paired_bootstrap_intervals_verified':len(pdifs),'new_network_simulations':0,'environment':{'python':sys.version.split()[0],'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'platform':platform.platform()},'input_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [F/'per_mission.csv',F/'metrics.csv',F/'comparisons.csv',E/'deployment_paired_comparisons.csv']},'scope':'Arithmetic, interval and pairing consistency; not an independent classifier implementation.'}
(ROOT/'validation/extended_results_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
