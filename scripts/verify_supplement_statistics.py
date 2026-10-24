"""Independently recompute supplemental CSV summaries and percentile intervals."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path('outputs/paper_supplements_20261022')
def check(raw,summary,key):
 df=pd.read_csv(ROOT/raw);agg=pd.read_csv(ROOT/summary);n=0
 for _,row in agg.iterrows():
  group=df[df[key]==row[key]]
  assert len(group)==row.missions
  for metric in agg.columns:
   if not metric.endswith('_ci') or metric[:-3] not in group:continue
   name=metric[:-3];x=group[name].dropna().to_numpy()
   if not len(x):assert pd.isna(row[name]) and pd.isna(row[metric]);continue
   assert np.isclose(row[name],x.mean(),atol=1e-10)
   ci=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
   assert np.isclose(row[metric],ci,atol=1e-10);n+=1
 return n
if __name__=='__main__':
 n=check('energy-confirmation-per-mission.csv','energy-confirmation-metrics.csv','config')
 n+=check('history-hgb-per-mission.csv','history-hgb-metrics.csv','config')
 n+=check('supp_size_control/per_mission.csv','supp_size_control/metrics.csv','config_key')
 primary=pd.read_csv(ROOT.parent/'paper_primary_qwen25_7b_v3/per_mission.csv')
 def bootstrap(x,y,row):
  d=x.sort_index().to_numpy()-y.sort_index().to_numpy();assert len(d)==int(row.n)
  rng=np.random.default_rng(2026);values=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
  lo,hi=np.percentile(values,[2.5,97.5])
  assert np.isclose(d.mean(),row.mean_diff) and np.isclose(lo,row.lo) and np.isclose(hi,row.hi)
 paired=0
 for kind in ['energy-confirmation','history-hgb','size-control']:
  comparisons=pd.read_csv(ROOT/(kind+'-comparisons.csv'))
  raw=pd.read_csv(ROOT/('supp_size_control/per_mission.csv' if kind=='size-control' else kind+'-per-mission.csv'))
  for _,row in comparisons.iterrows():
   if kind=='energy-confirmation':
    x=raw[raw.config=='full_b1'].set_index('mission')[row.metric];y=raw[raw.config=='matched_const'].set_index('mission')[row.metric]
   else:
    col='config_key' if kind=='size-control' else 'config'
    x=raw[raw[col]==row.config].set_index('mission')[row.metric]
    key=row.config if kind=='size-control' else row.config+'_olsr_n20_v15_gm_quad'
    y=primary[primary.config_key==key].set_index('mission')[row.metric]
   bootstrap(x,y,row);paired+=1
 history=pd.read_csv(ROOT/'history-hgb-per-mission.csv')
 x=history[history.config=='full_b0.2'].set_index('mission').role_acc_h30.sort_index()
 y=history[history.config=='rot_b0'].set_index('mission').role_acc_h30.sort_index()
 d=x.to_numpy()-y.to_numpy();values=d[np.random.default_rng(2026).integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
 lo,hi=np.percentile(values,[2.5,97.5]);contrast={'mean_diff':float(d.mean()),'lo':float(lo),'hi':float(hi),'n':len(d),'exploratory':True}
 report={'extra_history_full_minus_rotation_role_h30':contrast,'supplemental_means_and_t_intervals_recomputed':n,'paired_bootstrap_intervals_recomputed':paired,'scope':'CSV consistency and artifact presence; simulator and attacker provenance checked separately on host'}
 for name in ['analysis.complete.json','energy-match-quality.json','energy-selection.json']:
  report[name]=json.loads((ROOT/name).read_text())
 Path('validation/supplement_verification_20261022.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
