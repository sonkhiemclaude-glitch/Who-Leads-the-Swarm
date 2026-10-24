from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from scipy.stats import t
p=Path('outputs/paper_primary_qwen25_7b_v3');a=pd.read_csv(p/'metrics.csv');b=pd.read_csv(p/'per_mission.csv');c=pd.read_csv(p/'comparisons.csv')
assert len(a)==28 and len(b)==288 and len(c)==5
checks=0
for _,row in a.iterrows():
 g=b[b.config_key==row.config_key];assert len(g)==int(row.missions)
 for key in a.columns:
  if key.endswith('_ci') or key not in b.columns or key in ['mission','fold','config_key','defense','budget','routing','n','speed','mobility','monitors','missions','provenance_label']:continue
  if not pd.api.types.is_numeric_dtype(b[key]):continue
  vals=g[key].dropna().to_numpy()
  if not len(vals):assert pd.isna(row[key]);continue
  assert np.isclose(vals.mean(),row[key],atol=1e-10),key
  if key+'_ci' in a:
   half=t.ppf(.975,len(vals)-1)*vals.std(ddof=1)/np.sqrt(len(vals))
   assert np.isclose(half,row[key+'_ci'],atol=1e-10),key
  checks+=1
for _,row in c.iterrows():
 x=b[b.config_key==row.a].set_index('mission')[row.metric];y=b[b.config_key==row.b].set_index('mission')[row.metric];d=(x-y).dropna();assert len(d)==row.n and np.isclose(d.mean(),row.mean_diff)
 boots=d.to_numpy()[np.random.default_rng(2026).integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
 lo,hi=np.percentile(boots,[2.5,97.5]);assert np.isclose(lo,row.lo) and np.isclose(hi,row.hi)
q=json.loads(Path('logs/collection-trace-quality.json').read_text());assert len(q)==24 and all(x['validated'] and x['finish_reasons']=={'stop':x['model_calls']} for x in q)
prov=json.loads((p/'provenance.json').read_text());assert prov['synthetic'] is False and prov['n_runs']==288
report={'aggregate_means_and_t_intervals_verified':checks,'paired_mean_differences_and_bootstrap_intervals_verified':5,'validated_traces':24,'model_calls':sum(x['model_calls'] for x in q),'logical_messages':sum(x['messages'] for x in q),'n_runs':288,'source_sha256':prov['source_sha256'],'config_hash':prov['config_hash'],'artifact_sha256':{x:hashlib.sha256((p/x).read_bytes()).hexdigest() for x in ['metrics.csv','per_mission.csv','comparisons.csv','provenance.json']},'scope':'Recomputed all aggregate means/intervals and paired means from saved per-mission results; not an independent simulator or attacker reimplementation.'}
Path('validation/results_verification_20261022.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
