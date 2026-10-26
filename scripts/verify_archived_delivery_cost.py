"""Independent saved-output arithmetic and vectorized raw-message spot checks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/archive_reanalysis_20261026';BASE=ROOT/'outputs/paper_primary_qwen25_7b_v3'
def check_summary(raw,summary,by,metrics):
 checks=0
 for _,row in summary.iterrows():
  mask=np.ones(len(raw),bool)
  for key in by:mask&=raw[key].to_numpy()==row[key]
  g=raw[mask];assert g.mission.nunique()==row.missions
  for key in metrics:
   x=g[key].dropna().to_numpy();ci=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x));assert np.isclose(row[key],x.mean()) and np.isclose(row[key+'_ci'],ci);checks+=1
 return checks
U=pd.read_csv(D/'original_traffic_per_mission.csv');H=pd.read_csv(D/'handover_payloads_per_mission.csv');T=pd.read_csv(D/'handover_payloads_per_pair.csv');N=pd.read_csv(D/'per_uav_costs.csv');C=pd.read_csv(D/'original_traffic_paired_comparisons.csv')
a=check_summary(U,pd.read_csv(D/'original_traffic_summary.csv'),['config','routing','kind'],['pdr','lat_ms_median','lat_ms_p95'])
b=check_summary(H,pd.read_csv(D/'handover_payload_summary.csv'),['config','routing'],['outgoing_delivered','return_delivered','both_delivered','outgoing_before_switch'])
assert len(T)==918 and H.scheduled_pairs.sum()==918 and H.mapping_transitions.sum()==918
checks=0
for _,row in C.iterrows():
 x=U[(U.config==row.config)&(U.routing==row.routing)&(U.kind==row.kind)].set_index('mission').sort_index()[row.metric];y=U[(U.config=='none_b0')&(U.routing==row.routing)&(U.kind==row.kind)].set_index('mission').sort_index()[row.metric];d=(x-y).dropna().to_numpy();assert len(d)==row.n;boot=d[np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))].mean(1);lo,hi=np.percentile(boot,[2.5,97.5]);assert np.allclose([row.mean_diff,row.lo,row.hi],[d.mean(),lo,hi]);checks+=1
costchecks=0
for _,row in pd.read_csv(D/'per_uav_cost_summary.csv').iterrows():
 g=N[(N.config==row.config)&(N.routing==row.routing)];assert len(g)==240
 for k in ['increment_radio_j','ledger_j','utilization']:
  x=g[k].dropna().to_numpy()
  if not len(x):continue
  for q in [.1,.5,.9]:assert np.isclose(row[k+f'_q{int(q*100)}'],np.quantile(x,q));costchecks+=1
probes=0
for config,routing in [('none_b0','olsr'),('pad_b0','olsr'),('rot_b0','olsr'),('full_b0.2','olsr'),('const_b0.2','olsr'),('full_b0.2','aodv')]:
 for mission in [1,6,12]:
  with np.load(BASE/'runs'/f'{config}_{routing}_n20_v15_gm_quad_m{mission}.npz',allow_pickle=False) as z:
   real=z['msg_real'].astype(bool)&~np.char.startswith(z['msg_kind'],'handover_');gid=z['msg_gid'][real];order=np.argsort(gid,kind='stable');gid=gid[order];rx=z['rx_time'][real][order];send=z['msg_t'][real][order];kind=z['msg_kind'][real][order];ix=np.r_[0,np.flatnonzero(np.diff(gid))+1];ok=np.minimum.reduceat(np.isfinite(rx).astype(int),ix).astype(bool);last=np.maximum.reduceat(np.where(np.isfinite(rx),rx,-np.inf),ix);first=np.minimum.reduceat(send,ix);lat=(last-first)*1000;ks=kind[ix]
   for pool in ['all_original','plan','ack','report']:
    mask=np.ones(len(ix),bool) if pool=='all_original' else ks==pool;row=U[(U.config==config)&(U.routing==routing)&(U.mission==mission)&(U.kind==pool)].iloc[0];assert mask.sum()==row.n_original and ok[mask].sum()==row.n_delivered;assert np.isclose(np.quantile(lat[mask&ok],.95),row.lat_ms_p95)
   q=N[(N.config==config)&(N.routing==routing)&(N.mission==mission)].sort_values('uav');assert np.allclose(q.total_radio_j,z['energy_j'])
  probes+=1
proof={'original_summary_means_and_t_intervals':a,'handover_summary_means_and_t_intervals':b,'paired_bootstrap_intervals':checks,'descriptive_quantiles':costchecks,'independent_vectorized_raw_run_probes':probes,'total_payload_pairs':len(T),'new_simulations':0,'new_model_fits':0,'scope':'Saved-record consistency, not independent simulator or migration-protocol validation.'}
(ROOT/'validation/archive_reanalysis_verification_20261026.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
