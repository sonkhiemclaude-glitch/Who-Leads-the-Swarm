"""Additional delivery/cost summaries from existing, hash-verified raw ns-3 records."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
from scipy.stats import t
from swarmleak.protocol.energy import RadioModel
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/paper_primary_qwen25_7b_v3';OUT=ROOT/'outputs/archive_reanalysis_20261026';OUT.mkdir(exist_ok=True)
def logical(a):
 rows=[]
 for gid in np.unique(a['msg_gid'][a['msg_real'].astype(bool)]):
  ix=(a['msg_gid']==gid)&a['msg_real'].astype(bool);kind=np.unique(a['msg_kind'][ix]);assert len(kind)==1
  ok=bool(np.isfinite(a['rx_time'][ix]).all());s=float(a['msg_t'][ix].min());r=float(a['rx_time'][ix].max()) if ok else np.nan
  rows.append(dict(gid=int(gid),kind=str(kind[0]),send=s,delivered=ok,arrival=r,lat_ms=(r-s)*1000 if ok else np.nan))
 return pd.DataFrame(rows).sort_values('gid').reset_index(drop=True)
def summarize(df,by,metrics):
 rows=[]
 for keys,g in df.groupby(by):
  keys=keys if isinstance(keys,tuple) else (keys,);r=dict(zip(by,keys));r['missions']=g.mission.nunique()
  for k in metrics:
   x=g[k].dropna().to_numpy();r[k]=x.mean();r[k+'_ci']=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else np.nan
  rows.append(r)
 return pd.DataFrame(rows)
def main():
 manifest=json.loads((ROOT/'validation/primary_run_input_hashes.json').read_text());expected={x['path']:x['sha256'] for x in manifest['files']};verified={};utility=[];nodes=[];transfers=[];hm=[];base={};base_trace={};radio=RadioModel();old=pd.read_csv(BASE/'per_mission.csv')
 cases=[(k,'olsr') for k in ['none_b0','pad_b0','rot_b0','full_b0.05','full_b0.2','full_b1','const_b0.2','const_b1','cover_uniform_b0.2','cover_targeted_b0.2','full_anyhop_b0.2']]+[(k,'aodv') for k in ['none_b0','full_b0.2','full_anyhop_b0.2']]
 for config,routing in cases:
  for mission in range(1,13):
   key=f'{config}_{routing}_n20_v15_gm_quad';p=BASE/'runs'/f'{key}_m{mission}.npz';rel=str(p.relative_to(ROOT));sha=hashlib.sha256(p.read_bytes()).hexdigest();assert sha==expected[rel];verified[rel]=sha
   with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   meta=json.loads(str(a['meta']));assert meta['synthetic'] is False
   l=logical(a);o=l[~l.kind.str.startswith('handover_')].reset_index(drop=True)
   if config=='none_b0':base[routing,mission]=(o,a['energy_j']);base_trace[routing,mission]=meta['trace_sha256']
   assert meta['trace_sha256']==base_trace[routing,mission]
   ref,e0=base[routing,mission];assert list(o.kind)==list(ref.kind) and np.allclose(o.send,ref.send,atol=1e-10)
   for kind in ['all_original','plan','ack','report']:
    g=o if kind=='all_original' else o[o.kind==kind];lat=g.loc[g.delivered,'lat_ms']
    if len(g):utility.append(dict(config=config,routing=routing,mission=mission,kind=kind,n_original=len(g),n_delivered=int(g.delivered.sum()),pdr=g.delivered.mean(),lat_ms_median=lat.median(),lat_ms_p95=lat.quantile(.95)))
   spent=np.zeros(len(a['energy_j']));mask=~a['msg_real'].astype(bool)
   for src,dst,size in zip(a['msg_src'][mask],a['msg_dst'][mask],a['msg_size'][mask]):
    size=int(size)-radio.security_overhead_b;spent[int(src)]+=radio.tx_cost(size);spent[int(dst)]+=radio.rx_cost(size)
   assert np.isclose(spent.max(),meta['cover_est_energy_max_j'],atol=1e-10);budget=float(meta['budget']);assert spent.max()<=budget+1e-10
   de=a['energy_j']-e0;assert np.isclose(de.mean(),old[(old.config_key==key)&(old.mission==mission)].iloc[0].dE_j_mean)
   for i,(total,delta,ledger) in enumerate(zip(a['energy_j'],de,spent)):nodes.append(dict(config=config,routing=routing,mission=mission,uav=i,total_radio_j=total,increment_radio_j=delta,ledger_j=ledger,nominal_budget_j=budget,utilization=ledger/budget if budget else np.nan))
   pairs=[]
   for _,h in l[l.kind=='handover_orch'].iterrows():
    w=l[l.gid==h.gid+1];assert len(w)==1 and w.iloc[0].kind=='handover_worker';w=w.iloc[0]
    row=dict(config=config,routing=routing,mission=mission,gid=int(h.gid),send=h.send,outgoing_delivered=bool(h.delivered),return_delivered=bool(w.delivered),both_delivered=bool(h.delivered and w.delivered),outgoing_before_switch=bool(h.delivered and h.arrival<=h.send+.05+1e-10));pairs.append(row);transfers.append(row)
   assert len(pairs)==len(l[l.kind=='handover_worker'])
   if pairs:
    q=pd.DataFrame(pairs);hm.append(dict(config=config,routing=routing,mission=mission,scheduled_pairs=len(pairs),mapping_transitions=meta['n_rotations'],**{k:q[k].mean() for k in ['outgoing_delivered','return_delivered','both_delivered','outgoing_before_switch']}))
  print('Checked',config,routing,flush=True)
 U=pd.DataFrame(utility);N=pd.DataFrame(nodes);T=pd.DataFrame(transfers);H=pd.DataFrame(hm)
 for name,df in [('original_traffic_per_mission',U),('per_uav_costs',N),('handover_payloads_per_pair',T),('handover_payloads_per_mission',H)]:df.to_csv(OUT/(name+'.csv'),index=False)
 summarize(U,['config','routing','kind'],['pdr','lat_ms_median','lat_ms_p95']).to_csv(OUT/'original_traffic_summary.csv',index=False)
 summarize(H,['config','routing'],['outgoing_delivered','return_delivered','both_delivered','outgoing_before_switch']).to_csv(OUT/'handover_payload_summary.csv',index=False)
 costs=[]
 for (c,r),g in N.groupby(['config','routing']):
  row=dict(config=c,routing=r,missions=g.mission.nunique(),uav_mission_values=len(g))
  for k in ['increment_radio_j','ledger_j','utilization']:
   x=g[k].dropna()
   if len(x):
    row[k+'_mean']=x.mean();row[k+'_max']=x.max()
    for q in [.1,.5,.9]:row[k+f'_q{int(q*100)}']=x.quantile(q)
  costs.append(row)
 pd.DataFrame(costs).to_csv(OUT/'per_uav_cost_summary.csv',index=False)
 contrasts=[]
 for (c,r,k),g in U[U.config!='none_b0'].groupby(['config','routing','kind']):
  a=g.set_index('mission').sort_index();b=U[(U.config=='none_b0')&(U.routing==r)&(U.kind==k)].set_index('mission').sort_index();assert a.index.equals(b.index)
  for metric in ['pdr','lat_ms_p95']:
   d=(a[metric]-b[metric]).dropna().to_numpy();n=len(d);boot=d[np.random.default_rng(2026).integers(0,n,(10000,n))].mean(1);lo,hi=np.percentile(boot,[2.5,97.5]);contrasts.append(dict(config=c,routing=r,kind=k,metric=metric,n=n,mean_diff=d.mean(),lo=lo,hi=hi))
 pd.DataFrame(contrasts).to_csv(OUT/'original_traffic_paired_comparisons.csv',index=False)
 proof={'raw_runs_hash_verified':len(verified),'raw_input_hashes':verified,'new_llm_calls':0,'new_ns3_simulations':0,'new_attacker_fits':0,'original_sequences_matched_to_baseline':True,'ledger_reconstruction_matches_original_metadata':True,'handover_metric':'payload reception only; not successful migration or a delivery-aware protocol','per_node_quantiles':'descriptive pooled UAV/mission values; not independent samples','exploratory':True,'multiplicity_adjusted':False,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'provenance.json').write_text(json.dumps(proof,indent=2));print('Completed',len(verified),'archived runs;',len(T),'scheduled payload pairs',flush=True)
if __name__=='__main__':main()
