"""Exploratory feature-access controls; saved primary frames remain unchanged."""
from pathlib import Path
from dataclasses import replace
import json,hashlib,time
import numpy as np
import pandas as pd
from scipy.stats import t
import swarmleak.attack.evaluate as ev
from swarmleak.data.schemas import RunRecord
from swarmleak.experiments.provenance import source_digest
ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'outputs/feature_ablation_20261024'
MASKS={'size_blind':{'role':[0,2,4,5,12,13],'window':[0,2,9,10,11]},'counts_timing':{'role':[0,2,12,13],'window':[0,2,9]}}
def main():
 DEST.mkdir(parents=True,exist_ok=True)
 plan={'design':'post-primary exploratory feature-access controls','configs':['none_b0','full_b0.2'],'missions':list(range(1,13)),'masks':MASKS,'folds':4,'seed':2026,'rf_trees':200,'hyperparameter_search':False,'network_replay':False,'notes':'Mask columns before training/testing; original frames and labels retained. These are group controls, not individual-feature importance.'}
 (DEST/'plan.json').write_text(json.dumps(plan,indent=2))
 assert source_digest()=='a8537b8386e853f01030a6ed68945e7e000464b85cb76a76bf15e297a4b5bd55'
 orig=(ev.role_dataset,ev.phase_dataset,ev.plan_dataset);rows=[];start=time.time()
 for config in plan['configs']:
  runs=[RunRecord.load(ROOT/'outputs/paper_primary_qwen25_7b_v3/runs'/f'{config}_olsr_n20_v15_gm_quad_m{m}.npz') for m in plan['missions']]
  cache={id(r):(orig[0](r),orig[1](r),orig[2](r)) for r in runs};print('Cached',config,flush=True)
  for name,mask in MASKS.items():
   rc=mask['role']+[i+16 for i in mask['role']];wc=mask['window'];pc=wc+[i+14 for i in wc]
   def role(r,*a,**kw):
    d=cache[id(r)][0];x=np.zeros_like(d.X);x[:,rc]=d.X[:,rc];return replace(d,X=x)
   def phase(r,*a,**kw):
    x,y=cache[id(r)][1];return x[:,wc],y
   def change(r,*a,**kw):
    x,y,z=cache[id(r)][2];return x[:,pc],y,z
   ev.role_dataset,ev.phase_dataset,ev.plan_dataset=role,phase,change
   print('Evaluating',config,name,flush=True)
   for row in ev.evaluate_config(runs,ev.AttackConfig(n_jobs=2),2026):
    row.pop('role_acc_heur',None);row.update(config=config,feature_access=name);rows.append(row)
   pd.DataFrame(rows).to_csv(DEST/'per_mission.csv',index=False)
 ev.role_dataset,ev.phase_dataset,ev.plan_dataset=orig
 df=pd.DataFrame(rows);agg=[];contrasts=[];primary=pd.read_csv(ROOT/'outputs/paper_primary_qwen25_7b_v3/per_mission.csv')
 for (config,access),g in df.groupby(['config','feature_access']):
  out={'config':config,'feature_access':access,'missions':len(g)}
  for metric in ['role_acc_h0','role_acc_h10','role_acc_h30','phase_acc','plan_f1','plan_prec','plan_rec']:
   x=g.set_index('mission')[metric].sort_index().to_numpy();out[metric]=x.mean();out[metric+'_ci']=t.ppf(.975,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
   ref=primary[primary.config_key==config+'_olsr_n20_v15_gm_quad'].set_index('mission')[metric].sort_index().to_numpy();d=x-ref
   boots=d[np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))].mean(1);lo,hi=np.percentile(boots,[2.5,97.5]);contrasts.append(dict(config=config,feature_access=access,metric=metric,n=len(d),mean_diff=d.mean(),lo=lo,hi=hi))
  agg.append(out)
 pd.DataFrame(agg).to_csv(DEST/'metrics.csv',index=False);pd.DataFrame(contrasts).to_csv(DEST/'comparisons.csv',index=False)
 proof={'source_sha256':source_digest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'n_feature_configurations':4,'mission_scores':len(df),'elapsed_s':time.time()-start,'new_network_simulations':0,'completed':True}
 (DEST/'provenance.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
if __name__=='__main__':main()
