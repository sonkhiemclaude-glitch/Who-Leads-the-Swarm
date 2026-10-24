"""Mission-disjoint supplemental attack evaluation; no search or in-sample scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import swarmleak.attack.evaluate as ev
from swarmleak.attack.evaluate import AttackConfig
from swarmleak.data.schemas import RunRecord
from swarmleak.metrics.confidence_intervals import paired_bootstrap,t_interval
from swarmleak.metrics.network import incremental_energy,network_metrics
ROOT=Path('outputs/paper_supplements_20261022');PRIMARY=Path('outputs/paper_primary_qwen25_7b_v3');SC='olsr_n20_v15_gm_quad'
def runs(folder,key,missions):return [RunRecord.load(folder/'runs'/f'{key}_{SC}_m{m}.npz') for m in missions]
def aggregate(rows):
 df=pd.DataFrame(rows);result=[]
 for config,g in df.groupby('config'):
  row={'config':config,'missions':len(g)}
  for k in g.select_dtypes(include='number').columns:
   if k in ['mission','fold']:continue
   mean,half=t_interval(g[k]);row[k]=mean;row[k+'_ci']=half
  result.append(row)
 return pd.DataFrame(result)
def lag(x,stride=1):
 out=[x]
 for offset in [stride,2*stride]:
  old=np.zeros_like(x)
  if len(x)>offset:old[offset:]=x[:-offset]
  out.append(old)
 return np.hstack(out)
def main():
 from swarmleak.experiments.config import ExperimentConfig
 from swarmleak.experiments.integrity import validate_run_files,expected_runs,validate_run
 from swarmleak.experiments.runner import load_trace
 from swarmleak.utils.io import stable_hash
 for name in ['mac_studio_structured_v3','supp_energy_calibration','supp_energy_confirmation','supp_size_control']:
  config=ExperimentConfig.load('configs/'+name+'.yaml');files=validate_run_files(config);expected=expected_runs(config)
  for file in files:
   entry,mission=expected[file.stem];record=RunRecord.load(file)
   validate_run(record,config,entry,mission,stable_hash(load_trace(config,entry.scenario,mission).to_dict()))
   assert not record.meta.get('synthetic') and len(record.frames)>0
 primary=pd.read_csv(PRIMARY/'per_mission.csv')
 size=pd.read_csv(ROOT/'supp_size_control/per_mission.csv');size_comparisons=[]
 for key in ['none_b0','full_b0.2']:
  key=key+'_'+SC
  x=size[size.config_key==key].set_index('mission');y=primary[primary.config_key==key].set_index('mission')
  for metric in ['role_acc_h0','role_acc_h30','phase_acc','plan_f1']:
   size_comparisons.append({'config':key,'metric':metric,**paired_bootstrap(x[metric].sort_index(),y[metric].sort_index(),10000,seed=2026)})
 pd.DataFrame(size_comparisons).to_csv(ROOT/'size-control-comparisons.csv',index=False)
 selection=json.loads((ROOT/'energy-selection.json').read_text());b=selection['frozen_const_budget'];missions=list(range(5,13));energy_rows=[]
 for name,folder,key in [('full_b1',PRIMARY,'full_b1'),('matched_const',ROOT/'supp_energy_confirmation',f'const_b{b:g}')]:
  records=runs(folder,key,missions)
  for row,r,m in zip(ev.evaluate_config(records,AttackConfig(),2026),records,missions):
   row.update(network_metrics(r));row['dE_j_mean']=incremental_energy(r,runs(PRIMARY,'none_b0',[m])[0]);row['config']=name;energy_rows.append(row)
 energy=pd.DataFrame(energy_rows);energy.to_csv(ROOT/'energy-confirmation-per-mission.csv',index=False);agg=aggregate(energy_rows);agg.to_csv(ROOT/'energy-confirmation-metrics.csv',index=False)
 comparison=[]
 for metric in ['role_acc_h0','role_acc_h30','phase_acc','plan_f1','dE_j_mean','pdr']:
  a=energy[energy.config=='full_b1'].set_index('mission')[metric].sort_index();c=energy[energy.config=='matched_const'].set_index('mission')[metric].sort_index();comparison.append({'metric':metric,**paired_bootstrap(a,c,10000,seed=2026)})
 pd.DataFrame(comparison).to_csv(ROOT/'energy-confirmation-comparisons.csv',index=False)
 A=agg.set_index('config');ratio=float(A.loc['matched_const','dE_j_mean']/A.loc['full_b1','dE_j_mean']);(ROOT/'energy-match-quality.json').write_text(json.dumps({'confirmation_energy_ratio_const_over_full':ratio,'within_10_percent':abs(ratio-1)<=.1,'no_test_outcome_retuning':True},indent=2))
 # Causal lag construction resets independently for each mission, before folds.
 original_role=ev.role_dataset;original_phase=ev.phase_dataset;original_plan=ev.plan_dataset
 def role(*a,**kw):
  r=original_role(*a,**kw);r.X=lag(r.X,r.n);return r
 def phase(*a,**kw):
  x,y=original_phase(*a,**kw);return lag(x),y
 def plan(*a,**kw):
  x,y,t=original_plan(*a,**kw);return lag(x),y,t
 ev.role_dataset=role;ev.phase_dataset=phase;ev.plan_dataset=plan
 ev.make_rf=lambda seed,*args:HistGradientBoostingClassifier(random_state=seed,max_iter=150,max_leaf_nodes=31,l2_regularization=1.,class_weight='balanced')
 rows=[]
 for key in ['none_b0','rot_b0','full_b0.2','full_b1','const_b0.2']:
  print('Alternative attacker',key,flush=True)
  for row in ev.evaluate_config(runs(PRIMARY,key,range(1,13)),AttackConfig(),2026):row['config']=key;rows.append(row)
  pd.DataFrame(rows).to_csv(ROOT/'history-hgb-per-mission.csv',index=False)
 aggregate(rows).to_csv(ROOT/'history-hgb-metrics.csv',index=False)
 history=pd.DataFrame(rows);hc=[]
 for key in history.config.unique():
  x=history[history.config==key].set_index('mission');y=primary[primary.config_key==key+'_'+SC].set_index('mission')
  for metric in ['role_acc_h0','role_acc_h10','role_acc_h30','phase_acc','plan_f1']:
   hc.append({'config':key,'metric':metric,**paired_bootstrap(x[metric].sort_index(),y[metric].sort_index(),10000,seed=2026)})
 pd.DataFrame(hc).to_csv(ROOT/'history-hgb-comparisons.csv',index=False)
 from swarmleak.experiments.provenance import environment,source_digest
 import hashlib
 (ROOT/'history-hgb-provenance.json').write_text(json.dumps({'environment':environment(),'frozen_scientific_source_sha256':source_digest(),'model':'HistGradientBoostingClassifier','max_iter':150,'max_leaf_nodes':31,'l2_regularization':1.,'class_weight':'balanced','seed':2026,'mission_folds':4,'lags':[0,1,2],'role_lag_stride':'one window, same UAV','hyperparameter_search':False,'plan_sha256':hashlib.sha256(Path('scripts/paper_supplement_plan.json').read_bytes()).hexdigest(),'evaluation_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
 (ROOT/'analysis.complete.json').write_text(json.dumps({'energy_confirmation_missions':8,'alternative_attacker_configs':5,'fixed_size_runs':24,'scope':'post-primary exploratory robustness checks','hyperparameter_search':False},indent=2))
if __name__=='__main__':main()
