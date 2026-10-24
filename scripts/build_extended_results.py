"""Recompute descriptive contrasts and publication figures from saved mission results.
No model calls, network simulations, or changes to the scientific source.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/extended_analysis_20261024';OUT.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--manuscript-dir',type=Path);args=a.parse_args()
FD=OUT/'figures';FD.mkdir(exist_ok=True)
D=pd.read_csv(ROOT/'outputs/paper_primary_qwen25_7b_v3/per_mission.csv');M=pd.read_csv(ROOT/'outputs/paper_primary_qwen25_7b_v3/metrics.csv').set_index('config_key')
S=ROOT/'outputs/paper_supplements_20261022';H=pd.read_csv(S/'history-hgb-per-mission.csv')
C=['#236192','#D06739','#2B877A','#9467A3','#6F747C','#B69B31'];SC='_olsr_n20_v15_gm_quad'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':False,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
def save(fig,name):
 fig.savefig(FD/(name+'.png'),dpi=450,bbox_inches='tight');fig.savefig(FD/(name+'.svg'),bbox_inches='tight')
 if args.manuscript_dir:
  args.manuscript_dir.mkdir(exist_ok=True,parents=True);fig.savefig(args.manuscript_dir/(name+'.pdf'),bbox_inches='tight')
 plt.close(fig)
def vals(key,metric):return D[D.config_key==key][metric].to_numpy()
def ci(x):return t.ppf(.975,len(x)-1)*np.std(x,ddof=1)/np.sqrt(len(x))
def axes(ax):ax.grid(axis='x',alpha=.18,zorder=0);ax.set_axisbelow(True)
def label(ax,s):ax.set_title(s,loc='left',fontweight='bold',pad=12)
# Matched mission contrasts across every deployment variant.
rows=[]
scenarios=[('Default OLSR','olsr_n20_v15_gm_quad'),('AODV','aodv_n20_v15_gm_quad'),('Group mobility','olsr_n20_v15_rpgm_quad'),('Edge monitor','olsr_n20_v15_gm_edge'),('10 UAVs','olsr_n10_v15_gm_quad'),('30 UAVs','olsr_n30_v15_gm_quad'),('5 m/s','olsr_n20_v5_gm_quad'),('25 m/s','olsr_n20_v25_gm_quad')]
for name,sc in scenarios:
 x=D[D.config_key=='full_b0.2_'+sc].set_index('mission').sort_index();y=D[D.config_key=='none_b0_'+sc].set_index('mission').sort_index();assert x.index.equals(y.index)
 for metric in ['role_acc_h30','phase_acc','plan_f1','pdr','lat_ms_p95','dE_j_mean']:
  d=(x[metric]-y[metric]).to_numpy();boots=d[np.random.default_rng(2026).integers(0,len(d),(10000,len(d)))].mean(1);lo,hi=np.percentile(boots,[2.5,97.5]);rows.append(dict(scenario=name,metric=metric,n=len(d),mean_diff=d.mean(),lo=lo,hi=hi,negative_missions=int((d<0).sum()),positive_missions=int((d>0).sum())))
P=pd.DataFrame(rows);P.to_csv(OUT/'deployment_paired_comparisons.csv',index=False)
# Three secrets: shared configurations, separate panels and actual mission points.
keys=['none_b0','pad_b0','rot_b0','cover_uniform_b0.2','cover_targeted_b0.2','full_b0.2','const_b0.2'];names=['No defense','Padding','Rotation','Uniform cover','Targeted cover','Full protocol','Padding + constant']
fig,axs=plt.subplots(1,3,figsize=(8.3,3.25),sharey=True,layout='constrained')
for ax,metric,title in zip(axs,['role_acc_h30','phase_acc','plan_f1'],['(a) Role at 30 s','(b) Mission phase','(c) Plan-change F1']):
 for i,key in enumerate(keys):
  x=vals(key+SC,metric)*100;color=C[0] if key=='none_b0' else C[1] if key=='full_b0.2' else C[2];ax.scatter(x,np.arange(len(x))*0+ i+np.linspace(-.12,.12,len(x)),s=8,c=color,alpha=.25,zorder=2);ax.errorbar(x.mean(),i,xerr=ci(x),fmt='o',ms=4,color=color,capsize=2,zorder=3)
 ax.set_yticks(range(len(names)),names);ax.set_xlim(0,105);ax.set_xlabel('Attacker score (%)');label(ax,title);axes(ax)
axs[0].invert_yaxis();save(fig,'leakage_profile')
# Paired deployment contrasts. No independence across repeated configurations is assumed.
fig,axs=plt.subplots(1,3,figsize=(8.3,3.4),sharey=True,layout='constrained')
for ax,metric,title in zip(axs,['role_acc_h30','plan_f1','pdr'],['(a) Role at 30 s','(b) Plan-change F1','(c) Delivery ratio']):
 q=P[P.metric==metric].set_index('scenario').loc[[s[0] for s in scenarios]];x=q.mean_diff.to_numpy()*100;ax.errorbar(x,range(len(q)),xerr=np.vstack([x-q.lo.to_numpy()*100,q.hi.to_numpy()*100-x]),fmt='o',color=C[0],ms=4,capsize=2);ax.axvline(0,c=C[4],lw=1,ls='--');ax.set_yticks(range(len(q)),[s[0]+f' (n={int(q.iloc[i].n)})' for i,s in enumerate(scenarios)]);ax.set_xlabel('Full − no defense (pp)');label(ax,title);axes(ax)
axs[0].invert_yaxis();save(fig,'deployment_effects')
# Precision and recall operating points; these are not threshold-sweep curves.
fig,axs=plt.subplots(1,2,figsize=(8.3,3.2),layout='constrained')
for ax,metric,title in zip(axs,['plan_prec','plan_rec'],['(a) Event precision','(b) Event recall']):
 for i,key in enumerate(keys):
  x=vals(key+SC,metric)*100;ax.errorbar(x.mean(),i,xerr=ci(x),fmt='o',color=C[1] if key=='full_b0.2' else C[0],ms=4,capsize=2);ax.scatter(x,i+np.linspace(-.12,.12,len(x)),s=8,c=C[0],alpha=.2)
 ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_xlabel('Score (%)');label(ax,title);axes(ax)
save(fig,'event_operating_points')
# Energy-leakage space: measured energy, not nominal budget.
fig,axs=plt.subplots(1,2,figsize=(8.3,3.35),layout='constrained')
plotkeys=['rot_b0','cover_uniform_b0.2','cover_uniform_b1','cover_targeted_b0.2','cover_targeted_b1','full_b0.05','full_b0.2','full_b1','const_b0.2','const_b1'];codes=['R','U.2','U1','T.2','T1','F.05','F.2','F1','C.2','C1']
for ax,metric,title in zip(axs,['role_acc_h30','plan_f1'],['(a) Role at 30 s','(b) Plan-change F1']):
 for key,code in zip(plotkeys,codes):
  x=vals(key+SC,'dE_j_mean');y=vals(key+SC,metric)*100;color=C[1] if key.startswith('full') else C[2] if key.startswith('const') else C[0];ax.errorbar(x.mean(),y.mean(),xerr=ci(x),yerr=ci(y),fmt='s' if key.startswith('const') else 'o',color=color,ms=4,alpha=.85,lw=.7,capsize=1);ax.annotate(code,(x.mean(),y.mean()),xytext=({'T1':(5,7),'F1':(5,-10),'C.2':(-7,-10)}.get(code,(4,3)) if metric=='plan_f1' else (4,3)),textcoords='offset points',fontsize=7)
 ax.set_xlabel('Measured incremental energy (J/UAV)');ax.set_ylabel('Attacker score (%)');label(ax,title);ax.grid(alpha=.16);ax.set_xscale('symlog',linthresh=.08);ax.set_xlim(-.04,6);ax.set_xticks([0,.1,1,5],['0','0.1','1','5'])
save(fig,'measured_energy_tradeoffs')
# Routing costs with per-mission spread.
fig,axs=plt.subplots(1,3,figsize=(8.3,3.25),sharey=True,layout='constrained');routing=[('No defense / OLSR','none_b0_olsr'),('Full / OLSR','full_b0.2_olsr'),('Any-hop / OLSR','full_anyhop_b0.2_olsr'),('No defense / AODV','none_b0_aodv'),('Full / AODV','full_b0.2_aodv'),('Any-hop / AODV','full_anyhop_b0.2_aodv')]
for ax,metric,title,scale in zip(axs,['pdr','lat_ms_p95','bcast_fps'],['(a) Delivery ratio (%)','(b) Delivered-packet p95 (ms)','(c) Broadcast frames/s'],[100,1,1]):
 for i,(name,key) in enumerate(routing):
  x=vals(key+'_n20_v15_gm_quad',metric)*scale;ax.scatter(x,i+np.linspace(-.15,.15,len(x)),s=11,alpha=.4,color=C[1] if 'anyhop' in key else C[0]);ax.plot(np.median(x),i,'|',ms=12,color='black')
 ax.set_yticks(range(6),[x[0] for x in routing]);label(ax,title);axes(ax)
axs[0].invert_yaxis();axs[1].set_xscale('symlog',linthresh=1);save(fig,'routing_costs')
# Architecture schematic: drawn in vector form from the implemented pipeline.
fig,ax=plt.subplots(figsize=(8.3,3.55));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
def box(x,y,w,h,title,body,color):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.12,rounding_size=.12',facecolor=color,edgecolor='#9BA8B0',lw=.8));ax.text(x+w/2,y+h-.27,title,ha='center',va='top',fontsize=9.5,fontweight='bold',color='#17374A');ax.text(x+w/2,y+h/2-.22,body,ha='center',va='center',fontsize=8.5,linespacing=1.3)
def arrow(x,y,xx,yy,color='#4E6573',ls='-'):ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=12,lw=1.2,color=color,linestyle=ls))
box(.15,2.4,2.4,2,'LLM traces','Qwen2.5:7B\nPlans / reports\nPeer queries\nLogical time + size','#EAF1F7');box(3.05,2.4,2.5,2,'Obfuscation','Role reassignment\nOne-hop cover\nOptional padding\nLocal admission','#FBEDE5');box(6.1,2.4,2.45,2,'Wireless replay','3D mobility\nOLSR / AODV\n802.11 ad hoc, ns-3\nForwarding + energy','#E7F3EF');box(9.1,2.4,2.6,2,'Observation','Decoded MAC frames\nSize / time / TX / RX\nMerged monitor view','#ECECF5')
for x,xx in [(2.55,3.05),(5.55,6.1),(8.55,9.1)]:arrow(x,3.4,xx,3.4)
box(2.6,.25,3.6,1.3,'Held-out attacker','Mission-disjoint folds\nRole / phase / events','#EFF3F5');box(7,.25,3.6,1.3,'Evidence and cost','Mission score intervals\nDelivery / latency / energy','#EFF3F5');arrow(10.3,2.4,10.3,1.85);arrow(10.3,1.85,4.4,1.85);arrow(4.4,1.85,4.4,1.55);arrow(6.2,.9,7,.9);arrow(7.4,2.4,8.4,1.55,ls='--');ax.text(.15,.35,'Open-loop replay:\nloss does not alter\nlogical task timing.',fontsize=8,color='#586E7C');save(fig,'pipeline_overview')
# Timeline distinguishes mapping switch from later return state message.
fig,ax=plt.subplots(figsize=(8.3,2.9));lanes=[3,2,1,0];labs=['Orchestrator host','Decoy neighbor','Worker','Successor host']
for y,lab in zip(lanes,labs):ax.hlines(y,0,10,color='#D4DDE2',lw=.9);ax.text(-.15,y,lab,ha='right',va='center',fontsize=9)
for x,y,c in [(1,3,C[0]),(1.2,3,C[0]),(1.4,3,C[0]),(2.2,2,C[1]),(2.4,2,C[1]),(3.2,1,C[0]),(3.5,0,C[2]),(4.7,1,C[0]),(5,2,C[2]),(8.4,0,C[0]),(8.6,0,C[0]),(8.8,0,C[0])]:ax.plot(x,y,'s',c=c,ms=5)
ax.axvline(.85,c=C[4],ls='--',lw=.8);ax.text(.85,3.5,'Plan change',ha='center',fontsize=8);ax.axvline(6.6,c=C[4],ls='--',lw=.8);ax.text(6.6,3.5,'Handover scheduled',ha='center',fontsize=8)
ax.annotate('',(6.6,.05),(6.6,2.95),arrowprops=dict(arrowstyle='->',color=C[3],lw=1.2));ax.annotate('',(7,2.95),(7,.05),arrowprops=dict(arrowstyle='->',color=C[3],lw=1.2));ax.text(7.15,1.8,'State-sized datagrams;\nrole mapping switches\nafter 50 ms without ACK.',fontsize=8,va='center');ax.annotate('Later plans originate\nfrom the successor',(8.6,.1),(8.5,.7),fontsize=8,ha='center');ax.set_xlim(-2.8,10);ax.set_ylim(-.6,4);ax.axis('off');ax.text(1,-.55,'Blue: real traffic     Orange: decoy burst     Green: mirrored report',fontsize=8);save(fig,'handover_timeline')
# Supplemental figure generated only when the new feature control has completed.
F=ROOT/'outputs/feature_ablation_20261024'
if (F/'provenance.json').exists():
 q=pd.read_csv(F/'metrics.csv');fig,axs=plt.subplots(1,2,figsize=(8.3,3.1),layout='constrained')
 labels=['Original features','Size-blind access','Counts + timing']
 for ax,metric,title in zip(axs,['role_acc_h30','plan_f1'],['(a) Role at 30 s','(b) Plan-change F1']):
  for j,(config,name,color) in enumerate([('none_b0','No defense',C[0]),('full_b0.2','Full, B = 0.2 J',C[1])]):
   ref=M.loc[config+SC];ys=[ref[metric]];errs=[ref[metric+'_ci']]
   for mask in ['size_blind','counts_timing']:
    row=q[(q.config==config)&(q.feature_access==mask)].iloc[0];ys.append(row[metric]);errs.append(row[metric+'_ci'])
   ax.errorbar(np.arange(3)+(j-.5)*.10,np.array(ys)*100,yerr=np.array(errs)*100,fmt='o',color=color,capsize=2,label=name)
  ax.set_xticks(range(3),labels,rotation=12);ax.set_ylabel('Attacker score (%)');ax.set_ylim(0,110);ax.grid(axis='y',alpha=.16);label(ax,title);ax.legend(loc='lower left',frameon=False)
 save(fig,'feature_access_controls')
(OUT/'provenance.json').write_text(json.dumps({'input':'unchanged primary per_mission.csv and aggregate metrics.csv','paired_mission_contrasts':len(P),'bootstrap_samples':10000,'seed':2026,'new_network_simulations':0,'exploratory':True,'within_scenario_pairing':True,'multiple_comparisons_adjusted':False},indent=2))
print(P[P.metric.isin(['role_acc_h30','pdr'])].to_string(index=False))
