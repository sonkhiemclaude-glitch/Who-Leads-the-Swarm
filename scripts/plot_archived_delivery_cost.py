"""Figures for archived-run delivery and energy diagnostics; no fitted models."""
from pathlib import Path
import argparse
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'outputs/archive_reanalysis_20261026';F=D/'figures';F.mkdir(exist_ok=True)
a=argparse.ArgumentParser();a.add_argument('--manuscript-dir',type=Path);args=a.parse_args()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,'axes.labelsize':10,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
BLUE='#236192';TEAL='#2B877A';ORANGE='#D06739'
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=450,bbox_inches='tight');fig.savefig(F/(name+'.svg'),bbox_inches='tight')
 if args.manuscript_dir:fig.savefig(args.manuscript_dir/(name+'.pdf'),bbox_inches='tight')
 plt.close(fig)
C=pd.read_csv(D/'original_traffic_paired_comparisons.csv');H=pd.read_csv(D/'handover_payload_summary.csv');HM=pd.read_csv(D/'handover_payloads_per_mission.csv');N=pd.read_csv(D/'per_uav_costs.csv')
fig,axs=plt.subplots(1,2,figsize=(8.3,3.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.35]})
ax=axs[0]
for j,(routing,color) in enumerate([('olsr',BLUE),('aodv',TEAL)]):
 q=C[(C.config=='full_b0.2')&(C.routing==routing)&(C.metric=='pdr')].set_index('kind').loc[['all_original','plan','ack','report']];x=q.mean_diff.to_numpy()*100;y=np.arange(4)+(j-.5)*.12
 ax.errorbar(x,y,xerr=np.vstack([x-q.lo.to_numpy()*100,q.hi.to_numpy()*100-x]),fmt='o',color=color,capsize=2,ms=4,label=routing.upper())
ax.axvline(0,color='#777',lw=.7,ls='--');ax.set_yticks(range(4),['All originals','Plans','ACKs','Reports']);ax.invert_yaxis();ax.set_xlabel('Original PDR: full − none (pp)');ax.set_title('(a) Common-workload utility',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower left')
ax=axs[1];cases=[('rot_b0','olsr','Rotation / OLSR'),('full_b0.2','olsr','Full / OLSR'),('full_b0.2','aodv','Full / AODV'),('full_anyhop_b0.2','olsr','Any-hop / OLSR'),('full_anyhop_b0.2','aodv','Any-hop / AODV')]
for j,(metric,label,color) in enumerate([('both_delivered','Both payloads received',BLUE),('outgoing_before_switch','Outgoing received by 50 ms',ORANGE)]):
 for i,(c,r,name) in enumerate(cases):
  q=H[(H.config==c)&(H.routing==r)].iloc[0];ax.errorbar(q[metric]*100,i+(j-.5)*.14,xerr=q[metric+'_ci']*100,fmt='o',ms=4,color=color,capsize=2,label=label if i==0 else None)
ax.set_yticks(range(len(cases)),[x[2] for x in cases]);ax.invert_yaxis();ax.set_xlim(40,105);ax.set_xlabel('Mean per-mission fraction (%)');ax.set_title('(b) Handover payload reception',loc='left',fontweight='bold');ax.grid(axis='x',alpha=.15);ax.legend(frameon=False,loc='lower left',fontsize=7)
save(fig,'archived_utility_handover')
fig,axs=plt.subplots(1,2,figsize=(8.3,3.25),sharey=True,layout='constrained')
cases=[('full_b0.2','olsr','Full / OLSR'),('full_b0.2','aodv','Full / AODV'),('full_anyhop_b0.2','olsr','Any-hop / OLSR'),('full_anyhop_b0.2','aodv','Any-hop / AODV'),('const_b0.2','olsr','Pad.+additive / OLSR')]
for ax,metric,scale,title,xlabel in zip(axs,['utilization','increment_radio_j'],[100,1],['(a) Ledger utilization','(b) Incremental radio energy'],['Estimated ledger use (% of B)','Paired radio increment (J/UAV)']):
 for i,(c,r,name) in enumerate(cases):
  x=N[(N.config==c)&(N.routing==r)][metric].to_numpy()*scale;color=TEAL if r=='aodv' else BLUE
  ax.scatter(x,i+np.linspace(-.14,.14,len(x)),s=4,color=color,alpha=.10);lo,med,hi=np.quantile(x,[.1,.5,.9]);ax.hlines(i,lo,hi,color=color,lw=2.2);ax.plot(med,i,'o',color=color,ms=4)
 ax.set_yticks(range(len(cases)),[x[2] for x in cases]);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel(xlabel);ax.grid(axis='x',alpha=.15)
axs[0].invert_yaxis();axs[0].set_xlim(0,105);axs[1].axvline(0,color='#777',lw=.7);save(fig,'per_uav_ledger_energy')
