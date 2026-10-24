"""Plot saved verified aggregates only; does not rerun or alter experiments."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
p=root/'outputs/paper_primary_qwen25_7b_v3'
d=pd.read_csv(p/'metrics.csv').set_index('config_key')
keys=['none_b0','pad_b0','rot_b0','full_b0.2','full_b1','const_b0.2']
labels=['No defense','Fixed padding','Rotation only','Full, B = 0.2 J','Full, B = 1 J','Padding + constant, B = 0.2 J']
rows=d.loc[[x+'_olsr_n20_v15_gm_quad' for x in keys]]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,bx)=plt.subplots(1,2,figsize=(9,4.8),sharey=True,gridspec_kw={'width_ratios':[1.3,1]})
y=np.arange(len(rows))
for h,dy,col,mark in [(0,-.18,'#4477AA','o'),(10,0,'#EE6677','s'),(30,.18,'#228833','^')]:
 ax.errorbar(rows[f'role_acc_h{h}']*100,y+dy,xerr=rows[f'role_acc_h{h}_ci']*100,fmt=mark,color=col,markersize=4,capsize=2,linewidth=1,label=f'{h} s')
bx.errorbar(rows.plan_f1*100,y,xerr=rows.plan_f1_ci*100,fmt='o',color='#555555',markersize=4,capsize=2,linewidth=1)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,104);bx.set_xlim(40,104)
for a in [ax,bx]:a.grid(axis='x',alpha=.22);a.set_axisbelow(True)
ax.set_xlabel('Role accuracy (%)');bx.set_xlabel('Plan-change F1 (%)')
ax.set_title('(a) Action horizon');bx.set_title('(b) Plan-change detection')
ax.legend(title='Horizon',loc='lower left',ncol=3,fontsize=10,title_fontsize=10,frameon=False)
fig.tight_layout();dest=root/'outputs/figures';dest.mkdir(parents=True,exist_ok=True)
fig.savefig(dest/'verified_results.png',dpi=300,bbox_inches='tight');plt.close(fig)
