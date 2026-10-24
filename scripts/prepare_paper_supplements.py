"""Prepare isolated supplementary configurations without modifying primary traces/runs."""
import copy,hashlib,json,statistics
from pathlib import Path
import yaml
BASE=Path.cwd();ROOT=BASE/'outputs/paper_supplements_20261022';ROOT.mkdir(exist_ok=True)
PRIMARY=BASE/'outputs/paper_primary_qwen25_7b_v3'
def cfg(name,grid,trace_root='data/primary_qwen25_7b_v3'):
 c=yaml.safe_load((BASE/'configs/mac_studio_structured_v3.yaml').read_text());c['experiment'].update(name=name,output_dir=str(Path('outputs/paper_supplements_20261022')/name),jobs=2);c['data']['root']=trace_root;c['grid']=grid;c['evaluation']={'bootstrap':10000,'comparisons':[]};return c
def save(name,c):
 p=BASE/'configs'/f'{name}.yaml';p.write_text(yaml.safe_dump(c,sort_keys=False));return p
if __name__=='__main__':
 originals=[BASE/'data/primary_qwen25_7b_v3'/f'llm_n20_m{i}.json' for i in range(1,13)]
 size=int(statistics.median(m['size'] for p in originals for m in json.loads(p.read_text())['messages']))
 dest=BASE/'data/size_control_20261022';dest.mkdir(exist_ok=True)
 manifest=[]
 for p in originals:
  d=json.loads(p.read_text());d['generator']['supplemental_treatment']={'kind':'pooled_median_fixed_size','size_bytes':size,'parent_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
  for m in d['messages']:m['size']=size
  q=dest/p.name;q.write_text(json.dumps(d));manifest.append({'parent':str(p.relative_to(BASE)),'parent_sha256':d['generator']['supplemental_treatment']['parent_sha256'],'derived':str(q.relative_to(BASE))})
 (ROOT/'size-treatment-manifest.json').write_text(json.dumps({'fixed_size_bytes':size,'traces':manifest},indent=2))
 save('supp_size_control',cfg('supp_size_control',[{'defense':'none','n_missions':12},{'defense':'full','budget':.2,'n_missions':12}],str(dest.relative_to(BASE))))
 save('supp_energy_calibration',cfg('supp_energy_calibration',[{'defense':'const','budgets':[.05,.1,.15],'missions':[1,2,3,4]}]))
 print('Prepared 24 size-control and 12 calibration runs; fixed size',size)
