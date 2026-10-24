"""Freeze energy-only selection using missions 1--4; never inspect attack outcomes."""
import json
from pathlib import Path
import numpy as np
from prepare_paper_supplements import cfg,save,ROOT,PRIMARY
from swarmleak.data.schemas import RunRecord
from swarmleak.metrics.network import incremental_energy
base='olsr_n20_v15_gm_quad'
def load(folder,defense,budget,mission):return RunRecord.load(folder/'runs'/f'{defense}_b{budget:g}_{base}_m{mission}.npz')
rows=[]
for budget in [.05,.1,.15]:
 vals=[incremental_energy(load(ROOT/'supp_energy_calibration','const',budget,m),load(PRIMARY,'none',0,m)) for m in range(1,5)]
 rows.append({'budget':budget,'mean_energy':float(np.mean(vals)),'per_mission_energy':vals})
target=float(np.mean([incremental_energy(load(PRIMARY,'full',1,m),load(PRIMARY,'none',0,m)) for m in range(1,5)]))
energies=[r['mean_energy'] for r in rows]
assert all(b>a for a,b in zip(energies,energies[1:])),'Non-monotone energy calibration: stop rather than choose by leakage'
budget=round(float(np.interp(target,energies,[r['budget'] for r in rows])),6)
selection={'calibration_missions':[1,2,3,4],'confirmation_missions':list(range(5,13)),'target_full_b1_energy_j':target,'calibration':rows,'frozen_const_budget':budget,'clipped_to_calibration_range':not energies[0]<=target<=energies[-1],'selection_uses_leakage':False}
p=ROOT/'energy-selection.json'
if p.exists():assert json.loads(p.read_text())==selection,'Frozen selection changed'
else:p.write_text(json.dumps(selection,indent=2))
save('supp_energy_confirmation',cfg('supp_energy_confirmation',[{'defense':'const','budget':budget,'missions':list(range(5,13))}]))
print(json.dumps(selection,indent=2))
