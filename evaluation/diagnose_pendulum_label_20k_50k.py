import json
from pathlib import Path
import torch
from models.prompt_aligned_injection import JointLabelTokenInjection

roots = {
 '20k': '/work/nvme/bhje/hyu29/ckpts/pendulum/joint_label_lora/minisd_label_equal_budget_official_protocol_20k_seed42/checkpoints/step_020000',
 '50k': '/work/nvme/bhje/hyu29/ckpts/pendulum/paper_setting_50k_bs40/label_only/minisd_label_50k_bs40_seed42/final',
}
result = {}
for name, root in roots.items():
 payload = torch.load(Path(root)/'mapper/pai_mapper.pt', map_location='cpu')
 meta = payload['mapper']
 model = JointLabelTokenInjection(meta['attr_names'], int(meta['hidden_dim']), int(meta.get('projector_hidden_dim',256)))
 model.load_state_dict(payload['model_state_dict']); model.eval()
 labels = torch.full((9,4),0.5)
 for j in range(4): labels[1+2*j,j]=0; labels[2+2*j,j]=1
 with torch.no_grad(): tokens=model.causal_tokens(labels)
 result[name]={'token_rms':float(tokens.square().mean().sqrt()), 'parameter_norms':{k:float(v.norm()) for k,v in model.state_dict().items()},
  'attribute_endpoint_delta_rms':{a:float((tokens[1+2*j]-tokens[2+2*j]).square().mean().sqrt()) for j,a in enumerate(meta['attr_names'])}}
print(json.dumps(result,indent=2))
out=Path('/work/nvme/bhje/hyu29/evals/pendulum/label_20k_vs_50k_mapper_diagnostic.json')
out.write_text(json.dumps(result,indent=2)+'\n')
