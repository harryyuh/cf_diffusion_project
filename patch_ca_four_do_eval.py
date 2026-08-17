from pathlib import Path

p = Path("evaluation/compare_causal_adapter_parent_only_celeba.py")
text = p.read_text()

if 'p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, default="Male")' not in text:
    text = text.replace(
        '    p.add_argument("--ca-editing", choices=["p2p", "standard"], default="p2p")\n',
        '    p.add_argument("--ca-editing", choices=["p2p", "standard"], default="p2p")\n'
        '    p.add_argument("--intervention-attr", choices=COMPLEX_ATTRS, default="Male")\n',
    )

text = text.replace(
    '        target_male = 1.0 - labels["Male"]\n',
    '        intervention_attr = self.args.intervention_attr\n'
    '        intervention_idx = COMPLEX_ATTRS.index(intervention_attr)\n'
    '        target_attr = 1.0 - labels[intervention_attr]\n',
)
text = text.replace(
    '            target_value = target_male[i, 0]\n',
    '            target_value = target_attr[i, 0]\n',
)
text = text.replace(
    '                    intervention_indx=1,\n',
    '                    intervention_indx=intervention_idx,\n',
)
text = text.replace(
    '                target_labels[:, 1] = target_value\n',
    '                target_labels[:, intervention_idx] = target_value\n',
)
text = text.replace(
    '            slot["counterfactuals"].extend(\n'
    '                zip(_pixel_embedding(fake), cf["Male"].detach().cpu().numpy().astype(np.int64))\n'
    '            )\n'
    '            slot["interventions"].extend(["Male"] * real.shape[0])\n',
    '            intervention_attr = args.intervention_attr\n'
    '            slot["counterfactuals"].extend(\n'
    '                zip(_pixel_embedding(fake), cf[intervention_attr].detach().cpu().numpy().astype(np.int64))\n'
    '            )\n'
    '            slot["interventions"].extend([intervention_attr] * real.shape[0])\n',
)
text = text.replace(
    '        "num_samples": len(indices),\n',
    '        "num_samples": len(indices),\n'
    '        "intervention_attr": args.intervention_attr,\n',
)

p.write_text(text)
print("patched", p)
