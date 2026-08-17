from pathlib import Path

files = [
    Path("slurm/train/train_token_lora_celeba_four_label.slurm"),
    Path("slurm/train/train_token_lora_ctc_celeba_four_label.slurm"),
    Path("slurm/evaluation/evaluate_token_lora_four_do_guidance_sweep.slurm"),
    Path("slurm/evaluation/evaluate_token_lora_ctc_four_do_guidance_sweep.slurm"),
    Path("slurm/evaluation/export_token_lora_four_do_ca_style_grids_guidance.slurm"),
    Path("slurm/evaluation/export_token_lora_ctc_four_do_ca_style_grids_guidance.slurm"),
    Path("slurm/evaluation/export_token_only_four_do_ca_style_grids.slurm"),
    Path("slurm/evaluation/export_prompt_lora_balanced_four_do_ca_style_grids.slurm"),
]

insert = """if ! type module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/lmod/lmod/init/bash 2>/dev/null || true
fi
"""

for path in files:
    if not path.exists():
        print(f"missing {path}")
        continue
    text = path.read_text()
    if insert not in text:
        marker = "module load external\n"
        if marker not in text:
            print(f"no marker {path}")
            continue
        text = text.replace(marker, insert + marker, 1)
        path.write_text(text)
    print(f"patched {path}")
