"""Merge restartable full-test shards using the exact CA FID reference split."""
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset

ROOT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_fulltest_ddim100")
DATA = Path("/work/nvme/bhje/hyu29/datasets/celebA")
EXPECTED = 19962

class RealImages(Dataset):
    def __init__(self, split):
        ds = CelebADataset(str(DATA), split, CELEBA_ATTR_ORDER, image_size=64)
        self.names = list(ds._filenames)
        self.tf = transforms.Compose([transforms.CenterCrop(150), transforms.Resize((64,64),
            interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
    def __len__(self): return len(self.names)
    def __getitem__(self, i): return self.tf(Image.open(DATA/"img_align_celeba"/self.names[i]).convert("RGB"))

def make_metric(device):
    return FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)

def chunks(method):
    files = sorted((ROOT/method).glob("shard_*_*/tensors/batch_*.pt"))
    count = 0
    for path in files:
        x = torch.load(path, map_location="cpu")["images"].float(); count += len(x); yield x
    if count != EXPECTED: raise RuntimeError(f"{method}: expected {EXPECTED}, found {count} across {len(files)} files")

device = torch.device("cuda")
metrics = {m: make_metric(device) for m in ("ca", "label_only")}
for batch in DataLoader(RealImages("train"), batch_size=128, num_workers=8, pin_memory=True):
    for metric in metrics.values(): metric.update(batch.to(device, non_blocking=True), real=True)
result = {}
for name, metric in metrics.items():
    for batch in chunks(name): metric.update(batch.to(device), real=False)
    result[name] = {"fid_ca_protocol_fulltest_fake_full_train_real": float(metric.compute().cpu()), "num_fake": EXPECTED}
result["protocol"] = {"real_reference": "complete CelebA train split", "fake_cohort": "complete 19,962-image test split",
    "interventions": "four attributes uniformly pooled", "ddim_steps": 100,
    "implementation": "TorchMetrics FID normalize=True; CenterCrop(150), bicubic 64x64"}
(ROOT/"merged_metrics.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result, indent=2))
