import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance

from data.celeba_dataset import CELEBA_ATTR_ORDER, CelebADataset

ROOT = Path("/work/nvme/bhje/hyu29/evaluations/celeba/ca_protocol_n2048_ddim100")
DATA = Path("/work/nvme/bhje/hyu29/datasets/celebA")


class RealImages(Dataset):
    def __init__(self, split):
        ds = CelebADataset(root=str(DATA), split=split, factor_cols=list(CELEBA_ATTR_ORDER), image_size=64)
        self.names = list(ds._filenames)
        self.tf = transforms.Compose([
            transforms.CenterCrop(150),
            transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        return self.tf(Image.open(DATA / "img_align_celeba" / str(self.names[i])).convert("RGB"))


def metric(device):
    return FrechetInceptionDistance(normalize=True, reset_real_features=False).set_dtype(torch.float32).to(device)


def fake_chunks(method):
    files = sorted((ROOT / method).glob("shard_*_*/tensors/batch_*.pt"))
    count = 0
    for path in files:
        x = torch.load(path, map_location="cpu")["images"].float()
        count += len(x)
        yield x
    if count != 2048:
        raise RuntimeError(f"{method}: expected 2048 generated images, found {count} in {len(files)} chunks")


def main():
    device = torch.device("cuda")
    methods = ["ca", "label_only"]
    metrics = {name: metric(device) for name in methods}
    real_loader = DataLoader(RealImages("train"), batch_size=128, shuffle=False, num_workers=8, pin_memory=True)
    for batch in real_loader:
        batch = batch.to(device, non_blocking=True)
        for obj in metrics.values(): obj.update(batch, real=True)
    results = {}
    for name, obj in metrics.items():
        for batch in fake_chunks(name): obj.update(batch.to(device), real=False)
        results[name] = {"fid_ca_protocol_n2048_fake_full_train_real": float(obj.compute().cpu()), "num_fake": 2048}

    # Same-distribution finite-sample diagnostic, independent 2,048-image test subsets.
    sanity = metric(device)
    sanity_loader = DataLoader(RealImages("test"), batch_size=128, shuffle=False, num_workers=8)
    seen = 0
    for batch in sanity_loader:
        if seen >= 4096: break
        take = min(len(batch), 4096 - seen)
        batch = batch[:take].to(device)
        cut = max(0, min(take, 2048 - seen))
        if cut: sanity.update(batch[:cut], real=True)
        if seen + take > 2048: sanity.update(batch[max(0, 2048-seen):], real=False)
        seen += take
    results["real_vs_real_n2048"] = {"fid": float(sanity.compute().cpu())}
    results["protocol"] = {
        "real_reference": "complete CelebA train split",
        "fake_cohort": "first 2048 indices from complete test selection",
        "interventions": "balanced pooled; two 256-image shards per Young/Male/No_Beard/Bald",
        "implementation": "TorchMetrics FrechetInceptionDistance(normalize=True), 64x64 CenterCrop(150)+bicubic",
    }
    out = ROOT / "merged_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(out)
    print(json.dumps(results, indent=2))

if __name__ == "__main__": main()

