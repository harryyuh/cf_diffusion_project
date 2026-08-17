"""Train the four-output Pendulum regressor used for intervention evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data.pendulum_dataset import PENDULUM_ATTRS, PendulumDataset


class PendulumRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 4))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image)).sigmoid()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device('cuda')

    full = PendulumDataset(args.data_root, 'train', image_size=128)
    n_val = max(1, round(0.1 * len(full)))
    train, val = random_split(full, [len(full) - n_val, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = PendulumRegressor().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best = float('inf')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            pred = model(batch['image'].to(device))
            loss = F.mse_loss(pred, batch['factors'].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        abs_sum = torch.zeros(4, device=device)
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch['image'].to(device))
                abs_sum += (pred - batch['factors'].to(device)).abs().sum(0)
                count += pred.shape[0]
        mae = abs_sum / count
        score = float(mae.mean())
        print(f'epoch={epoch + 1} val_mae={score:.6f} per_attr={mae.tolist()}', flush=True)
        if score < best:
            best = score
            torch.save({'model_state_dict': model.state_dict(), 'attrs': PENDULUM_ATTRS,
                        'image_size': 128, 'val_mae_normalized': mae.cpu(), 'epoch': epoch + 1}, out)

    (out.with_suffix('.json')).write_text(json.dumps({'best_val_mae_normalized': best}, indent=2))


if __name__ == '__main__':
    main()
