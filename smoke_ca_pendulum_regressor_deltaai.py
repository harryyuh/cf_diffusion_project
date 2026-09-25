import os
import sys

BENCHMARK_ROOT = (
    "/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba/external/Causal-Adapter/"
    "counterfactual-benchmark/counterfactual_benchmark"
)
CA_ROOT = "/work/nvme/bhje/hyu29/repos/cf-diffusion-celeba/external/Causal-Adapter"
sys.path.insert(0, os.path.join(BENCHMARK_ROOT, "methods", "deepscm"))
sys.path.insert(0, BENCHMARK_ROOT)
sys.path.insert(0, CA_ROOT)

import pytorch_lightning
import torch

from ctf_datasets.pendulum import dataset
from models.classifiers.pendulum_classifier import PendClassifier
import train_classifier  # noqa: F401 - validates the official training entrypoint


root = str(dataset.PENDULUM_ROOT)
files = os.listdir(os.path.join(root, "train"))
assert files, root

model = PendClassifier(
    attr="pendulum", width=8, num_outputs=1, context_dim=2, lr=1e-4
).cuda()
output = model(torch.rand(1, 3, 96, 96, device="cuda"))
output.square().mean().backward()

print(
    "SMOKE_OK",
    torch.__version__,
    pytorch_lightning.__version__,
    torch.cuda.get_device_name(0),
    root,
    len(files),
    tuple(output.shape),
)
