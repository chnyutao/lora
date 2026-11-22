import torch
import tyro
from tqdm.auto import tqdm

from config import Config
from utils import Assets, make_dataset, make_model

# configuration
config = tyro.cli(Config)
acts: list[torch.Tensor] = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# initialize dataset & model
dataset = make_dataset(config)
tokenizer, model = make_model(config, device)
model.lm_head.register_forward_pre_hook(
    lambda _, h: acts.append(h[0].squeeze().detach().cpu())
)

# loop over batches
n_batches = config.dataset.n_samples // config.dataset.batch_size
for batch in tqdm(dataset.take(n_batches), desc='Inference'):
    assert isinstance(batch, dict)
    x = tokenizer(batch['x'], padding=True, return_tensors='pt')
    with torch.inference_mode():
        model.generate(**x.to(device), max_new_tokens=1)  # type: ignore

# compute and save metrics
H = torch.cat(acts, dim=0).float()
print(f'H matrix shape: {H.shape}')
S = torch.linalg.svdvals(H)
torch.save(S, Assets.sv(config) / f'{config.seed}.pt')
K = int(((S**2).cumsum(dim=0) < 0.99 * (S**2).sum()).sum()) + 1
print(f'Top-{K} singular values >= 99% spectral energy.')
