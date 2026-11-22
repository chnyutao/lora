import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from config import Config


class Assets:
    @staticmethod
    def sv(config: Config) -> Path:
        model = config.model.name.split('/')[-1]
        dataset = config.dataset.name
        path = Path(f'asset/sv/{model}-{dataset}')
        os.makedirs(path, exist_ok=True)
        return path


def make_dataset(config: Config) -> Dataset:
    ds = load_dataset(
        config.dataset.path,
        config.dataset.name,
        split=config.dataset.split,
    )
    ds = ds.shuffle(config.seed)
    print(f'Loading dataset: {ds}')
    ds = ds.batch(config.dataset.batch_size)
    match (config.dataset.path, config.dataset.name):
        case ('nyu-mll/glue', 'cola' | 'sst2'):
            return ds.map(lambda x: {'x': x['sentence']})
        case ('nyu-mll/glue', 'qnli'):
            return ds.map(lambda x: {'x': x['question']})
        case _:
            raise NotImplementedError


def make_model(
    config: Config, device: str | torch.device
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    name = config.model.name
    if name.startswith('google-t5/'):
        tokenizer = T5Tokenizer.from_pretrained(name)
        model = T5ForConditionalGeneration.from_pretrained(name)
    else:
        raise NotImplementedError
    torch.nn.Module.to(model, device)
    return tokenizer, model
