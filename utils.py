import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5ForConditionalGeneration,
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
    if name.lower().startswith('google-t5/'):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = T5ForConditionalGeneration.from_pretrained(name)
    elif name.lower().startswith('qwen/'):
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(name)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    else:
        raise NotImplementedError
    torch.nn.Module.to(model, device)
    return tokenizer, model
