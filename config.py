from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    path: str = 'nyu-mll/glue'
    """Dataset path."""

    name: str = 'qnli'
    """Dataset name."""

    split: str = 'train'
    """Dataset split."""

    batch_size: int = 64
    """Batch size."""

    n_samples: int = -1
    """Number of samples."""


@dataclass
class ModelConfig:
    name: str = 'google-t5/t5-base'
    """Model name."""


@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""

    seed: int = 42
    """Random seed."""
