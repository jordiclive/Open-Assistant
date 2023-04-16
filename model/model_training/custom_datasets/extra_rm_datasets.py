from model_training.custom_datasets.rank_datasets import AnthropicRLHF, HellaSwagDataset, SHPDataset
from torch.utils.data import Dataset


def load_anthropic_rlhf() -> Tuple[Dataset, Dataset]:
    train = AnthropicRLHF(split="train")
    validation = AnthropicRLHF(split="test")
    return train, validation


def load_shp() -> Tuple[Dataset, Dataset]:
    train = SHPDataset(split="train")
    validation = SHPDataset(split="validation")
    return train, validation


def load_hellaswag() -> Tuple[Dataset, Dataset]:
    train = HellaSwagDataset(split="train")
    validation = HellaSwagDataset(split="validation")
    return train, validation
