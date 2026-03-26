from typing import Any

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt: str, tokenizer, context_size: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - context_size, stride):
            input_chunks = token_ids[i : i + context_size]
            target_chunks = token_ids[i + 1 : i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    context_size=256,
    stride=128,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, context_size, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def text_to_tokens(text: str, tokenizer: Any) -> list[int]:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor


def tokens_to_text(tokens: torch.Tensor, tokenizer: Any) -> str:
    tokens_list = tokens.squeeze(0).tolist()
    text = tokenizer.decode(tokens_list)
    return text


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(model, optimizer, filepath, **kwargs):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, filepath, device=None):
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint
