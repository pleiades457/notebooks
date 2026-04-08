from typing import Any

import tiktoken
import torch
from gpt import GPTModel
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(
        self, txt: str, tokenizer: tiktoken.Encoding, context_size: int, stride: int
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        self.target_ids: list[torch.Tensor] = []

        token_ids = tokenizer.encode(txt)
        for i in range(0, len(token_ids) - context_size, stride):
            input_chunks = token_ids[i : i + context_size]
            target_chunks = token_ids[i + 1 : i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt: str,
    context_size: int = 256,
    stride: int = 128,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
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


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: str,
    **kwargs: Any,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
        if optimizer is not None
        else None,
    }
    checkpoint.update(kwargs)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    filepath: str,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_hf_gpt2_params(model_name: str = "gpt2") -> tuple[Any, dict[str, Any]]:
    """Load GPT-2 params from HuggingFace. Returns the model config and a dict of params."""
    from transformers import GPT2Model

    model = GPT2Model.from_pretrained(model_name)
    hf_sd = model.state_dict()
    params: dict[str, Any] = {
        "wte": hf_sd["wte.weight"].numpy(),
        "wpe": hf_sd["wpe.weight"].numpy(),
        "g": hf_sd["ln_f.weight"].numpy(),
        "b": hf_sd["ln_f.bias"].numpy(),
        "blocks": [],
    }

    for i in range(model.config.n_layer):
        block = {
            "attn": {
                "c_attn": {
                    "w": hf_sd[f"h.{i}.attn.c_attn.weight"].numpy(),
                    "b": hf_sd[f"h.{i}.attn.c_attn.bias"].numpy(),
                },
                "c_proj": {
                    "w": hf_sd[f"h.{i}.attn.c_proj.weight"].numpy(),
                    "b": hf_sd[f"h.{i}.attn.c_proj.bias"].numpy(),
                },
            },
            "mlp": {
                "c_fc": {
                    "w": hf_sd[f"h.{i}.mlp.c_fc.weight"].numpy(),
                    "b": hf_sd[f"h.{i}.mlp.c_fc.bias"].numpy(),
                },
                "c_proj": {
                    "w": hf_sd[f"h.{i}.mlp.c_proj.weight"].numpy(),
                    "b": hf_sd[f"h.{i}.mlp.c_proj.bias"].numpy(),
                },
            },
            "ln_1": {
                "g": hf_sd[f"h.{i}.ln_1.weight"].numpy(),
                "b": hf_sd[f"h.{i}.ln_1.bias"].numpy(),
            },
            "ln_2": {
                "g": hf_sd[f"h.{i}.ln_2.weight"].numpy(),
                "b": hf_sd[f"h.{i}.ln_2.bias"].numpy(),
            },
        }
        params["blocks"].append(block)

    return model.config, params


def _assign(left: torch.Tensor, right: Any) -> torch.nn.Parameter:
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.as_tensor(right))


def load_weights_into_gpt(gpt: GPTModel, params: dict[str, Any]) -> None:
    gpt.pos_embedding.weight = _assign(gpt.pos_embedding.weight, params["wpe"])
    gpt.token_embedding.weight = _assign(gpt.token_embedding.weight, params["wte"])

    for i, blk in enumerate(params["blocks"]):
        tb = gpt.transformer_blocks[i]

        tb.attention.qkv.weight = _assign(
            tb.attention.qkv.weight, blk["attn"]["c_attn"]["w"].T
        )
        tb.attention.qkv.bias = _assign(
            tb.attention.qkv.bias, blk["attn"]["c_attn"]["b"]
        )
        tb.attention.out_proj.weight = _assign(
            tb.attention.out_proj.weight, blk["attn"]["c_proj"]["w"].T
        )
        tb.attention.out_proj.bias = _assign(
            tb.attention.out_proj.bias, blk["attn"]["c_proj"]["b"]
        )

        tb.ff.net[0].weight = _assign(tb.ff.net[0].weight, blk["mlp"]["c_fc"]["w"].T)
        tb.ff.net[0].bias = _assign(tb.ff.net[0].bias, blk["mlp"]["c_fc"]["b"])
        tb.ff.net[2].weight = _assign(tb.ff.net[2].weight, blk["mlp"]["c_proj"]["w"].T)
        tb.ff.net[2].bias = _assign(tb.ff.net[2].bias, blk["mlp"]["c_proj"]["b"])

        tb.ln1.scale = _assign(tb.ln1.scale, blk["ln_1"]["g"])
        tb.ln1.shift = _assign(tb.ln1.shift, blk["ln_1"]["b"])
        tb.ln2.scale = _assign(tb.ln2.scale, blk["ln_2"]["g"])
        tb.ln2.shift = _assign(tb.ln2.shift, blk["ln_2"]["b"])

    gpt.final_layer_norm.scale = _assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = _assign(gpt.final_layer_norm.shift, params["b"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, params["wte"])
