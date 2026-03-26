from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn as nn


@dataclass
class GPTConfig:
    # default is GPT-2 small config
    vocab_size: int = 50257
    context_length: int = field(default=1024, metadata={"alias": "n_ctx"})
    embedding_dim: int = field(default=768, metadata={"alias": "n_embd"})
    num_heads: int = field(default=12, metadata={"alias": "n_head"})
    num_layers: int = field(default=12, metadata={"alias": "n_layer"})
    dropout: float = field(default=0.1, metadata={"alias": "attn_pdrop"})
    qkv_bias: bool = False

    @classmethod
    def from_dict(cls, config_dict: dict):
        d = {}
        for field_ in fields(cls):
            name = field_.name
            alias = field_.metadata.get("alias")
            if name in config_dict:
                d[name] = config_dict[name]
            if alias and alias in config_dict:
                d[name] = config_dict[alias]
        return cls(**d)

    def to_dict(self):
        return asdict(self)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, embedding_dim, num_heads, context_length, dropout=0.0, qkv_bias=False
    ):
        super().__init__()
        assert embedding_dim % num_heads == 0, (
            "embedding_dim must be divisible by num_heads"
        )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # combine q, k, v projections into a single linear layer for efficiency
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # compute q, k, v in one go and then split them
        qkv = self.qkv(x)  # (batch_size, seq_len, embedding_dim * 3)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, batch_size, num_heads, seq_len, head_dim)
        query, keys, values = qkv[0], qkv[1], qkv[2]

        attention_scores = query @ keys.transpose(-2, -1)
        # trim the mask to the current sequence length and apply it to attention scores
        attention_scores.masked_fill_(self.mask.bool()[:seq_len, :seq_len], -torch.inf)

        attention_weights = torch.softmax(
            attention_scores / (self.head_dim**0.5), dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        context_vectors = attention_weights @ values
        # reshape back to (batch_size, seq_len, num_heads, head_dim)
        context_vectors = context_vectors.transpose(1, 2).contiguous()
        context_vectors = context_vectors.view(batch_size, seq_len, self.embedding_dim)
        output = self.out_proj(context_vectors)
        return output


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            x
            * 0.5
            * (
                1.0
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps  # prevent division by zero
        self.scale = nn.Parameter(torch.ones(embedding_dim))
        self.shift = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * normalized_x + self.shift


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attention = MultiHeadAttention(
            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            context_length=cfg.context_length,
            dropout=cfg.dropout,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(cfg.embedding_dim)
        self.ln1 = LayerNorm(cfg.embedding_dim)
        self.ln2 = LayerNorm(cfg.embedding_dim)
        self.dropout = nn.Dropout(p=cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply attention and feedforward with residual connections and layer normalization
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embedding_dim)
        self.pos_embedding = nn.Embedding(cfg.context_length, cfg.embedding_dim)
        self.dropout = nn.Dropout(cfg.dropout)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.num_layers)]
        )
        self.final_layer_norm = LayerNorm(cfg.embedding_dim)
        self.out_head = nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        token_embeddings = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=input_ids.device)
        pos_embeddings = self.pos_embedding(pos_ids)
        x = self.dropout(token_embeddings + pos_embeddings)

        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        # project to vocabulary size to get logits for each token position
        logits = self.out_head(x)
        return logits


def generate_text(
    model: GPTModel, input_ids: torch.Tensor, max_length: int, context_length: int
) -> torch.Tensor:
    for _ in range(max_length):
        # ensure the input to the model does not exceed the context length
        input_ids_cutted = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(input_ids_cutted)

        # get the logits for the last token position and sample the next token
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        # use greedy decoding to select the token with the highest probability
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        input_ids = torch.cat((input_ids, next_token_id), dim=1)

    return input_ids


def generate(
    model: GPTModel,
    input_ids: torch.Tensor,
    max_length: int,
    context_length: int,
    temperature: float = 0.0,
    top_k: int = 0,
):
    for _ in range(max_length):
        # ensure the input to the model does not exceed the context length
        input_ids_cutted = input_ids[:, -context_length:]
        with torch.no_grad():
            logits = model(input_ids_cutted)

        # get the logits for the last token position and sample the next token
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(next_token_logits, k=top_k)
            # create a new tensor filled with -inf
            next_token_logits = torch.full_like(next_token_logits, float("-inf"))
            # scatter the top-k logits back to their original positions
            next_token_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

        if temperature > 0.0:
            next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        input_ids = torch.cat((input_ids, next_token_id), dim=1)
    return input_ids
