import math

import torch
import torch.nn as nn
from gpt import GPTModel, generate_text
from tiktoken import Encoding
from torch.utils.data import DataLoader
from utils import text_to_tokens, tokens_to_text


def generate_text_and_print(
    model: GPTModel, tokenizer: Encoding, start_context: str, max_length: int = 50
) -> None:
    model.eval()
    context_length = model.pos_embedding.weight.shape[0]
    device = next(model.parameters()).device
    input_tokens = text_to_tokens(start_context, tokenizer).to(device)
    with torch.no_grad():
        generated_tokens = generate_text(
            model,
            input_tokens,
            max_length=max_length,
            context_length=context_length,
        )
    generated_text = tokens_to_text(generated_tokens, tokenizer)
    print(generated_text.replace("\n", " "))
    model.train()


def train_model(
    model: GPTModel,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer: Encoding,
    start_context: str,
) -> tuple[list[float], list[float], list[int]]:
    train_losses, valid_losses, track_tokens_seen = [], [], []
    step, tokens_seen = 0, 0

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            # set gradients to zero before backward pass to avoid accumulation
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model)
            # compute gradients of loss w.r.t. model parameters
            loss.backward()
            # update parameters using the computed gradients
            optimizer.step()
            step += 1
            tokens_seen += input_batch.numel()

            if step == 1 or step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(
                    model, train_dataloader, valid_dataloader, eval_iter
                )
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {step}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

        generate_text_and_print(model, tokenizer, start_context)
    return train_losses, valid_losses, track_tokens_seen


def train_model_2(
    model: GPTModel,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    eval_freq: int,
    eval_iter: int,
    tokenizer: Encoding,
    start_context: str,
    warmup_steps: int,
    lr_init: float = 3e-5,
    lr_min: float = 1e-6,
) -> tuple[list[float], list[float], list[int], list[float]]:
    """Train the model with a learning rate schedule that includes a linear warmup followed by cosine decay."""
    train_losses, valid_losses, track_tokens_seen, track_lrs = [], [], [], []
    step, tokens_seen = 0, 0

    lr_max = optimizer.param_groups[0]["lr"]
    total_training_steps = len(train_dataloader) * num_epochs
    lr_incr = (lr_max - lr_init) / warmup_steps

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_dataloader:
            # set gradients to zero before backward pass to avoid accumulation
            optimizer.zero_grad()

            # update learning rate according to the schedule
            if step < warmup_steps:
                lr = lr_init + step * lr_incr
            else:
                progress = (step - warmup_steps) / (total_training_steps - warmup_steps)
                lr = lr_min + 0.5 * (lr_max - lr_min) * (
                    1 + math.cos(math.pi * progress)
                )

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)

            loss = calc_loss_batch(input_batch, target_batch, model)
            # compute gradients of loss w.r.t. model parameters
            loss.backward()

            # clip gradients to avoid exploding gradients
            if step > warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # update parameters using the computed gradients
            optimizer.step()
            step += 1
            tokens_seen += input_batch.numel()

            if step == 1 or step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(
                    model, train_dataloader, valid_dataloader, eval_iter
                )
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                track_tokens_seen.append(tokens_seen)

                print(
                    f"Epoch {epoch + 1}/{num_epochs}, Step {step}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
                )

        generate_text_and_print(model, tokenizer, start_context)
    return train_losses, valid_losses, track_tokens_seen, track_lrs


def calc_loss_batch(
    input_batch: torch.Tensor, target_batch: torch.Tensor, model: GPTModel
) -> torch.Tensor:
    device = next(model.parameters()).device
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    logits = model(input_batch)
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    return loss


def calc_loss_dataloader(
    dataloader: DataLoader, model: GPTModel, num_batches: int | None = None
) -> float:
    total_loss = 0.0
    if len(dataloader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    for i, (input_batch, target_batch) in enumerate(dataloader):
        if i >= num_batches:
            break
        loss = calc_loss_batch(input_batch, target_batch, model)
        total_loss += loss.item()
    average_loss = total_loss / num_batches
    return average_loss


def evaluate_model(
    model: GPTModel,
    train_dataloader: DataLoader,
    valid_dataloader: DataLoader,
    eval_iter: int,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_dataloader(
            train_dataloader, model, num_batches=eval_iter
        )
        valid_loss = calc_loss_dataloader(
            valid_dataloader, model, num_batches=eval_iter
        )
    model.train()
    return train_loss, valid_loss
