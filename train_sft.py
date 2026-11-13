"""Turn-masked LoRA Supervised Fine-Tuning (SFT) entrypoint for WSP-Gym."""

from __future__ import annotations

import argparse
import json
import re
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ChatWordTokenizer:
    """Deterministic zero-dependency tokenizer supporting ChatML markers for CPU dry-runs."""

    _TOKEN_RE = re.compile(r"<\|im_start\|>|<\|im_end\|>|\s+|[A-Za-z_][A-Za-z0-9_]*|[0-9]+|.")

    def __init__(self, vocab_size: int = 1024) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        self.vocab_size = int(vocab_size)
        self.pad_token_id = 0
        self.eos_token_id = 1
        self._special = {
            "<|pad|>": 0,
            "<|eos|>": 1,
            "<|im_start|>": 2,
            "<|im_end|>": 3,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids: list[int] = []
        bucket_span = self.vocab_size - 4
        for tok in self._TOKEN_RE.findall(text):
            if tok in self._special:
                ids.append(self._special[tok])
            else:
                h = zlib.crc32(tok.encode("utf-8")) % bucket_span
                ids.append(4 + int(h))
        return ids

    def decode(self, ids: list[int] | torch.Tensor, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, torch.Tensor):
            ids = [int(x) for x in ids.tolist()]
        out: list[str] = []
        for idx in ids:
            if idx == self.pad_token_id:
                if not skip_special_tokens:
                    out.append("<|pad|>")
            elif idx == self.eos_token_id:
                if not skip_special_tokens:
                    out.append("<|eos|>")
            elif idx == 2:
                if not skip_special_tokens:
                    out.append("<|im_start|>")
            elif idx == 3:
                if not skip_special_tokens:
                    out.append("<|im_end|>")
            else:
                out.append(f"tok_{idx}")
        return " ".join(out)


class TinyCausalLM(nn.Module):
    """Lightweight causal Transformer LM used for CPU verification of SFT and GRPO."""

    def __init__(
        self,
        vocab_size: int = 1024,
        d_model: int = 64,
        n_heads: int = 4,
        max_seq_len: int = 4096,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.max_seq_len = int(max_seq_len)
        self.tok_emb = nn.Embedding(self.vocab_size, d_model)
        self.pos_emb = nn.Embedding(self.max_seq_len, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
        )
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        bsz, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}")
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        h_in = self.norm1(x)
        attn_out, _ = self.attn(
            h_in,
            h_in,
            h_in,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        logits = self.lm_head(self.norm_f(x))

        loss = None
        if labels is not None and seq_len > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def tokenize_and_mask_chat(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int = 4096,
) -> dict[str, Any]:
    """Tokenize multi-turn chat messages and mask non-assistant tokens with -100."""
    input_ids: list[int] = []
    labels: list[int] = []

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        train_on_turn = bool(msg.get("loss_mask", role == "assistant"))

        prefix_ids = tokenizer.encode(f"<|im_start|>{role}\n", add_special_tokens=False)
        body_ids = tokenizer.encode(f"{content}<|im_end|>\n", add_special_tokens=False)

        input_ids.extend(prefix_ids)
        input_ids.extend(body_ids)
        labels.extend([-100] * len(prefix_ids))
        if train_on_turn:
            labels.extend(body_ids)
        else:
            labels.extend([-100] * len(body_ids))

    input_ids = input_ids[: int(max_length)]
    labels = labels[: int(max_length)]
    attention_mask = [1] * len(input_ids)

    assistant_token_count = sum(1 for tok in labels if tok != -100)
    masked_token_count = sum(1 for tok in labels if tok == -100)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "assistant_token_count": int(assistant_token_count),
        "masked_token_count": int(masked_token_count),
    }


def _collate_sft_batch(
    features: list[dict[str, Any]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    max_len = max(int(f["input_ids"].numel()) for f in features)
    bsz = len(features)
    input_ids = torch.full((bsz, max_len), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, feat in enumerate(features):
        n = int(feat["input_ids"].numel())
        input_ids[i, :n] = feat["input_ids"]
        attention_mask[i, :n] = feat["attention_mask"]
        labels[i, :n] = feat["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class MaskedSFTDataset(Dataset):
    """Dataset of tokenized and turn-masked multi-turn SFT episodes."""

    def __init__(self, features: list[dict[str, Any]]) -> None:
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.features[idx]


def _load_sft_records(sft_path: str | Path) -> list[dict[str, Any]]:
    src = Path(sft_path)
    if not src.exists():
        raise FileNotFoundError(f"SFT dataset not found: {src}")
    records = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"SFT dataset is empty: {src}")
    return records


def _grad_l2_norm(model: nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total += float(param.grad.detach().pow(2).sum().item())
    return total**0.5


def configure_model_for_lora_training(model: nn.Module, gradient_checkpointing: bool = True) -> None:
    """Prepare a base CausalLM for memory-safe LoRA training with optional gradient checkpointing."""
    if gradient_checkpointing:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()


def run_sft_dry_run(
    sft_path: str | Path = "trajectories/sft_train.jsonl",
    max_length: int = 4096,
    batch_size: int = 4,
    grad_accum: int = 8,
    gradient_checkpointing: bool = True,
) -> dict[str, Any]:
    """Validate all SFT records, verify assistant-only token masking, and run 2 CPU steps."""
    torch.manual_seed(0)
    records = _load_sft_records(sft_path)
    tokenizer = ChatWordTokenizer(vocab_size=1024)

    tokenized: list[dict[str, Any]] = []
    total_tokens_list: list[int] = []
    asst_tokens_list: list[int] = []
    for rec in records:
        feat = tokenize_and_mask_chat(rec["messages"], tokenizer, max_length=max_length)
        if feat["assistant_token_count"] <= 0:
            raise ValueError(f"Record {rec.get('episode_id')} has zero active assistant tokens.")
        if feat["masked_token_count"] <= 0:
            raise ValueError(f"Record {rec.get('episode_id')} has zero masked prompt tokens.")
        tokenized.append(feat)
        total_tokens_list.append(int(feat["input_ids"].numel()))
        asst_tokens_list.append(int(feat["assistant_token_count"]))

    model = TinyCausalLM(vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, max_seq_len=max_length)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    step_losses: list[float] = []
    grad_norm = 0.0
    for step_idx in range(2):
        start = (step_idx * batch_size) % len(tokenized)
        slice_feats = [tokenized[(start + k) % len(tokenized)] for k in range(min(batch_size, len(tokenized)))]
        batch = _collate_sft_batch(slice_feats, pad_token_id=tokenizer.pad_token_id)

        optimizer.zero_grad(set_to_none=True)
        out = model(**batch)
        loss = out.loss
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Invalid SFT dry-run loss at step {step_idx}: {loss}")
        loss.backward()
        grad_norm = _grad_l2_norm(model)
        if grad_norm <= 0.0:
            raise RuntimeError(f"Zero gradient norm at SFT dry-run step {step_idx}")
        optimizer.step()
        step_losses.append(round(float(loss.item()), 6))

    sum_total = sum(total_tokens_list)
    sum_asst = sum(asst_tokens_list)
    return {
        "num_records": len(tokenized),
        "mean_total_tokens": round(sum_total / len(tokenized), 2),
        "mean_assistant_tokens": round(sum_asst / len(tokenized), 2),
        "assistant_token_fraction": round(sum_asst / max(1, sum_total), 6),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "effective_batch_size": int(batch_size) * max(1, int(grad_accum)),
        "initial_loss": step_losses[0],
        "step2_loss": step_losses[-1],
        "grad_norm": round(grad_norm, 6),
    }


def run_sft_cloud_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run turn-masked LoRA SFT training on HuggingFace CausalLM + PEFT."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    records = _load_sft_records(args.sft_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    features = [
        tokenize_and_mask_chat(rec["messages"], tokenizer, max_length=args.max_length)
        for rec in records
    ]
    dataset = MaskedSFTDataset(features)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        collate_fn=lambda b: _collate_sft_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    use_cuda = torch.cuda.is_available()
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
    }
    if use_cuda:
        model_kwargs["device_map"] = "auto"

    gradient_checkpointing = not bool(getattr(args, "no_gradient_checkpointing", False))
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    configure_model_for_lora_training(base_model, gradient_checkpointing=gradient_checkpointing)
    lora_cfg = LoraConfig(
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.train()
    device = next(model.parameters()).device

    grad_accum = max(1, int(args.grad_accum))
    total_optim_steps = max(1, (len(loader) * int(args.epochs) + grad_accum - 1) // grad_accum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_optim_steps // 10),
        num_training_steps=total_optim_steps,
    )

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    loss_history: list[float] = []
    micro_step = 0

    for _epoch in range(int(args.epochs)):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            raw_loss = out.loss
            (raw_loss / grad_accum).backward()
            loss_history.append(float(raw_loss.detach().item()))
            micro_step += 1

            if micro_step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

    if micro_step % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    return {
        "num_records": len(features),
        "epochs": int(args.epochs),
        "optimizer_steps": global_step,
        "gradient_checkpointing": gradient_checkpointing,
        "effective_batch_size": int(args.batch_size) * grad_accum,
        "mean_loss": round(sum(loss_history) / max(1, len(loss_history)), 6),
        "final_loss": round(loss_history[-1], 6) if loss_history else 0.0,
        "output_dir": str(out_dir),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Turn-masked LoRA SFT warmup for WSP-Gym.")
    parser.add_argument("--sft-path", default="trajectories/sft_train.jsonl")
    parser.add_argument("--output-dir", default="checkpoints/sft_lora")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        summary = run_sft_dry_run(
            sft_path=args.sft_path,
            max_length=args.max_length,
            batch_size=max(2, args.batch_size),
            grad_accum=args.grad_accum,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        )
    else:
        summary = run_sft_cloud_training(args)
    print(json.dumps(summary, indent=2))
