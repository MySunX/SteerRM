import json
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_special_token_ids(tokenizer) -> Optional[torch.Tensor]:
    special_token_ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is None:
            continue
        if isinstance(token_id, list):
            special_token_ids.update(int(t) for t in token_id if t is not None)
        else:
            special_token_ids.add(int(token_id))
    additional_special_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
    if additional_special_ids:
        special_token_ids.update(int(t) for t in additional_special_ids if t is not None)
    return torch.tensor(sorted(special_token_ids), dtype=torch.long) if len(special_token_ids) > 0 else None


def load_wildchat_texts(num_samples: int = 500, seed: int = 42) -> List[str]:
    ds = load_dataset("allenai/WildChat", split="train")
    ds = ds.shuffle(seed=seed)
    texts: List[str] = []
    conversations_processed = 0
    for item in ds:
        if len(texts) >= num_samples:
            break
        conversation = item.get("conversation", [])
        if isinstance(conversation, list):
            for turn in conversation:
                if len(texts) >= num_samples:
                    break
                if isinstance(turn, dict):
                    content = turn.get("content", "")
                    if content and isinstance(content, str) and len(content.strip()) > 0:
                        texts.append(content.strip())
        conversations_processed += 1
    return texts


def _weighted_stats(values: List[float], weights: List[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    values_arr = np.array(values, dtype=np.float64)
    weights_arr = np.array(weights, dtype=np.float64)
    weight_sum = weights_arr.sum()
    if weight_sum <= 0:
        return 0.0, 0.0
    mean = float(np.average(values_arr, weights=weights_arr))
    if len(values_arr) == 1:
        return mean, 0.0
    variance = float(np.average((values_arr - mean) ** 2, weights=weights_arr))
    std = float(np.sqrt(max(variance, 0.0)))
    return mean, std


class BaseModelSAEEvaluator:
    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        sae_release: str,
        target_layers: List[int],
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.special_token_ids = get_special_token_ids(self.tokenizer)
        self.target_layers = target_layers
        self.sae_release = sae_release
        available_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device_pool = [f"cuda:{i}" for i in range(available_device_count)] if available_device_count > 0 else ["cpu"]
        self.layer_devices: Dict[int, str] = {}
        for i, layer_idx in enumerate(self.target_layers):
            self.layer_devices[layer_idx] = device_pool[i % len(device_pool)]
        self.sae_models: Dict[int, SAE] = {}
        if "lxr" in self.sae_release:
            suffix = "r"
        elif "lxm" in self.sae_release:
            suffix = "m"
        else:
            suffix = "m"
        for layer_idx in self.target_layers:
            sae_id = f"l{layer_idx}{suffix}_32x"
            sae_device = self.layer_devices[layer_idx]
            sae_model, _, _ = SAE.from_pretrained(release=self.sae_release, sae_id=sae_id, device=sae_device)
            sae_model.eval()
            self.sae_models[layer_idx] = sae_model

    def _compute_valid_mask(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        valid_mask = attention_mask.bool()
        if self.special_token_ids is not None and self.special_token_ids.numel() > 0:
            special_ids = self.special_token_ids.to(input_ids.device)
            special_mask = torch.isin(input_ids, special_ids)
            valid_mask = valid_mask & (~special_mask)
        return valid_mask

    @torch.no_grad()
    def evaluate_on_texts(self, texts: List[str], batch_size: int = 2, max_length: int = 4096) -> Dict[int, Dict[str, float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids_all = enc["input_ids"]
        attention_mask_all = enc["attention_mask"]
        layer_stats: Dict[int, Dict[str, List[float]]] = {
            layer_idx: {
                "recon_mse_values": [],
                "recon_mse_weights": [],
                "recon_l2_values": [],
                "recon_l2_weights": [],
                "l0_values": [],
                "l0_weights": [],
            }
            for layer_idx in self.sae_models.keys()
        }
        for layer_idx, sae in self.sae_models.items():
            for start in tqdm(range(0, input_ids_all.size(0), batch_size), desc=f"Layer {layer_idx}"):
                end = min(start + batch_size, input_ids_all.size(0))
                input_ids = input_ids_all[start:end].to(self.device)
                attention_mask = attention_mask_all[start:end].to(self.device)
                valid_token_mask = self._compute_valid_mask(input_ids, attention_mask)
                batch_metrics: Dict[str, Optional[float]] = {
                    "recon_mse": None,
                    "recon_l2": None,
                    "l0": None,
                    "token_count": float(valid_token_mask.sum().item()),
                }
                sae_param = next(sae.parameters())
                sae_device = sae_param.device
                sae_dtype = sae_param.dtype
                valid_mask_on_sae = valid_token_mask.to(sae_device)
                token_count_value = batch_metrics["token_count"]

                def sae_hook(module, inputs, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    sae_hidden = hidden.to(device=sae_device, dtype=sae_dtype)
                    feature_acts = sae.encode(sae_hidden)
                    recon = sae.decode(feature_acts).to(device=hidden.device, dtype=hidden.dtype)
                    if valid_token_mask is not None:
                        token_mask = valid_token_mask.to(hidden.device).unsqueeze(-1)
                        recon_masked = torch.where(token_mask, recon, hidden)
                    else:
                        recon_masked = recon
                    if token_count_value and token_count_value > 0:
                        mask_float_hidden = valid_token_mask.to(hidden.device).float()
                        mask_float_sae = valid_mask_on_sae.float()
                        diff = recon - hidden
                        per_token_mse = diff.pow(2).mean(dim=-1)
                        per_token_l2 = diff.pow(2).sum(dim=-1).sqrt()
                        mse_value = (per_token_mse * mask_float_hidden).sum() / token_count_value
                        l2_value = (per_token_l2 * mask_float_hidden).sum() / token_count_value
                        l0_per_token = (feature_acts > 0).float().sum(dim=-1)
                        l0_value = (l0_per_token * mask_float_sae).sum() / token_count_value
                        batch_metrics["recon_mse"] = mse_value.detach().cpu().item()
                        batch_metrics["recon_l2"] = l2_value.detach().cpu().item()
                        batch_metrics["l0"] = l0_value.detach().cpu().item()
                    if isinstance(output, tuple):
                        return (recon_masked,) + output[1:]
                    return recon_masked

                target_module = self.model.model.layers[layer_idx]
                handle = target_module.register_forward_hook(sae_hook)
                try:
                    _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
                finally:
                    handle.remove()
                stats = layer_stats[layer_idx]
                if batch_metrics["recon_mse"] is not None:
                    stats["recon_mse_values"].append(batch_metrics["recon_mse"])
                    stats["recon_mse_weights"].append(token_count_value)
                if batch_metrics["recon_l2"] is not None:
                    stats["recon_l2_values"].append(batch_metrics["recon_l2"])
                    stats["recon_l2_weights"].append(token_count_value)
                if batch_metrics["l0"] is not None:
                    stats["l0_values"].append(batch_metrics["l0"])
                    stats["l0_weights"].append(token_count_value)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        layer_results: Dict[int, Dict[str, float]] = {}
        for layer_idx, stats in layer_stats.items():
            recon_mse_mean, recon_mse_std = _weighted_stats(stats["recon_mse_values"], stats["recon_mse_weights"])
            recon_l2_mean, recon_l2_std = _weighted_stats(stats["recon_l2_values"], stats["recon_l2_weights"])
            l0_mean, l0_std = _weighted_stats(stats["l0_values"], stats["l0_weights"])
            layer_results[layer_idx] = {
                "reconstruction_mse_mean": recon_mse_mean,
                "reconstruction_mse_std": recon_mse_std,
                "reconstruction_l2_mean": recon_l2_mean,
                "reconstruction_l2_std": recon_l2_std,
                "l0_sparsity_mean": l0_mean,
                "l0_sparsity_std": l0_std,
            }
        return layer_results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--sae_release", type=str, required=True)
    parser.add_argument("--target_layers", type=str, default="0-31")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output", type=str, default="base_sae_metrics.json")
    args = parser.parse_args()
    
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path
    
    if "-" in args.target_layers:
        start, end = args.target_layers.split("-")
        target_layers = list(range(int(start), int(end) + 1))
    else:
        target_layers = [int(x) for x in args.target_layers.split(",")]
    
    texts = load_wildchat_texts(num_samples=args.num_samples)
    evaluator = BaseModelSAEEvaluator(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        sae_release=args.sae_release,
        target_layers=target_layers,
    )
    layer_results = evaluator.evaluate_on_texts(texts, batch_size=args.batch_size)
    output = {
        "model": args.model_path,
        "sae_release": args.sae_release,
        "texts_evaluated": len(texts),
        "results": layer_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

