import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import numpy as np
from sae_lens import SAE
from typing import List, Dict, Tuple, Optional, Literal
import json
import glob
from tqdm import tqdm
from dataclasses import dataclass


def find_transformer_layers(model) -> List[torch.nn.Module]:
    candidate_paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("model", "encoder", "layers"),
        ("model", "transformer", "layers"),
        ("transformer", "h"),
        ("backbone", "layers"),
        ("base_model", "model", "layers"),
        ("model", "model", "layers"),
    ]
    for path in candidate_paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if not ok:
            continue
        try:
            _ = len(node)
            return node
        except Exception:
            continue
    raise ValueError("Cannot locate Transformer layers in model structure")


@dataclass(frozen=True)
class FeatureSelectionConfig:
    method: Literal["composite", "topk", "quantile", "auto", "knee"] = "composite"
    select_scope: Literal["per_layer", "global"] = "per_layer"
    score_mode: Literal["weighted_sum", "product", "harmonic"] = "weighted_sum"
    mean_weight: float = 1.0
    stability_weight: float = 0.5
    score_min: float = 1.0
    norm_diff_mean_abs_min: float = 0.6
    variance_max: float = 0.02
    top_k: Optional[int] = None
    score_quantile: float = 0.999
    auto_top_fraction: float = 0.001
    max_features_per_layer: Optional[int] = None


class RMSAEAnalyzer:
    def __init__(
        self,
        rm_model_path: str,
        sae_release: str,
        tokenizer_path: str,
        device: str = "auto",
        target_layers: List[int] = None,
        model_type: str = "rm",
        batch_size: int = 1,
        max_length: int = 4096,
    ):
        self.device = self._resolve_device(device)
        self.model_type = model_type.lower()
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        if target_layers is None:
            self.target_layers = [16]
        else:
            self.target_layers = target_layers
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.model_type == "rm":
            self.rm_model = AutoModelForSequenceClassification.from_pretrained(
                rm_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        elif self.model_type == "base":
            self.rm_model = AutoModelForCausalLM.from_pretrained(
                rm_model_path,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}, expected 'rm' or 'base'")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if getattr(self.rm_model.config, "pad_token_id", None) is None:
            self.rm_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.special_token_ids = self._get_special_token_ids(self.tokenizer)
        self.rm_model.to(self.device)
        self.rm_model.eval()
        self.layers_module = find_transformer_layers(self.rm_model)
        self.sae_models = {}
        self.sae_device_map = self._prepare_sae_device_map()
        for layer_idx in self.target_layers:
            sae_id = f"l{layer_idx}r_32x"
            sae_device = self.sae_device_map.get(layer_idx, device)
            sae, sae_cfg, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=sae_device
            )
            self.sae_models[layer_idx] = sae
            self.sae_models[layer_idx].eval()

    def _resolve_device(self, device: str) -> str:
        device = (device or "").strip().lower()
        if device == "auto":
            if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                return "cpu"
            best_idx = 0
            best_free = -1
            for idx in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(idx)
                except Exception:
                    free, total = (0, 0)
                if free > best_free:
                    best_free = free
                    best_idx = idx
            return f"cuda:{best_idx}"
        if device in ("cuda", "cpu"):
            return device
        if device.startswith("cuda:"):
            return device
        return device
    
    def _get_special_token_ids(self, tokenizer) -> Optional[torch.Tensor]:
        special_ids = set()
        for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is None:
                continue
            if isinstance(token_id, list):
                special_ids.update(int(t) for t in token_id if t is not None)
            else:
                special_ids.add(int(token_id))
        additional_ids = getattr(tokenizer, "additional_special_tokens_ids", None)
        if additional_ids:
            special_ids.update(int(t) for t in additional_ids if t is not None)
        return (
            torch.tensor(sorted(special_ids), dtype=torch.long) if len(special_ids) > 0 else None
        )
    
    def _format_conversation(self, conv: List[Dict[str, str]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            formatted = self.tokenizer.apply_chat_template(conv, tokenize=False)
        else:
            formatted = ""
            for turn in conv:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    formatted += f"User: {content}\n"
                else:
                    formatted += f"Assistant: {content}\n"
            formatted = formatted.rstrip("\n")
        if self.tokenizer.bos_token and formatted.startswith(self.tokenizer.bos_token):
            formatted = formatted[len(self.tokenizer.bos_token):]
        return formatted
    
    def _prepare_sae_device_map(self) -> Dict[int, str]:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                ids = [v.strip() for v in visible.split(",") if v.strip() != ""]
                devices = [f"cuda:{i}" for i in range(len(ids))]
            else:
                devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
        device_map: Dict[int, str] = {}
        for idx, layer in enumerate(self.target_layers):
            device_map[layer] = devices[idx % len(devices)]
        return device_map
    
    def extract_rm_features(self, prompts: List[str], responses: List[str], batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        all_features = []
        all_scores = []
        if batch_size is None:
            batch_size = self.batch_size
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting RM features"):
            batch_prompts = prompts[i:i+batch_size]
            batch_responses = responses[i:i+batch_size]
            batch_conversations = []
            for prompt, response in zip(batch_prompts, batch_responses):
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                batch_conversations.append(conv)
            batch_formatted = []
            for conv in batch_conversations:
                formatted = self._format_conversation(conv)
                batch_formatted.append(formatted)
            inputs = self.tokenizer(
                batch_formatted,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            valid_token_mask = attention_mask.bool()
            if self.special_token_ids is not None:
                special_ids = self.special_token_ids.to(input_ids.device)
                special_mask = torch.isin(input_ids, special_ids)
                valid_token_mask = valid_token_mask & (~special_mask)
            with torch.no_grad():
                captured: Dict[int, torch.Tensor] = {}
                handles = []
                def _make_hook(layer_idx: int):
                    def hook(_module, _inputs, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        if isinstance(hidden, torch.Tensor):
                            captured[layer_idx] = hidden
                        return output
                    return hook
                for layer_idx in self.target_layers:
                    if layer_idx < 0 or layer_idx >= len(self.layers_module):
                        continue
                    handles.append(self.layers_module[layer_idx].register_forward_hook(_make_hook(layer_idx)))
                outputs = self.rm_model(**inputs, return_dict=True)
                if self.model_type == "rm":
                    scores = outputs.logits.squeeze(-1)
                else:
                    scores = torch.zeros(inputs["input_ids"].shape[0], device=self.device)
                for h in handles:
                    h.remove()
                layer_features = []
                for layer_idx in self.target_layers:
                    if layer_idx not in captured:
                        continue
                    layer_hidden = captured[layer_idx]
                    sae = self.sae_models[layer_idx]
                    sae_param = next(sae.parameters())
                    hidden_on_sae = layer_hidden.to(device=sae_param.device, dtype=sae_param.dtype)
                    feature_acts = sae.encode(hidden_on_sae)
                    mask_on_sae = valid_token_mask.to(sae_param.device)
                    mask_float = mask_on_sae.unsqueeze(-1).to(feature_acts.dtype)
                    token_counts = mask_float.sum(dim=1).clamp(min=1.0)
                    feature_sum = (feature_acts * mask_float).sum(dim=1)
                    features = feature_sum / token_counts
                    layer_features.append(features.to(self.device))
                if len(layer_features) == 0:
                    raise RuntimeError("Failed to extract any target layer features")
                combined_features = torch.cat(layer_features, dim=1)
                all_features.append(combined_features.cpu())
                all_scores.append(scores.cpu())
                del outputs, captured, handles, layer_features, combined_features
                torch.cuda.empty_cache()
        if len(all_features) > 1:
            standard_feature_dim = all_features[0].shape[1]
            last_batch = all_features[-1]
            if last_batch.shape[1] != standard_feature_dim:
                if last_batch.shape[1] > standard_feature_dim:
                    all_features[-1] = last_batch[:, :standard_feature_dim]
                else:
                    padding = torch.zeros(last_batch.shape[0], standard_feature_dim - last_batch.shape[1])
                    all_features[-1] = torch.cat([last_batch, padding], dim=1)
        all_features = [f.contiguous() for f in all_features]       
        return torch.cat(all_features, dim=0), torch.cat(all_scores, dim=0)
    
    def analyze_markdown_plain_dataset(
        self,
        dataset: List[Dict],
        selection: Optional[FeatureSelectionConfig] = None,
    ) -> Dict:
        all_markdown = []
        all_plain = []
        import re
        def _first_existing(d: Dict, keys: List[str]):
            for k in keys:
                if k in d and d[k] is not None:
                    return d[k]
            return None
        def _normalize_text(x) -> Optional[str]:
            if x is None:
                return None
            if isinstance(x, str):
                s = x.strip()
                return s if s != "" else None
            return None
        def _markdown_score(text: str) -> int:
            s = text
            score = 0
            if "```" in s:
                score += 4
            if re.search(r"^\s*#{1,6}\s+\S", s, flags=re.M):
                score += 3
            if re.search(r"\[[^\]]+\]\([^)]+\)", s):
                score += 2
            if re.search(r"\*\*[^*]{1,80}\*\*", s):
                score += 1
            if re.search(r"^\s*[-*+]\s+\S", s, flags=re.M):
                score += 1
            if re.search(r"^\s*\|.*\|\s*$", s, flags=re.M):
                score += 1
            if re.search(r"<br\s*/?>", s, flags=re.I):
                score += 1
            return score
        def _extract_prompt(item: Dict) -> Optional[str]:
            return _normalize_text(
                _first_existing(item, ["prompt", "question", "instruction", "query", "input"])
            )
        def _extract_md_plain_pair(item: Dict) -> Optional[Dict[str, str]]:
            prompt = _extract_prompt(item)
            if prompt is None:
                return None
            md = _normalize_text(
                _first_existing(
                    item,
                    [
                        "answer_markdown",
                        "response_markdown",
                        "output_markdown",
                        "markdown_answer",
                        "markdown",
                    ],
                )
            )
            pl = _normalize_text(
                _first_existing(
                    item,
                    [
                        "answer_plain",
                        "response_plain",
                        "output_plain",
                        "plain_answer",
                        "plain",
                        "text",
                    ],
                )
            )
            if md is not None and pl is not None:
                return {"prompt": prompt, "markdown": md, "plain": pl}
            candidates: List[str] = []
            for key in ["chosen", "rejected", "answers", "responses", "candidates"]:
                v = item.get(key, None)
                if isinstance(v, list):
                    for t in v:
                        nt = _normalize_text(t)
                        if nt is not None:
                            candidates.append(nt)
            if len(candidates) < 2:
                return None
            scored = [(t, _markdown_score(t)) for t in candidates]
            md_cands = [t for (t, sc) in scored if sc >= 2]
            pl_cands = [t for (t, sc) in scored if sc <= 0]
            if not md_cands:
                min_md_score = max(sc for _, sc in scored)
                md_cands = [t for (t, sc) in scored if sc == min_md_score]
            if not pl_cands:
                min_pl_score = min(sc for _, sc in scored)
                pl_cands = [t for (t, sc) in scored if sc == min_pl_score]
            md_best = max(md_cands, key=len) if md_cands else None
            pl_best = max(pl_cands, key=len) if pl_cands else None
            if md_best is None or pl_best is None:
                return None
            if md_best == pl_best:
                return None
            return {"prompt": prompt, "markdown": md_best, "plain": pl_best}
        skipped = 0
        used = 0
        for item in dataset:
            if not isinstance(item, dict):
                skipped += 1
                continue
            pair = _extract_md_plain_pair(item)
            if pair is None:
                skipped += 1
                continue
            used += 1
            all_markdown.append({"prompt": pair["prompt"], "answer": pair["markdown"], "type": "markdown"})
            all_plain.append({"prompt": pair["prompt"], "answer": pair["plain"], "type": "plain"})
        if len(all_markdown) == 0 or len(all_plain) == 0:
            raise ValueError(
                "No valid markdown/plain pairs extracted from dataset."
            )
        all_prompts = []
        all_responses = []
        text_types = []
        for item in all_plain:
            all_prompts.append(item['prompt'])
            all_responses.append(item['answer'])
            text_types.append('plain')
        for item in all_markdown:
            all_prompts.append(item['prompt'])
            all_responses.append(item['answer'])
            text_types.append('markdown')
        all_features, all_scores = self.extract_rm_features(all_prompts, all_responses)
        plain_indices = [i for i, t in enumerate(text_types) if t == 'plain']
        markdown_indices = [i for i, t in enumerate(text_types) if t == 'markdown']
        plain_features = all_features[plain_indices]
        plain_scores = all_scores[plain_indices]
        markdown_features = all_features[markdown_indices]
        markdown_scores = all_scores[markdown_indices]
        results = {}
        results['markdown_vs_plain_features'] = self._analyze_markdown_vs_long_features(
            markdown_features, markdown_scores,
            plain_features, plain_scores,
            selection=selection
        )
        return results
    
    def _analyze_markdown_vs_long_features(
        self,
        markdown_features,
        markdown_scores,
        long_features,
        long_scores,
        selection: Optional[FeatureSelectionConfig] = None,
    ) -> Dict:
        min_samples = min(markdown_features.shape[0], long_features.shape[0])
        markdown_features = markdown_features[:min_samples]
        long_features = long_features[:min_samples]
        if selection is None:
            selection = FeatureSelectionConfig()
        FEATURES_PER_LAYER = 128 * 1024
        layer_feature_diffs: Dict[int, Dict] = {}
        selected_global_indices: List[int] = []
        selected_mean_diffs: List[float] = []
        selected_stats: List[Dict] = []
        all_global_stats: List[Dict] = []
        if selection.select_scope == "global":
            raw_global_stats = []
            for layer_pos, layer_idx in enumerate(self.target_layers):
                start = layer_pos * FEATURES_PER_LAYER
                end = start + FEATURES_PER_LAYER
                if start >= markdown_features.shape[1]:
                    continue
                end = min(end, markdown_features.shape[1])
                md_layer = markdown_features[:, start:end]
                lg_layer = long_features[:, start:end]
                layer_diffs = self._calculate_feature_differences(md_layer, lg_layer, selection=selection, normalize=False)
                for stat in layer_diffs.get("feature_stats", []):
                    s = dict(stat)
                    local_id = int(s.get("feature_idx"))
                    s["layer"] = layer_idx
                    s["local_id"] = local_id
                    s["global_id"] = int(start + local_id)
                    raw_global_stats.append(s)
            all_global_stats = self._normalize_and_score_global(raw_global_stats, selection)
            for layer_idx in self.target_layers:
                layer_stats = [s for s in all_global_stats if s.get("layer") == layer_idx]
                layer_feature_diffs[layer_idx] = {
                    'feature_stats': layer_stats,
                    'total_features': len(layer_stats),
                    'total_pairs': min_samples,
                    'score_mode': selection.score_mode,
                    'score_max': all_global_stats[0].get('score_max') if all_global_stats else None,
                }
            global_filtered = self._filter_global_features(all_global_stats, selection=selection)
            selected_global_indices = global_filtered["feature_indices"]
            selected_mean_diffs = global_filtered["difference_values"]
            selected_stats = global_filtered.get("selected_feature_stats", [])
        else:
            for layer_pos, layer_idx in enumerate(self.target_layers):
                start = layer_pos * FEATURES_PER_LAYER
                end = start + FEATURES_PER_LAYER
                if start >= markdown_features.shape[1]:
                    continue
                end = min(end, markdown_features.shape[1])
                md_layer = markdown_features[:, start:end]
                lg_layer = long_features[:, start:end]
                layer_diffs = self._calculate_feature_differences(md_layer, lg_layer, selection=selection, normalize=True)
                for stat in layer_diffs.get("feature_stats", []):
                    s = dict(stat)
                    local_id = int(s.get("feature_idx"))
                    s["layer"] = layer_idx
                    s["local_id"] = local_id
                    s["global_id"] = int(start + local_id)
                    all_global_stats.append(s)
                filtered = self._filter_features(layer_diffs, selection=selection)
                layer_feature_diffs[layer_idx] = layer_diffs
                if filtered is not None:
                    for local_id, mean_diff, stat in zip(
                        filtered["feature_indices"],
                        filtered["difference_values"],
                        filtered.get("selected_feature_stats", []),
                    ):
                        global_id = start + local_id
                        selected_global_indices.append(global_id)
                        selected_mean_diffs.append(mean_diff)
                        stat = dict(stat)
                        stat["layer"] = layer_idx
                        stat["local_id"] = local_id
                        stat["global_id"] = global_id
                        selected_stats.append(stat)
        markdown_vs_long_diff = markdown_features.mean(0) - long_features.mean(0)
        results = {
            'feature_differences': layer_feature_diffs,
            'significant_features': {
                'feature_indices': selected_global_indices,
                'difference_values': selected_mean_diffs,
                'count': len(selected_global_indices),
                'selected_feature_stats': selected_stats,
            },
            'threshold_used': f"selection={selection}",
            'total_pairs': min_samples,
            'average_difference': markdown_vs_long_diff.cpu().tolist()
        }
        if results['significant_features']['count'] == 0:
            results['layer_specific_features'] = {
                'significant_features_by_layer': {layer: [] for layer in self.target_layers},
            }
            results['feature_distribution'] = {
                'significant_features_distribution': {layer: 0 for layer in self.target_layers},
            }
            return results
        significant_layer_features = self.convert_global_to_layer_ids(results['significant_features']['feature_indices'])
        significant_distribution = self.analyze_feature_distribution(results['significant_features']['feature_indices'], "Significant features")
        results['layer_specific_features'] = {
            'significant_features_by_layer': significant_layer_features,
        }
        results['feature_distribution'] = {
            'significant_features_distribution': significant_distribution,
        }
        return results
    
    def convert_global_to_layer_ids(self, global_ids: List[int], features_per_layer: int = 128*1024) -> Dict[int, List[int]]:
        layer_features = {layer: [] for layer in self.target_layers}
        for global_id in global_ids:
            layer_idx = global_id // features_per_layer
            layer_feature_id = global_id % features_per_layer
            if layer_idx < len(self.target_layers):
                actual_layer = self.target_layers[layer_idx]
                layer_features[actual_layer].append(layer_feature_id)
        return layer_features
    
    def analyze_feature_distribution(self, global_ids: List[int], dataset_name: str = "Features") -> Dict[int, int]:
        layer_counts = {layer: 0 for layer in self.target_layers}
        features_per_layer = 128 * 1024
        for global_id in global_ids:
            layer_idx = global_id // features_per_layer
            if layer_idx < len(self.target_layers):
                actual_layer = self.target_layers[layer_idx]
                layer_counts[actual_layer] += 1
        return layer_counts
    
    def _calculate_feature_differences(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        selection: Optional[FeatureSelectionConfig] = None,
        normalize: bool = True,
    ) -> Dict:
        if selection is None:
            selection = FeatureSelectionConfig()
        all_diffs = features1 - features2
        mean_diffs = all_diffs.mean(dim=0)
        var_diffs = all_diffs.var(dim=0, unbiased=False)
        if not normalize:
            feature_stats = []
            n_features = features1.shape[1]
            for feature_idx in range(n_features):
                feature_stats.append(
                    {
                        "feature_idx": feature_idx,
                        "mean_diff": mean_diffs[feature_idx].item(),
                        "variance": var_diffs[feature_idx].item(),
                    }
                )
            return {
                'feature_stats': feature_stats,
                'total_features': features1.shape[1],
                'total_pairs': features1.shape[0],
                'score_mode': None,
                'score_max': None,
            }
        eps = 1e-9
        mean_min, mean_max = mean_diffs.min(), mean_diffs.max()
        norm_mean = (mean_diffs - mean_min) / (mean_max - mean_min + eps)
        var_min, var_max = var_diffs.min(), var_diffs.max()
        norm_var = (var_diffs - var_min) / (var_max - var_min + eps)
        inv_norm_var = 1.0 - norm_var
        abs_norm_mean = norm_mean.abs()
        if selection.score_mode == "weighted_sum":
            score = abs_norm_mean * float(selection.mean_weight) + inv_norm_var * float(selection.stability_weight)
            score_max = float(selection.mean_weight) + float(selection.stability_weight)
        elif selection.score_mode == "product":
            score = abs_norm_mean * inv_norm_var
            score_max = 1.0
        elif selection.score_mode == "harmonic":
            score = (2.0 * abs_norm_mean * inv_norm_var) / (abs_norm_mean + inv_norm_var + eps)
            score_max = 1.0
        else:
            raise ValueError(f"Unsupported score_mode: {selection.score_mode}")
        feature_stats = []
        n_features = features1.shape[1]
        for feature_idx in tqdm(range(n_features), desc="Processing features"):
            feature_stats.append(
                {
                    "feature_idx": feature_idx,
                    "mean_diff": mean_diffs[feature_idx].item(),
                    "variance": var_diffs[feature_idx].item(),
                    "norm_diff_mean": norm_mean[feature_idx].item(),
                    "inv_norm_variance": inv_norm_var[feature_idx].item(),
                    "score": score[feature_idx].item(),
                }
            )
        feature_stats.sort(key=lambda x: x["score"], reverse=True)
        return {
            'feature_stats': feature_stats,
            'total_features': features1.shape[1],
            'total_pairs': features1.shape[0],
            'score_mode': selection.score_mode,
            'score_max': score_max,
        }
    
    def _normalize_and_score_global(self, raw_stats: List[Dict], selection: FeatureSelectionConfig) -> List[Dict]:
        if len(raw_stats) == 0:
            return []
        mean_diffs = np.array([float(s["mean_diff"]) for s in raw_stats], dtype=np.float32)
        variances = np.array([float(s["variance"]) for s in raw_stats], dtype=np.float32)
        eps = 1e-9
        mean_min, mean_max = mean_diffs.min(), mean_diffs.max()
        norm_mean = (mean_diffs - mean_min) / (mean_max - mean_min + eps)
        var_min, var_max = variances.min(), variances.max()
        norm_var = (variances - var_min) / (var_max - var_min + eps)
        inv_norm_var = 1.0 - norm_var
        abs_norm_mean = np.abs(norm_mean)
        if selection.score_mode == "weighted_sum":
            scores = abs_norm_mean * float(selection.mean_weight) + inv_norm_var * float(selection.stability_weight)
            score_max = float(selection.mean_weight) + float(selection.stability_weight)
        elif selection.score_mode == "product":
            scores = abs_norm_mean * inv_norm_var
            score_max = 1.0
        elif selection.score_mode == "harmonic":
            scores = (2.0 * abs_norm_mean * inv_norm_var) / (abs_norm_mean + inv_norm_var + eps)
            score_max = 1.0
        else:
            raise ValueError(f"Unsupported score_mode: {selection.score_mode}")
        result_stats = []
        for i, stat in enumerate(raw_stats):
            updated_stat = dict(stat)
            updated_stat["norm_diff_mean"] = float(norm_mean[i])
            updated_stat["inv_norm_variance"] = float(inv_norm_var[i])
            updated_stat["score"] = float(scores[i])
            updated_stat["score_max"] = score_max
            result_stats.append(updated_stat)
        result_stats.sort(key=lambda x: x["score"], reverse=True)
        return result_stats
    
    def _filter_features(self, feature_diffs: Dict, selection: FeatureSelectionConfig) -> Dict:
        feature_stats = feature_diffs["feature_stats"]
        selected: List[Dict] = []
        method = selection.method
        if method == "composite":
            for stats in feature_stats:
                if (
                    stats["score"] > selection.score_min
                    and abs(stats["norm_diff_mean"]) > selection.norm_diff_mean_abs_min
                    and stats["variance"] < selection.variance_max
                ):
                    selected.append(stats)
        elif method == "topk":
            if selection.top_k is None or selection.top_k <= 0:
                raise ValueError("selection.method='topk' requires selection.top_k to be a positive integer")
            selected = feature_stats[: min(selection.top_k, len(feature_stats))]
        elif method == "quantile":
            if not (0.0 < selection.score_quantile < 1.0):
                raise ValueError("selection.method='quantile' requires 0<score_quantile<1")
            scores = np.array([s["score"] for s in feature_stats], dtype=np.float32)
            thr = float(np.quantile(scores, selection.score_quantile))
            selected = [s for s in feature_stats if s["score"] >= thr]
            if selection.max_features_per_layer is not None and selection.max_features_per_layer > 0:
                selected = selected[:selection.max_features_per_layer]
        elif method == "auto":
            frac = float(selection.auto_top_fraction)
            if not (0.0 < frac < 1.0):
                raise ValueError("selection.method='auto' requires 0<auto_top_fraction<1")
            n = len(feature_stats)
            top_n = max(1, int(round(n * frac)))
            selected = feature_stats[:top_n]
            if selection.max_features_per_layer is not None and selection.max_features_per_layer > 0:
                selected = selected[:selection.max_features_per_layer]
        elif method == "knee":
            if len(feature_stats) == 0:
                selected = []
            else:
                stats_sorted = sorted(feature_stats, key=lambda s: s.get("score", 0.0), reverse=True)
                scores = np.array([float(s.get("score", 0.0)) for s in stats_sorted], dtype=np.float32)
                n = int(scores.shape[0])
                if n == 1:
                    knee_idx = 0
                else:
                    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    s_min = float(scores.min())
                    s_max = float(scores.max())
                    y = (scores - s_min) / (s_max - s_min + 1e-9)
                    y0 = float(y[0])
                    yN = float(y[-1])
                    y_line = y0 + (yN - y0) * x
                    dist = np.abs(y - y_line)
                    knee_idx = int(dist.argmax())
                selected = stats_sorted[: knee_idx + 1]
        else:
            raise ValueError(f"Unsupported selection.method: {method}")
        significant_indices = [s["feature_idx"] for s in selected]
        significant_values = [s["mean_diff"] for s in selected]
        return {
            'feature_indices': significant_indices,
            'difference_values': significant_values,
            'count': len(selected),
            'selected_feature_stats': selected,
        }

    def _filter_global_features(self, global_feature_stats: List[Dict], selection: FeatureSelectionConfig) -> Dict:
        if len(global_feature_stats) == 0:
            return {
                "feature_indices": [],
                "difference_values": [],
                "count": 0,
                "selected_feature_stats": [],
            }
        global_feature_stats = sorted(global_feature_stats, key=lambda x: x.get("score", 0.0), reverse=True)
        method = selection.method
        selected: List[Dict] = []
        if method == "topk":
            if selection.top_k is None or selection.top_k <= 0:
                raise ValueError("global+topk requires selection.top_k to be a positive integer")
            selected = global_feature_stats[: min(selection.top_k, len(global_feature_stats))]
        elif method == "knee":
            if len(global_feature_stats) == 0:
                selected = []
            else:
                stats_sorted = sorted(global_feature_stats, key=lambda x: x.get("score", 0.0), reverse=True)
                scores = np.array([float(s.get("score", 0.0)) for s in stats_sorted], dtype=np.float32)
                n = int(scores.shape[0])
                if n == 1:
                    knee_idx = 0
                else:
                    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    s_min = float(scores.min())
                    s_max = float(scores.max())
                    y = (scores - s_min) / (s_max - s_min + 1e-9)
                    y0 = float(y[0])
                    yN = float(y[-1])
                    y_line = y0 + (yN - y0) * x
                    dist = np.abs(y - y_line)
                    knee_idx = int(dist.argmax())
                selected = stats_sorted[: knee_idx + 1]
        elif method == "auto":
            frac = float(selection.auto_top_fraction)
            if not (0.0 < frac < 1.0):
                raise ValueError("global+auto requires 0<auto_top_fraction<1")
            n = len(global_feature_stats)
            top_n = max(1, int(round(n * frac)))
            selected = global_feature_stats[:top_n]
        elif method == "quantile":
            if not (0.0 < selection.score_quantile < 1.0):
                raise ValueError("global+quantile requires 0<score_quantile<1")
            scores = np.array([s.get("score", 0.0) for s in global_feature_stats], dtype=np.float32)
            thr = float(np.quantile(scores, selection.score_quantile))
            selected = [s for s in global_feature_stats if float(s.get("score", 0.0)) >= thr]
        elif method == "composite":
            for s in global_feature_stats:
                if (
                    float(s.get("score", 0.0)) > selection.score_min
                    and abs(float(s.get("norm_diff_mean", 0.0))) > selection.norm_diff_mean_abs_min
                    and float(s.get("variance", 0.0)) < selection.variance_max
                ):
                    selected.append(s)
        else:
            raise ValueError(f"Unsupported selection.method for global scope: {method}")
        feature_indices = [int(s.get("global_id")) for s in selected]
        difference_values = [float(s.get("mean_diff")) for s in selected]
        return {
            "feature_indices": feature_indices,
            "difference_values": difference_values,
            "count": len(selected),
            "selected_feature_stats": selected,
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze Markdown/Plain dataset with SAE features.")
    parser.add_argument("--dataset-dir", required=True, help="Directory containing JSON dataset files")
    parser.add_argument("--rm-model", required=True, help="Reward model path")
    parser.add_argument("--model-type", choices=["rm", "base"], default="rm")
    parser.add_argument("--sae-release", default="llama_scope_lxr_32x", help="SAE release identifier")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path")
    parser.add_argument("--layers", type=str, default="0-9", help="Target layers, e.g., '0-9' or '0,2,5'")
    parser.add_argument("--rm-device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", default="feature_analysis_results.json")
    parser.add_argument("--select-method", choices=["composite", "topk", "quantile", "auto", "knee"], default="composite")
    parser.add_argument("--select-scope", choices=["per_layer", "global"], default="per_layer")
    parser.add_argument("--score-mode", choices=["weighted_sum", "product", "harmonic"], default="weighted_sum")
    parser.add_argument("--mean-weight", type=float, default=1.0)
    parser.add_argument("--stability-weight", type=float, default=0.5)
    parser.add_argument("--score-min", type=float, default=1.0)
    parser.add_argument("--norm-abs-min", type=float, default=0.6)
    parser.add_argument("--var-max", type=float, default=0.02)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--score-quantile", type=float, default=0.999)
    parser.add_argument("--auto-top-fraction", type=float, default=0.001)
    parser.add_argument("--max-features-per-layer", type=int, default=None)
    args = parser.parse_args()

    if "-" in args.layers:
        start, end = args.layers.split("-")
        target_layers = list(range(int(start), int(end) + 1))
    else:
        target_layers = [int(idx) for idx in args.layers.split(",") if idx.strip() != ""]

    if args.tokenizer is None:
        args.tokenizer = args.rm_model

    selection = FeatureSelectionConfig(
        method=args.select_method,
        select_scope=args.select_scope,
        score_mode=args.score_mode,
        mean_weight=args.mean_weight,
        stability_weight=args.stability_weight,
        score_min=args.score_min,
        norm_diff_mean_abs_min=args.norm_abs_min,
        variance_max=args.var_max,
        top_k=args.top_k,
        score_quantile=args.score_quantile,
        auto_top_fraction=args.auto_top_fraction,
        max_features_per_layer=args.max_features_per_layer,
    )
    
    analyzer = RMSAEAnalyzer(
        args.rm_model,
        args.sae_release,
        args.tokenizer,
        device=args.rm_device,
        target_layers=target_layers,
        model_type=args.model_type,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    
    dataset_dir = os.path.abspath(args.dataset_dir)
    if not os.path.isdir(dataset_dir):
        raise ValueError(f"--dataset-dir is not a directory: {dataset_dir}")

    json_files = sorted(glob.glob(os.path.join(dataset_dir, "*.json")))
    if len(json_files) == 0:
        raise ValueError(f"No *.json files found in directory: {dataset_dir}")

    dataset: List[Dict] = []
    for fp in json_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Dataset file must be a JSON list, got {type(data)}: {fp}")
        dataset.extend(data)
    
    results = analyzer.analyze_markdown_plain_dataset(dataset, selection=selection)
    
    sig_feat = None
    if isinstance(results, dict):
        md_plain = results.get('markdown_vs_plain_features', {})
        if isinstance(md_plain, dict):
            sig_feat = md_plain.get('significant_features', None)

    output_path = args.output
    with open(output_path, 'w', encoding='utf-8') as f:
        if sig_feat is None:
            json.dump({}, f, indent=2, ensure_ascii=False)
        else:
            json.dump(sig_feat, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

