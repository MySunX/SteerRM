import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Any
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from sae_lens import SAE
from fastchat.conversation import get_conv_template
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RM_BENCH_SCRIPT_DIR = os.path.join(SCRIPT_DIR, "..", "RM-Bench", "scripts")
if RM_BENCH_SCRIPT_DIR not in sys.path:
    sys.path.append(RM_BENCH_SCRIPT_DIR)

try:
    from rewardbench import (
        REWARD_MODEL_CONFIG,
        check_tokenizer_chat_template,
        torch_dtype_mapping,
    )
    from utils import (
        compute_accuracy,
        convert_robust_dataset_to_preference_dataset_list,
        load_eval_dataset,
    )
except ImportError:
    REWARD_MODEL_CONFIG = None
    check_tokenizer_chat_template = None
    torch_dtype_mapping = None
    compute_accuracy = None
    convert_robust_dataset_to_preference_dataset_list = None
    load_eval_dataset = None


def get_special_token_ids(tokenizer) -> Optional[torch.Tensor]:
    token_ids = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, list):
            token_ids.update(int(v) for v in value if v is not None)
        else:
            token_ids.add(int(value))
    additional = getattr(tokenizer, "additional_special_tokens_ids", None)
    if additional:
        token_ids.update(int(v) for v in additional if v is not None)
    return torch.tensor(sorted(token_ids), dtype=torch.long) if token_ids else None


def find_transformer_layers(model) -> Sequence[torch.nn.Module]:
    logger = logging.getLogger(__name__)
    candidate_paths = [
        ("model", "layers"),
        ("model", "decoder", "layers"),
        ("model", "encoder", "layers"),
        ("model", "transformer", "layers"),
        ("transformer", "h"),
        ("backbone", "layers"),
    ]
    for path in candidate_paths:
        node = model
        ok = True
        for attr in path:
            if not hasattr(node, attr):
                ok = False
                break
            node = getattr(node, attr)
        if ok:
            try:
                logger.info(
                    f"Matched Transformer layer path {'/'.join(path)}, type={type(node)}, layer_count={len(node)}"
                )
            except Exception:
                logger.info(f"Matched Transformer layer path {'/'.join(path)}, type={type(node)}")
            return node
    raise ValueError("Cannot locate Transformer layer list in reward model structure")


def load_selected_features(path: str) -> Dict[int, List[int]]:
    def _finalize(layer_to_feats: Dict[int, List[int]]) -> Dict[int, List[int]]:
        cleaned: Dict[int, List[int]] = {}
        for layer_idx, feats in layer_to_feats.items():
            uniq = sorted({int(x) for x in feats if x is not None})
            if uniq:
                cleaned[int(layer_idx)] = uniq
        return cleaned

    def _parse_old_selected_features(obj: Dict[str, Any]) -> Optional[Dict[int, List[int]]]:
        selected = obj.get("selected_features")
        if selected is None or not isinstance(selected, dict):
            return None
        layer_to_features: Dict[int, List[int]] = {}
        for layer, items in selected.items():
            if not items:
                continue
            indices: List[int] = []
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "feature_id" in item:
                        indices.append(int(item["feature_id"]))
                    elif isinstance(item, (int, float)):
                        indices.append(int(item))
            if indices:
                layer_to_features[int(layer)] = indices
        return _finalize(layer_to_features)

    def _parse_new_significant_features(obj: Dict[str, Any]) -> Optional[Dict[int, List[int]]]:
        sig = obj.get("significant_features")
        if sig is None:
            sig = obj if "selected_feature_stats" in obj else None
        if sig is None or not isinstance(sig, dict):
            return None
        stats = sig.get("selected_feature_stats")
        if stats is None or not isinstance(stats, list):
            return None
        layer_to_features: Dict[int, List[int]] = {}
        for item in stats:
            if not isinstance(item, dict):
                continue
            if "layer" not in item:
                continue
            try:
                layer_idx = int(item["layer"])
            except Exception:
                continue
            feat = None
            for k in ("local_id", "feature_idx", "feature_id"):
                if k in item and item[k] is not None:
                    feat = item[k]
                    break
            if feat is None and "global_id" in item and item["global_id"] is not None:
                feat = item["global_id"]
            if feat is None:
                continue
            layer_to_features.setdefault(layer_idx, []).append(int(feat))
        return _finalize(layer_to_features)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        parsed = _parse_old_selected_features(data) or _parse_new_significant_features(data)
        if not parsed:
            raise ValueError(
                f"File {path} cannot be parsed: neither old format selected_features "
                f"nor new format significant_features.selected_feature_stats found."
            )
        return parsed

    if isinstance(data, list):
        best: Optional[Dict[int, List[int]]] = None
        best_count = -1
        best_idx = -1
        for idx, elem in enumerate(data):
            if not isinstance(elem, dict):
                continue
            parsed = _parse_old_selected_features(elem) or _parse_new_significant_features(elem)
            if not parsed:
                continue
            count = sum(len(v) for v in parsed.values())
            if count > best_count:
                best, best_count, best_idx = parsed, count, idx
        if best is None:
            raise ValueError(
                f"File {path} is a list, but no element contains old format selected_features or new format significant_features."
            )
        return best

    raise ValueError(f"File {path} JSON top-level type {type(data)} not supported (expected dict or list)")


class SAEFeatureSuppressor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        selected_features: Dict[int, List[int]],
        sae_release: str,
        sae_device: Optional[torch.device],
        sae_id_template: str,
        sae_id_suffix: Optional[str],
        mask_special_tokens: bool,
        suppression_mode: str,
        suppression_strength: float,
        logger: logging.Logger,
        layer_to_device: Optional[Dict[int, torch.device]] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.selected_features = selected_features
        self.sae_release = sae_release
        self.sae_device = sae_device
        self.sae_id_template = sae_id_template
        self.sae_id_suffix = sae_id_suffix
        self.mask_special_tokens = mask_special_tokens
        self.suppression_mode = suppression_mode
        self.suppression_strength = suppression_strength
        self.logger = logger
        self.layer_to_device = layer_to_device
        self.layers_module = find_transformer_layers(model)
        self.layer_count = len(self.layers_module)
        self.layers = sorted(selected_features.keys())
        for layer_idx in self.layers:
            if layer_idx >= self.layer_count:
                raise ValueError(
                    f"Feature list contains layer {layer_idx}, but model only has {self.layer_count} layers"
                )
        self.special_token_ids = (
            get_special_token_ids(tokenizer) if mask_special_tokens else None
        )
        self.current_valid_mask: Optional[torch.Tensor] = None
        self.sae_models: Dict[int, SAE] = {}
        self.feature_indices: Dict[int, torch.Tensor] = {}
        self.dec_vecs: Dict[int, torch.Tensor] = {}
        self._load_saes()
        self.handles = self._register_hooks()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def set_batch_context(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if not self.mask_special_tokens:
            self.current_valid_mask = attention_mask.bool()
            return
        valid_mask = attention_mask.bool()
        if self.special_token_ids is not None and self.special_token_ids.numel() > 0:
            special_ids = self.special_token_ids.to(input_ids.device)
            special_mask = torch.isin(input_ids, special_ids)
            valid_mask = valid_mask & (~special_mask)
        self.current_valid_mask = valid_mask

    def clear_batch_context(self):
        self.current_valid_mask = None

    def _infer_suffix(self) -> str:
        if self.sae_id_suffix is not None:
            return self.sae_id_suffix
        release_lower = self.sae_release.lower()
        if "lxr" in release_lower:
            return "r"
        if "lxm" in release_lower:
            return "m"
        return ""

    def _sae_id_for_layer(self, layer_idx: int) -> str:
        suffix = self._infer_suffix()
        try:
            return self.sae_id_template.format(layer=layer_idx, suffix=suffix)
        except KeyError as exc:
            raise ValueError(
                f"SAE ID template {self.sae_id_template} cannot fill layer/suffix parameters"
            ) from exc

    def _load_saes(self):
        for layer_idx in self.layers:
            sae_id = self._sae_id_for_layer(layer_idx)
            if self.layer_to_device and layer_idx in self.layer_to_device:
                layer_device = self.layer_to_device[layer_idx]
            elif self.sae_device is not None:
                layer_device = self.sae_device
            else:
                raise ValueError(f"Layer {layer_idx} has no device specified")
            sae_model, _, _ = SAE.from_pretrained(
                release=self.sae_release,
                sae_id=sae_id,
                device=str(layer_device),
            )
            sae_model.eval()
            self.sae_models[layer_idx] = sae_model
            indices = torch.tensor(
                self.selected_features[layer_idx],
                dtype=torch.long,
                device=layer_device,
            )
            self.feature_indices[layer_idx] = indices
            if self.suppression_mode == "decvec":
                if hasattr(sae_model, "W_dec") and hasattr(sae_model.W_dec, "weight"):
                    w_dec = sae_model.W_dec.weight
                else:
                    w_dec = sae_model.W_dec
                if w_dec.ndim != 2:
                    raise ValueError(f"SAE decoder matrix shape abnormal: {tuple(w_dec.shape)}")
                dec_vec = w_dec[indices].sum(dim=0) if indices.numel() > 0 else torch.zeros_like(w_dec[0])
                dec_vec = dec_vec.to(device=layer_device, dtype=next(sae_model.parameters()).dtype)
                self.dec_vecs[layer_idx] = dec_vec

    def _register_hooks(self):
        handles = []
        for layer_idx in self.layers:
            module = self.layers_module[layer_idx]
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            handles.append(handle)
        self.logger.info(
            f"Registered SAE suppression hooks on {len(handles)} Transformer layers: {self.layers}"
        )
        return handles

    def _make_hook(self, layer_idx: int):
        sae_model = self.sae_models[layer_idx]
        indices = self.feature_indices[layer_idx]
        dec_vec = self.dec_vecs.get(layer_idx, None)

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output

            if self.suppression_mode == "decvec":
                if dec_vec is None:
                    return output
                delta = -self.suppression_strength * dec_vec.to(device=hidden.device, dtype=hidden.dtype)
                suppressed = hidden + delta
            elif self.suppression_mode == "recon_zero":
                sae_param = next(sae_model.parameters())
                hidden_on_sae = hidden.to(device=sae_param.device, dtype=sae_param.dtype)
                z_full = sae_model.encode(hidden_on_sae)
                if indices.numel() > 0:
                    z_full[..., indices.to(z_full.device)] = 0.0
                recon_supp = sae_model.decode(z_full)
                recon_supp = recon_supp.to(device=hidden.device, dtype=hidden.dtype)
                if self.suppression_strength == 1:
                    suppressed = recon_supp
                elif self.suppression_strength == 0:
                    suppressed = hidden
                else:
                    suppressed = hidden + self.suppression_strength * (recon_supp - hidden)
            else:
                sae_param = next(sae_model.parameters())
                hidden_on_sae = hidden.to(device=sae_param.device, dtype=sae_param.dtype)
                z_full = sae_model.encode(hidden_on_sae)
                recon_full = sae_model.decode(z_full)
                if indices.numel() > 0:
                    z_supp = z_full.clone()
                    z_supp[..., indices.to(z_full.device)] = 0.0
                    recon_supp = sae_model.decode(z_supp)
                else:
                    recon_supp = recon_full
                delta = (recon_supp - recon_full).to(device=hidden.device, dtype=hidden.dtype)
                suppressed = hidden + delta

            if self.current_valid_mask is not None:
                mask = self.current_valid_mask.to(hidden.device).unsqueeze(-1)
                suppressed = torch.where(mask, suppressed, hidden)

            if isinstance(output, tuple):
                return (suppressed,) + output[1:]
            return suppressed

        return hook


def score_texts(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
    suppressor: Optional[SAEFeatureSuppressor] = None,
    desc: str = "Processing batches",
) -> List[float]:
    scores: List[float] = []
    if len(texts) == 0:
        return scores
    with tqdm(total=len(texts), desc=desc, unit="samples", ncols=100) as pbar:
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            if suppressor is not None:
                suppressor.set_batch_context(input_ids, attention_mask)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if suppressor is not None:
                suppressor.clear_batch_context()

            logits = outputs.logits
            if logits.dim() == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)
            elif logits.dim() == 2:
                logits = logits[:, 0]
            else:
                raise ValueError(
                    f"Cannot parse model output logits shape {tuple(logits.shape)}, only single-dim classification score supported."
                )
            batch_scores = logits.detach().cpu().tolist()
            scores.extend(batch_scores)
            pbar.update(len(batch_texts))
    return scores


def evaluate_dataset(
    model,
    tokenizer,
    device,
    dataset,
    batch_size: int,
    max_length: int,
    suppressor: Optional[SAEFeatureSuppressor],
    desc_prefix: str = "Inference",
) -> Dict[str, List[float]]:
    chosen_texts = dataset["text_chosen"]
    rejected_texts = dataset["text_rejected"]
    chosen_scores = score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=chosen_texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        suppressor=suppressor,
        desc=f"{desc_prefix} - chosen",
    )
    rejected_scores = score_texts(
        model=model,
        tokenizer=tokenizer,
        texts=rejected_texts,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        suppressor=suppressor,
        desc=f"{desc_prefix} - rejected",
    )
    if len(chosen_scores) != len(rejected_scores):
        raise ValueError("chosen/rejected sample count mismatch, cannot compare scores.")
    return {"chosen": chosen_scores, "rejected": rejected_scores}


def safe_compute_accuracy(results: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    if not results or len(results) == 0:
        return None
    if 'domain' in results[0]:
        from utils import split_dataset_by_domain
        split_results = split_dataset_by_domain(results)
        split_results = {domain: domain_results for domain, domain_results in split_results.items() 
                        if domain_results and len(domain_results) > 0}
        if not split_results:
            return None
        domain_results = {}
        for domain in split_results:
            domain_acc = safe_compute_accuracy(split_results[domain])
            if domain_acc is not None:
                domain_results[domain] = domain_acc
        if not domain_results:
            return None
        domain_avg_results = {}
        for domain in domain_results:
            domain_avg_results[domain] = np.mean(list(domain_results[domain].values()))
        domain_hard_normal_easy_acc = {
            "hard_acc": np.mean([domain_results[domain]["hard_acc"] for domain in domain_results]),
            "normal_acc": np.mean([domain_results[domain]["normal_acc"] for domain in domain_results]),
            "easy_acc": np.mean([domain_results[domain]["easy_acc"] for domain in domain_results])
        }
        total_avg_acc = np.mean([domain_avg_results[domain] for domain in domain_avg_results])
        final_results = {}
        final_results.update(domain_avg_results)
        final_results.update(domain_hard_normal_easy_acc)
        final_results.update({"total_avg_acc": total_avg_acc})
        return final_results
    else:
        try:
            return compute_accuracy(results)
        except (IndexError, KeyError) as e:
            return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate on RM-Bench with SAE feature suppression."
    )
    parser.add_argument("--model", type=str, required=True, help="Reward model name or weight path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional: specify tokenizer separately")
    parser.add_argument("--chat_template", type=str, default="tulu", help="FastChat conversation template name")
    parser.add_argument("--datapath", type=str, required=True, help="RM-Bench data path (JSON file)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Tokenizer max length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g., cuda:0")
    parser.add_argument("--gpus", type=str, required=True, help="Available GPU list, comma-separated, e.g., '0,2,4,5'")
    parser.add_argument("--sae_release", type=str, default=None, help="SAE release name")
    parser.add_argument("--sae_id_template", type=str, default="l{layer}{suffix}_32x", help="SAE ID template")
    parser.add_argument("--sae_id_suffix", type=str, default=None, help="Override suffix in template")
    parser.add_argument("--suppression_mode", type=str, default="delta", choices=["delta", "decvec", "recon_zero"])
    parser.add_argument("--suppression_strength", type=float, default=1, help="Suppression strength coefficient")
    parser.add_argument("--selected_features_path", type=str, default=None, help="Feature suppression list JSON")
    parser.add_argument("--mask_special_tokens", action="store_true", help="Only apply to normal tokens, ignore special symbol positions")
    parser.add_argument("--pref_sets", action="store_true", help="Use preference sets instead of core set")
    parser.add_argument("--compare_baseline", action="store_true", help="Compare with baseline (no suppression)")
    parser.add_argument("--output_dir", type=str, default=None, help="Result output directory")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--rm_bench_script_dir", type=str, default=None, help="RM-Bench scripts directory path")
    args = parser.parse_args()
    
    if args.rm_bench_script_dir:
        if args.rm_bench_script_dir not in sys.path:
            sys.path.insert(0, args.rm_bench_script_dir)
        try:
            from rewardbench import (
                REWARD_MODEL_CONFIG,
                check_tokenizer_chat_template,
                torch_dtype_mapping,
            )
            from utils import (
                compute_accuracy,
                convert_robust_dataset_to_preference_dataset_list,
                load_eval_dataset,
            )
        except ImportError:
            logger.warning("Failed to import RM-Bench utilities, falling back to simple JSON loading")
            REWARD_MODEL_CONFIG = None
            check_tokenizer_chat_template = None
            compute_accuracy = None
            convert_robust_dataset_to_preference_dataset_list = None
            load_eval_dataset = None

    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    try:
        gpu_ids = [int(x.strip()) for x in args.gpus.split(",")]
        available_gpus = [torch.device(f"cuda:{gpu_id}") for gpu_id in gpu_ids]
        logger.info(f"Specified GPU list: {args.gpus} -> {[str(d) for d in available_gpus]}")
    except ValueError as e:
        raise ValueError(f"Invalid GPU list format '{args.gpus}', should be comma-separated numbers, e.g., '0,2,4,5'") from e
    
    device = available_gpus[0]
    logger.info(f"RM model will use device: {device}")
    sae_gpus = available_gpus[1:]
    if not sae_gpus:
        logger.warning("Only one GPU specified, SAE will share device with RM model")
        sae_gpus = [device]

    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if check_tokenizer_chat_template:
        if not check_tokenizer_chat_template(tokenizer):
            tokenizer.add_eos_token = True

    conv = None
    if get_conv_template:
        conv = get_conv_template(args.chat_template)

    if REWARD_MODEL_CONFIG:
        official_model_name = args.model.replace("RewardModels/", "")
        config = REWARD_MODEL_CONFIG.get(official_model_name, REWARD_MODEL_CONFIG.get("default", {}))
        model_type = config.get("model_type", "Seq. Classifier")
        if model_type != "Seq. Classifier":
            raise NotImplementedError(f"Only Seq. Classifier type reward models are supported, got {model_type}")
        model_builder = config.get("model_builder")
        torch_dtype = config.get("torch_dtype") or torch.bfloat16
        if model_builder:
            model = model_builder(args.model, torch_dtype=torch_dtype, trust_remote_code=args.trust_remote_code)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code,
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=args.trust_remote_code,
        )
    
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        hasattr(model, "generation_config")
        and model.generation_config is not None
        and model.generation_config.pad_token_id is None
        and tokenizer.pad_token_id is not None
    ):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    suppressor: Optional[SAEFeatureSuppressor] = None
    if args.selected_features_path:
        if args.sae_release is None:
            raise ValueError(
                "Feature suppression requires --sae_release and --selected_features_path."
            )
        selected_features = load_selected_features(args.selected_features_path)
        layers = sorted(selected_features.keys())
        layer_to_device = {}
        for idx, layer_idx in enumerate(layers):
            gpu_idx = idx % len(sae_gpus)
            layer_to_device[layer_idx] = sae_gpus[gpu_idx]
        logger.info(f"SAE device allocation (round-robin by layer):")
        for layer_idx, assigned_device in sorted(layer_to_device.items()):
            logger.info(f"  Layer {layer_idx} -> {assigned_device}")
        sae_device = sae_gpus[0]
        
        suppressor = SAEFeatureSuppressor(
            model=model,
            tokenizer=tokenizer,
            selected_features=selected_features,
            sae_release=args.sae_release,
            sae_device=sae_device,
            sae_id_template=args.sae_id_template,
            sae_id_suffix=args.sae_id_suffix,
            mask_special_tokens=args.mask_special_tokens,
            suppression_mode=args.suppression_mode,
            suppression_strength=args.suppression_strength,
            logger=logger,
            layer_to_device=layer_to_device,
        )
    else:
        logger.info("No feature suppression list provided, will only run baseline evaluation.")

    if convert_robust_dataset_to_preference_dataset_list and load_eval_dataset:
        logger.info("Loading RM-Bench data using official utilities")
        dataset_list = convert_robust_dataset_to_preference_dataset_list(args.datapath)
        score_chosen_all: List[List[float]] = []
        score_rejected_all: List[List[float]] = []
        baseline_chosen_all: Optional[List[List[float]]] = [] if args.compare_baseline else None
        baseline_rejected_all: Optional[List[List[float]]] = [] if args.compare_baseline else None
        processed_ids_all: List[List[str]] = []

        for dataset_idx, raw_dataset in enumerate(dataset_list):
            logger.info(f"Processing sub-dataset {dataset_idx + 1}/{len(dataset_list)}")
            dataset, subsets = load_eval_dataset(
                raw_dataset,
                core_set=not args.pref_sets,
                conv=conv,
                custom_dialogue_formatting=False,
                tokenizer=tokenizer,
                logger=logger,
                keep_columns=["text_chosen", "text_rejected", "id"],
            )
            ids = dataset["id"]
            dataset = dataset.remove_columns("id")
            processed_ids_all.append(list(ids))

            if args.compare_baseline:
                logger.info("Computing baseline scores without suppression...")
                baseline_scores = evaluate_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dataset=dataset,
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                    suppressor=None,
                    desc_prefix=f"Sub-dataset {dataset_idx + 1}/{len(dataset_list)} - Baseline",
                )
                baseline_chosen_all.append(baseline_scores["chosen"])
                baseline_rejected_all.append(baseline_scores["rejected"])

            logger.info("Executing inference with feature suppression...")
            suppressed_scores = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                suppressor=suppressor,
                desc_prefix=f"Sub-dataset {dataset_idx + 1}/{len(dataset_list)} - Feature suppression",
            )
            score_chosen_all.append(suppressed_scores["chosen"])
            score_rejected_all.append(suppressed_scores["rejected"])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        with open(args.datapath, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)

        id_to_unit_idx: Dict[str, int] = {}
        for idx, unit in enumerate(dataset_json):
            unit_id = unit.get("id", str(idx))
            id_to_unit_idx[unit_id] = idx

        num_datasets = len(processed_ids_all)
        for unit in dataset_json:
            if "score_chosen" not in unit:
                unit["score_chosen"] = [None] * num_datasets
            elif len(unit["score_chosen"]) < num_datasets:
                unit["score_chosen"].extend([None] * (num_datasets - len(unit["score_chosen"])))
            if "score_rejected" not in unit:
                unit["score_rejected"] = [None] * num_datasets
            elif len(unit["score_rejected"]) < num_datasets:
                unit["score_rejected"].extend([None] * (num_datasets - len(unit["score_rejected"])))
            if args.compare_baseline:
                if "baseline_score_chosen" not in unit:
                    unit["baseline_score_chosen"] = [None] * num_datasets
                elif len(unit["baseline_score_chosen"]) < num_datasets:
                    unit["baseline_score_chosen"].extend([None] * (num_datasets - len(unit["baseline_score_chosen"])))
                if "baseline_score_rejected" not in unit:
                    unit["baseline_score_rejected"] = [None] * num_datasets
                elif len(unit["baseline_score_rejected"]) < num_datasets:
                    unit["baseline_score_rejected"].extend([None] * (num_datasets - len(unit["baseline_score_rejected"])))

        for dataset_idx, processed_ids in enumerate(processed_ids_all):
            for score_idx, unit_id in enumerate(processed_ids):
                if unit_id in id_to_unit_idx:
                    unit_idx = id_to_unit_idx[unit_id]
                    unit = dataset_json[unit_idx]
                    if score_idx < len(score_chosen_all[dataset_idx]):
                        unit["score_chosen"][dataset_idx] = score_chosen_all[dataset_idx][score_idx]
                    if score_idx < len(score_rejected_all[dataset_idx]):
                        unit["score_rejected"][dataset_idx] = score_rejected_all[dataset_idx][score_idx]
                    if args.compare_baseline and baseline_chosen_all is not None:
                        if score_idx < len(baseline_chosen_all[dataset_idx]):
                            unit["baseline_score_chosen"][dataset_idx] = baseline_chosen_all[dataset_idx][score_idx]
                        if score_idx < len(baseline_rejected_all[dataset_idx]):
                            unit["baseline_score_rejected"][dataset_idx] = baseline_rejected_all[dataset_idx][score_idx]
    else:
        logger.info("Loading data as simple JSON file")
        with open(args.datapath, "r", encoding="utf-8") as f:
            dataset_json = json.load(f)
        if not isinstance(dataset_json, list):
            raise ValueError(f"Dataset file must be a JSON list, got {type(dataset_json)}")
        dataset_dict = {
            "text_chosen": [item.get("chosen", [""])[0] if isinstance(item.get("chosen"), list) else item.get("chosen", "") for item in dataset_json],
            "text_rejected": [item.get("rejected", [""])[0] if isinstance(item.get("rejected"), list) else item.get("rejected", "") for item in dataset_json],
        }
        suppressed_scores = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            device=device,
            dataset=dataset_dict,
            batch_size=args.batch_size,
            max_length=args.max_length,
            suppressor=suppressor,
            desc_prefix="Feature suppression",
        )
        for i, unit in enumerate(dataset_json):
            if i < len(suppressed_scores["chosen"]):
                unit["score_chosen"] = suppressed_scores["chosen"][i]
            if i < len(suppressed_scores["rejected"]):
                unit["score_rejected"] = suppressed_scores["rejected"][i]

    filename = os.path.basename(args.datapath).replace(".json", "")
    model_name = args.model.split("/")[-1]
    if REWARD_MODEL_CONFIG:
        official_model_name = args.model.replace("RewardModels/", "")
        output_root = (
            args.output_dir
            if args.output_dir
            else os.path.join("results", "Seq_Classifier", official_model_name)
        )
    else:
        output_root = (
            args.output_dir
            if args.output_dir
            else os.path.join("results", model_name)
        )
    os.makedirs(output_root, exist_ok=True)

    suffix = "SAE_SUPPRESSED" if suppressor is not None else "BASELINE"
    output_path = os.path.join(
        output_root,
        f"{filename}_{model_name}_{suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=4, ensure_ascii=False)

    if compute_accuracy and convert_robust_dataset_to_preference_dataset_list:
        def has_valid_scores(unit: Dict) -> bool:
            if "score_chosen" not in unit or "score_rejected" not in unit:
                return False
            score_chosen = unit["score_chosen"]
            score_rejected = unit["score_rejected"]
            if not isinstance(score_chosen, list) or not isinstance(score_rejected, list):
                return False
            num_datasets = len(score_chosen)
            if len(score_rejected) != num_datasets:
                return False
            return all(
                s is not None and isinstance(s, (int, float)) 
                for s in score_chosen + score_rejected
            )
        
        if convert_robust_dataset_to_preference_dataset_list:
            num_datasets = len(processed_ids_all) if 'processed_ids_all' in locals() else 1
            dataset_with_scores = [unit for unit in dataset_json if has_valid_scores(unit)]
            if dataset_with_scores and len(dataset_with_scores) > 0:
                acc_dict = safe_compute_accuracy(dataset_with_scores)
                if acc_dict is not None:
                    logger.info(
                        f"Accuracy after feature suppression (based on {len(dataset_with_scores)}/{len(dataset_json)} samples):\n{json.dumps(acc_dict, ensure_ascii=False, indent=2)}"
                    )
            if args.compare_baseline:
                def has_valid_baseline_scores(unit: Dict) -> bool:
                    if "baseline_score_chosen" not in unit or "baseline_score_rejected" not in unit:
                        return False
                    score_chosen = unit["baseline_score_chosen"]
                    score_rejected = unit["baseline_score_rejected"]
                    if not isinstance(score_chosen, list) or not isinstance(score_rejected, list):
                        return False
                    if len(score_chosen) != len(score_rejected):
                        return False
                    return all(
                        s is not None and isinstance(s, (int, float)) 
                        for s in score_chosen + score_rejected
                    )
                baseline_dataset = [
                    {
                        **unit,
                        "score_chosen": unit["baseline_score_chosen"],
                        "score_rejected": unit["baseline_score_rejected"],
                    }
                    for unit in dataset_json
                    if has_valid_baseline_scores(unit)
                ]
                if baseline_dataset and len(baseline_dataset) > 0:
                    baseline_acc = safe_compute_accuracy(baseline_dataset)
                    if baseline_acc is not None:
                        logger.info(
                            f"Baseline accuracy without suppression (based on {len(baseline_dataset)}/{len(dataset_json)} samples):\n{json.dumps(baseline_acc, ensure_ascii=False, indent=2)}"
                        )

    if suppressor is not None:
        suppressor.close()
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()

