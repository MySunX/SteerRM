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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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


def load_bias_directions(path: str) -> Dict[int, torch.Tensor]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    bias_directions = {}
    for key, vec in data.items():
        if not key.startswith("layer_"):
            continue
        layer_idx = int(key.split("_")[1])
        bias_directions[layer_idx] = torch.tensor(vec, dtype=torch.float32)
    if not bias_directions:
        raise ValueError(f"File {path} contains no valid layer_X keys")
    return bias_directions


class ActivationBiasSuppressor:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        bias_directions: Dict[int, torch.Tensor],
        suppression_strength: float,
        mask_special_tokens: bool,
        logger: logging.Logger,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.bias_directions = bias_directions
        self.suppression_strength = suppression_strength
        self.mask_special_tokens = mask_special_tokens
        self.logger = logger
        self.layers_module = find_transformer_layers(model)
        self.layer_count = len(self.layers_module)
        self.target_layers = sorted(bias_directions.keys())
        for layer_idx in self.target_layers:
            if layer_idx >= self.layer_count:
                raise ValueError(
                    f"Bias directions contains layer {layer_idx}, but model only has {self.layer_count} layers"
                )
        self.special_token_ids = (
            get_special_token_ids(tokenizer) if mask_special_tokens else None
        )
        self.current_valid_mask: Optional[torch.Tensor] = None
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        self.bias_directions_tensor: Dict[int, torch.Tensor] = {}
        for layer_idx, direction in bias_directions.items():
            self.bias_directions_tensor[layer_idx] = direction.to(
                device=model_device, dtype=model_dtype
            )
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

    def _register_hooks(self):
        handles = []
        for layer_idx in self.target_layers:
            module = self.layers_module[layer_idx]
            handle = module.register_forward_hook(self._make_hook(layer_idx))
            handles.append(handle)
        self.logger.info(
            f"Registered Activation suppression hooks on {len(handles)} Transformer layers: {self.target_layers}"
        )
        return handles

    def _make_hook(self, layer_idx: int):
        bias_direction = self.bias_directions_tensor[layer_idx]

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            delta = self.suppression_strength * bias_direction.to(
                device=hidden.device, dtype=hidden.dtype
            )
            suppressed = hidden - delta

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
    suppressor: Optional[ActivationBiasSuppressor] = None,
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
    suppressor: Optional[ActivationBiasSuppressor],
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
        description="Evaluate on RM-Bench with activation bias suppression."
    )
    parser.add_argument("--model", type=str, required=True, help="Reward model name or weight path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional: specify tokenizer separately")
    parser.add_argument("--chat_template", type=str, default="tulu", help="FastChat conversation template name")
    parser.add_argument("--datapath", type=str, required=True, help="RM-Bench data path (JSON file)")
    parser.add_argument("--batch_size", type=int, default=32, help="Inference batch size")
    parser.add_argument("--max_length", type=int, default=2048, help="Tokenizer max length")
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device, e.g., cuda:0")
    parser.add_argument("--bias_directions_path", type=str, default=None, help="Bias direction vector file path (JSON format)")
    parser.add_argument("--suppression_strength", type=float, default=1.0, help="Suppression strength coefficient")
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

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

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

    suppressor: Optional[ActivationBiasSuppressor] = None
    if args.bias_directions_path:
        logger.info(f"Loading bias directions: {args.bias_directions_path}")
        bias_directions = load_bias_directions(args.bias_directions_path)
        logger.info(f"Loaded {len(bias_directions)} layers of bias directions: {sorted(bias_directions.keys())}")
        
        suppressor = ActivationBiasSuppressor(
            model=model,
            tokenizer=tokenizer,
            bias_directions=bias_directions,
            suppression_strength=args.suppression_strength,
            mask_special_tokens=args.mask_special_tokens,
            logger=logger,
        )
    else:
        logger.info("No bias directions provided, will only run baseline evaluation.")

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

            logger.info("Executing inference with activation suppression...")
            suppressed_scores = evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                device=device,
                dataset=dataset,
                batch_size=args.batch_size,
                max_length=args.max_length,
                suppressor=suppressor,
                desc_prefix=f"Sub-dataset {dataset_idx + 1}/{len(dataset_list)} - Activation suppression",
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
            desc_prefix="Activation suppression",
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

    suffix = "ACTIVATION_SUPPRESSED" if suppressor is not None else "BASELINE"
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
                        f"Accuracy after activation suppression (based on {len(dataset_with_scores)}/{len(dataset_json)} samples):\n{json.dumps(acc_dict, ensure_ascii=False, indent=2)}"
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

