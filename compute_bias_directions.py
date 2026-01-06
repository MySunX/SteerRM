import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


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


def load_data(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError(f"Data file {path} should be a list or dict with 'data' key, got {type(data)}")


def load_paired_data_dir(dir_path: str) -> List[Dict]:
    if dir_path is None:
        raise ValueError("dir_path cannot be empty")
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"paired_data_dir does not exist or is not a directory: {dir_path}")
    files = sorted(
        [
            os.path.join(dir_path, fn)
            for fn in os.listdir(dir_path)
            if fn.endswith(".json")
        ]
    )
    if len(files) == 0:
        raise FileNotFoundError(f"No .json files found in paired_data_dir: {dir_path}")
    merged: List[Dict] = []
    for fp in files:
        part = load_data(fp)
        if not isinstance(part, list):
            raise ValueError(f"File {fp} is not a list after parsing")
        merged.extend(part)
    logger.info(f"Loaded paired data from directory: dir={dir_path}, json_files={len(files)}, total_samples={len(merged)}")
    return merged


def load_rm_bench_dataset(split: str = "train", max_samples: Optional[int] = None):
    ds = load_dataset("THU-KEG/RM-Bench")
    if split not in ds:
        raise ValueError(f"RM-Bench split '{split}' does not exist, available: {list(ds.keys())}")
    dataset = ds[split]
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def convert_rm_bench_to_pairs(dataset) -> Tuple[List[Dict], List[Dict]]:
    biased = []
    unbiased = []
    for item in dataset:
        prompt = item.get("prompt")
        chosen = item.get("chosen", [])
        rejected = item.get("rejected", [])
        if prompt is None or len(chosen) < 3 or len(rejected) < 3:
            logger.warning("Sample missing required fields or insufficient length, skipping")
            continue
        unbiased.append({"prompt": prompt, "response": chosen[1]})
        unbiased.append({"prompt": prompt, "response": rejected[1]})
        biased.append({"prompt": prompt, "response": chosen[2]})
        biased.append({"prompt": prompt, "response": rejected[2]})
    logger.info(f"Converted from RM-Bench: biased={len(biased)}, unbiased={len(unbiased)}")
    return biased, unbiased


def extract_biased_unbiased_from_paired_data(
    data: List[Dict],
    biased_key: str = "answer_markdown",
    unbiased_key: str = "answer_plain",
) -> Tuple[List[Dict], List[Dict]]:
    biased = []
    unbiased = []
    for item in data:
        if "prompt" not in item:
            logger.warning(f"Sample missing 'prompt' field, skipping: {list(item.keys())}")
            continue
        prompt = item["prompt"]
        if biased_key in item and unbiased_key in item:
            biased.append({"prompt": prompt, "response": item[biased_key]})
            unbiased.append({"prompt": prompt, "response": item[unbiased_key]})
        else:
            logger.warning(
                f"Sample missing '{biased_key}' or '{unbiased_key}' field, skipping: {list(item.keys())}"
            )
    logger.info(f"Extracted {len(biased)} pairs of biased/unbiased samples from paired data")
    return biased, unbiased


def format_conversation(conv: List[Dict[str, str]], tokenizer) -> str:
    if getattr(tokenizer, "chat_template", None):
        formatted = tokenizer.apply_chat_template(conv, tokenize=False)
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
    if tokenizer.bos_token and formatted.startswith(tokenizer.bos_token):
        formatted = formatted[len(tokenizer.bos_token):]
    return formatted


def format_text(item: Dict, tokenizer, chat_template: Optional[str] = None) -> str:
    if "text" in item:
        return item["text"]
    if "prompt" in item:
        prompt = item["prompt"]
        response = item.get("response") or item.get("answer")
        if response is None:
            raise ValueError(f"Data item contains 'prompt' but missing 'response' or 'answer': {item.keys()}")
    else:
        raise ValueError(f"Data item must contain 'text' or 'prompt': {item.keys()}")
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    return format_conversation(conversation, tokenizer)


def get_special_token_ids(tokenizer) -> Optional[torch.Tensor]:
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


def extract_activations(
    model,
    tokenizer,
    data: List[Dict],
    target_layers: List[int],
    batch_size: int,
    max_length: int,
    device: torch.device,
    mask_special_tokens: bool,
    chat_template: Optional[str],
) -> Dict[int, torch.Tensor]:
    special_token_ids = get_special_token_ids(tokenizer) if mask_special_tokens else None
    layer_activations: Dict[int, List[torch.Tensor]] = {layer: [] for layer in target_layers}
    layers_module = find_transformer_layers(model)
    logger.info(f"Extracting activations for {len(data)} samples...")
    for start_idx in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, len(data))
        batch_data = data[start_idx:end_idx]
        batch_texts = [format_text(item, tokenizer, chat_template) for item in batch_data]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        valid_token_mask = attention_mask.bool()
        if special_token_ids is not None:
            special_ids = special_token_ids.to(input_ids.device)
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

            for layer_idx in target_layers:
                if layer_idx < 0 or layer_idx >= len(layers_module):
                    logger.warning(f"Target layer {layer_idx} out of model layer range, skipping")
                    continue
                handles.append(layers_module[layer_idx].register_forward_hook(_make_hook(layer_idx)))

            try:
                try:
                    _ = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True,
                    )
                except TypeError:
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
            finally:
                for h in handles:
                    h.remove()
        for layer_idx in target_layers:
            if layer_idx not in captured:
                logger.warning(f"Failed to capture hidden activations for layer {layer_idx} (hook not triggered or structure mismatch)")
                continue
            layer_hidden = captured[layer_idx]
            mask_expanded = valid_token_mask.unsqueeze(-1)
            for b in range(layer_hidden.size(0)):
                valid_positions = mask_expanded[b].squeeze(-1)
                if valid_positions.any():
                    valid_acts = layer_hidden[b][valid_positions]
                    layer_activations[layer_idx].append(valid_acts.cpu())
        del captured, handles
        torch.cuda.empty_cache()
    avg_activations = {}
    for layer_idx in target_layers:
        if len(layer_activations[layer_idx]) == 0:
            logger.warning(f"Layer {layer_idx} collected no valid activations")
            avg_activations[layer_idx] = None
            continue
        all_acts = torch.cat(layer_activations[layer_idx], dim=0)
        avg_act = all_acts.mean(dim=0)
        avg_activations[layer_idx] = avg_act
        logger.info(f"Layer {layer_idx}: collected {all_acts.size(0)} token activations")
    return avg_activations


def compute_bias_directions_mean_diff(
    biased_acts: Dict[int, torch.Tensor],
    unbiased_acts: Dict[int, torch.Tensor],
    normalize: bool,
) -> Dict[int, torch.Tensor]:
    bias_directions = {}
    for layer_idx in biased_acts.keys():
        if biased_acts[layer_idx] is None or unbiased_acts[layer_idx] is None:
            logger.warning(f"Layer {layer_idx} activation is empty, skipping")
            continue
        direction = biased_acts[layer_idx] - unbiased_acts[layer_idx]
        if normalize:
            direction = direction / (direction.norm() + 1e-8)
        bias_directions[layer_idx] = direction
        logger.info(f"Layer {layer_idx}: bias direction norm = {direction.norm().item():.4f}")
    return bias_directions


def compute_bias_directions_pca(
    biased_acts: Dict[int, torch.Tensor],
    unbiased_acts: Dict[int, torch.Tensor],
    n_components: int,
    normalize: bool,
) -> Dict[int, torch.Tensor]:
    from sklearn.decomposition import PCA
    bias_directions = {}
    for layer_idx in biased_acts.keys():
        if biased_acts[layer_idx] is None or unbiased_acts[layer_idx] is None:
            logger.warning(f"Layer {layer_idx} activation is empty, skipping")
            continue
        diff = (biased_acts[layer_idx] - unbiased_acts[layer_idx]).unsqueeze(0).numpy()
        pca = PCA(n_components=min(n_components, diff.shape[0], diff.shape[1]))
        pca.fit(diff)
        direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        if normalize:
            direction = direction / (direction.norm() + 1e-8)
        bias_directions[layer_idx] = direction
        logger.info(f"Layer {layer_idx}: PCA bias direction norm = {direction.norm().item():.4f}, "
                   f"explained variance ratio = {pca.explained_variance_ratio_[0]:.4f}")
    return bias_directions


def compute_bias_directions_contrast(
    biased_acts: Dict[int, torch.Tensor],
    unbiased_acts: Dict[int, torch.Tensor],
    normalize: bool,
) -> Dict[int, torch.Tensor]:
    bias_directions = {}
    for layer_idx in biased_acts.keys():
        if biased_acts[layer_idx] is None or unbiased_acts[layer_idx] is None:
            logger.warning(f"Layer {layer_idx} activation is empty, skipping")
            continue
        biased_norm = biased_acts[layer_idx] / (biased_acts[layer_idx].norm() + 1e-8)
        unbiased_norm = unbiased_acts[layer_idx] / (unbiased_acts[layer_idx].norm() + 1e-8)
        direction = biased_norm - unbiased_norm
        if normalize:
            direction = direction / (direction.norm() + 1e-8)
        bias_directions[layer_idx] = direction
        logger.info(f"Layer {layer_idx}: contrast bias direction norm = {direction.norm().item():.4f}")
    return bias_directions


def main():
    parser = argparse.ArgumentParser(
        description="Compute bias directions from biased and unbiased data activation differences"
    )
    parser.add_argument("--model", type=str, required=True, help="Reward model path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to model path)")
    parser.add_argument(
        "--use_rm_bench",
        action="store_true",
        help="Load data from THU-KEG/RM-Bench (markdown vs long)",
    )
    parser.add_argument(
        "--rm_bench_split",
        type=str,
        default="train",
        help="RM-Bench split, default train",
    )
    parser.add_argument(
        "--biased_data",
        type=str,
        default=None,
        help="Biased data JSON file path, format: [{'prompt': ..., 'response': ...}, ...]",
    )
    parser.add_argument(
        "--unbiased_data",
        type=str,
        default=None,
        help="Unbiased data JSON file path, same format as above",
    )
    parser.add_argument(
        "--paired_data",
        type=str,
        default=None,
        help="Paired data JSON file path, format: [{'prompt': ..., 'answer_markdown': ..., 'answer_plain': ...}, ...]",
    )
    parser.add_argument(
        "--paired_data_dir",
        type=str,
        default=None,
        help="Directory containing paired data JSON files (will read all .json files and merge)",
    )
    parser.add_argument(
        "--biased_key",
        type=str,
        default="answer_markdown",
        help="Field name for biased response in paired data (effective when using --paired_data or --paired_data_dir)",
    )
    parser.add_argument(
        "--unbiased_key",
        type=str,
        default="answer_plain",
        help="Field name for unbiased response in paired data (effective when using --paired_data or --paired_data_dir)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output bias directions JSON file path",
    )
    parser.add_argument(
        "--target_layers",
        type=str,
        default="0,8,16,24",
        help="Transformer layers to extract, comma-separated, e.g., '0,8,16,24'",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mean_diff",
        choices=["mean_diff", "pca", "contrast"],
        help="Method to compute bias direction: mean_diff (average difference), pca (principal component analysis), contrast (contrastive learning style)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Inference device",
    )
    parser.add_argument(
        "--mask_special_tokens",
        action="store_true",
        help="Ignore special tokens when extracting activations",
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=1,
        help="Number of principal components for PCA method (default 1, takes first principal component)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize bias direction to unit vector",
    )
    parser.add_argument(
        "--chat_template",
        type=str,
        default=None,
        help="Chat template to use (optional, not needed if data is already formatted text)",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Allow loading custom code",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for quick testing)",
    )
    args = parser.parse_args()
    
    device = torch.device(args.device)
    target_layers = [int(x.strip()) for x in args.target_layers.split(",")]
    
    logger.info(f"Target layers: {target_layers}")
    logger.info(f"Method: {args.method}")
    logger.info(f"Device: {device}")
    
    logger.info(f"Loading model: {args.model}")
    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    
    if args.use_rm_bench:
        logger.info(f"Loading data from RM-Bench split={args.rm_bench_split}")
        dataset = load_rm_bench_dataset(split=args.rm_bench_split, max_samples=args.max_samples)
        biased_data, unbiased_data = convert_rm_bench_to_pairs(dataset)
    elif args.paired_data:
        logger.info(f"Loading from paired data: {args.paired_data}")
        logger.info(f"  Biased key: {args.biased_key}")
        logger.info(f"  Unbiased key: {args.unbiased_key}")
        paired_data = load_data(args.paired_data)
        biased_data, unbiased_data = extract_biased_unbiased_from_paired_data(
            paired_data,
            biased_key=args.biased_key,
            unbiased_key=args.unbiased_key,
        )
    elif args.paired_data_dir:
        logger.info(f"Loading from paired data directory: {args.paired_data_dir}")
        logger.info(f"  Biased key: {args.biased_key}")
        logger.info(f"  Unbiased key: {args.unbiased_key}")
        paired_data = load_paired_data_dir(args.paired_data_dir)
        biased_data, unbiased_data = extract_biased_unbiased_from_paired_data(
            paired_data,
            biased_key=args.biased_key,
            unbiased_key=args.unbiased_key,
        )
    elif args.biased_data and args.unbiased_data:
        logger.info(f"Loading biased data: {args.biased_data}")
        biased_data = load_data(args.biased_data)
        logger.info(f"Loading unbiased data: {args.unbiased_data}")
        unbiased_data = load_data(args.unbiased_data)
    else:
        raise ValueError(
            "Must provide one of the following:\n"
            "  1. --use_rm_bench (directly use RM-Bench markdown vs long)\n"
            "  2. --paired_data (paired data file)\n"
            "  2b. --paired_data_dir (paired data directory, reads all json files and merges)\n"
            "  3. --biased_data and --unbiased_data (provide two files separately)"
        )
    
    if args.max_samples:
        biased_data = biased_data[:args.max_samples]
        unbiased_data = unbiased_data[:args.max_samples]
        logger.info(f"Limited sample count to: {args.max_samples}")
    
    logger.info(f"Biased samples: {len(biased_data)}")
    logger.info(f"Unbiased samples: {len(unbiased_data)}")
    
    logger.info("=" * 60)
    logger.info("Extracting biased data activations")
    logger.info("=" * 60)
    biased_acts = extract_activations(
        model=model,
        tokenizer=tokenizer,
        data=biased_data,
        target_layers=target_layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        mask_special_tokens=args.mask_special_tokens,
        chat_template=args.chat_template,
    )
    
    logger.info("=" * 60)
    logger.info("Extracting unbiased data activations")
    logger.info("=" * 60)
    unbiased_acts = extract_activations(
        model=model,
        tokenizer=tokenizer,
        data=unbiased_data,
        target_layers=target_layers,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        mask_special_tokens=args.mask_special_tokens,
        chat_template=args.chat_template,
    )
    
    logger.info("=" * 60)
    logger.info("Computing bias directions")
    logger.info("=" * 60)
    
    if args.method == "mean_diff":
        bias_directions = compute_bias_directions_mean_diff(
            biased_acts, unbiased_acts, normalize=args.normalize
        )
    elif args.method == "pca":
        bias_directions = compute_bias_directions_pca(
            biased_acts, unbiased_acts, n_components=args.pca_components, normalize=args.normalize
        )
    elif args.method == "contrast":
        bias_directions = compute_bias_directions_contrast(
            biased_acts, unbiased_acts, normalize=args.normalize
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    output_data = {}
    for layer_idx, direction in bias_directions.items():
        output_data[f"layer_{layer_idx}"] = direction.tolist()
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Bias directions saved to: {args.output_path}")
    logger.info(f"Contains {len(bias_directions)} layer direction vectors")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

