# SteerRM

Training-free debiasing of reward models using Sparse Autoencoder (SAE)-based interventions.

## Acknowledgments

This implementation reuses utilities from the [RM-Bench](https://github.com/THU-KEG/RM-Bench) repository. We thank the authors for providing:
- Data loading and formatting utilities (`convert_robust_dataset_to_preference_dataset_list`, `load_eval_dataset`)
- Model configuration utilities (`REWARD_MODEL_CONFIG`, `check_tokenizer_chat_template`)
- Evaluation metrics (`compute_accuracy`)

## File Descriptions

- **synthesize_dataset.py**: Synthesizes paired format-controlled responses (Markdown vs Plain) using LLM APIs.

- **identify_features.py**: Identifies format-related SAE features by analyzing activation differences between Markdown and Plain responses using strength-stability criterion.

- **compute_bias_directions.py**: Computes bias direction vectors by comparing activations from biased and unbiased data. Supports multiple computation methods.

- **evaluate_base_sae.py**: Evaluates SAE reconstruction performance (MSE, L0 sparsity) on base language models using WildChat dataset.

- **evaluate_rm_sae.py**: Evaluates SAE reconstruction performance and reward preservation (Reward Delta) on reward models using WildChat dataset.

- **evaluate_rm_bench_activation.py**: Evaluates reward models on RM-Bench with direct activation suppression (without SAEs). Requires bias directions computed by `compute_bias_directions.py`.

- **evaluate_rm_bench_features.py**: Evaluates reward models on RM-Bench with SAE feature suppression. Requires selected features from `identify_features.py`.

## Usage

The scripts follow a three-stage pipeline:

1. **Data Synthesis**: Run `synthesize_dataset.py` to generate paired format-controlled responses.

2. **Feature Identification**: Run `identify_features.py` on the synthesized dataset to identify format-related SAE features.

3. **Evaluation**: Run `evaluate_rm_bench_features.py` to evaluate the model with feature suppression on RM-Bench.

For activation-based suppression (baseline comparison), use `compute_bias_directions.py` followed by `evaluate_rm_bench_activation.py`.

## Requirements

- PyTorch
- transformers
- sae-lens
- datasets
- RM-Bench repository (for evaluation scripts)

