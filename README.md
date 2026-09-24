# Feature Fusion Gated Recurrent Unit for Pipe Burst Detection and Localization in Water Distribution Networks

Implementation of FF-GRU and seven comparison models for pipe burst detection and localization.

## Models

The baseline models are GRU, CNN, GCN and FA-DenseNet. Their feature-fusion counterparts are FF-GRU, FF-CNN, FF-GCN and FF-FA-DenseNet. Model definitions are in `models/` and `fusion_models/`; `data/` contains hydraulic generation and preprocessing code.

## Setup

Python 3.9 is recommended. Install dependencies with `pip install -r requirements.txt`, then copy `config.py.example` to `config.py` and set the local paths, ordered sensor IDs, partition count and `MAX_EPOCHS`.

## Workflow

1. Generate simulation inputs with `python scripts/generate_data.py`.
2. Allocate source periods and burst events before windowing with `python scripts/prepare_splits.py --manifest LOCAL_MANIFEST.csv --out-dir outputs/splits --window 60 --seed 66 --pressure-sensors P1 P2 --flow-sensors Q1 Q2`. The manifest columns are `group_id,kind,pressure,flow,partition_id`; `partition_id` is empty for normal periods and 1-based for burst events. Use one `group_id` for each source period or parent burst event, including paired normal and burst inputs.
3. Set `TASK` to `detection` or `localization`, choose `MODEL_NAME`, and run `python scripts/train.py`. Keep the source-group split fixed across models; use training seeds 66-70 for repeated runs.
4. Evaluate matched task predictions with `python scripts/evaluate_metrics.py --detection DETECTION.csv --localization LOCALIZATION.csv --output METRICS.json`. Both prediction files require `sample_id`, `y_true` and `y_pred`; localization IDs must match the true burst IDs in detection.

The training entrypoint selects the lowest validation-loss checkpoint before test evaluation. `FUSION_MODES` in `config.py` controls the feature-fusion variants. `scripts/joint_training.py` provides an optional joint fine-tuning step for two pretrained task models; the main comparison trains the tasks separately.

## Metrics

`scripts/metrics.py` computes detection accuracy (DAC), precision (PRE), recall (REC), false-alarm rate (FAR), F1, localization accuracy (LAC), macro-F1, coordinated accuracy (CAC) and PB-specific comprehensive accuracy (`CAC_PB`, true burst samples only). Five-seed summaries use mean and sample SD.

## License

See [LICENSE](LICENSE).
