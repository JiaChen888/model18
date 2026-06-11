#!/usr/bin/env bash
set -euo pipefail

PROCESS_DIR="${1:-/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/preprocess_data_current_clean_300_model10v2_threshold50_65_78_20260605}"
SPLIT_SOURCE="${2:-/media/zj/f6835876-8873-4931-8383-70fe383c9b741/IDPss-output/model_current_clean_300_epoch100_model8from6a_20260605}"
OUTDIR="${3:-outputs/preprocess_db_current_clean_300}"

python3 scripts/preprocess_step1_build_index.py --process-dir "$PROCESS_DIR" --outdir "$OUTDIR"
python3 scripts/preprocess_step2_smoke_validate_arrays.py --index "$OUTDIR/preprocessed_sample_index.csv" --outdir "$OUTDIR" --max-samples 12
python3 scripts/preprocess_step3_build_training_manifest.py --index "$OUTDIR/preprocessed_sample_index.csv" --outdir "$OUTDIR" --split-source "$SPLIT_SOURCE"
