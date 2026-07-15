#!/usr/bin/env bash
# Run evaluation_comparison.py against the article-only baseline run.
#
# Base run:    best_article_only_20260419_195147
#              (BestArticleOnlyCnf -- article-only HPO-selected baseline)
#
# Each "current" run below is a saved model directory containing the
# pickled artifacts required by evaluation_comparison (predictions.pkl
# and eval_corpus.pkl). The config_name passed to the comparison CLI is
# derived from the canonical config_map keys in src/iptc_entity_pipeline/config.py
# so the output folder/workbook clearly identifies what is being compared.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVED_MODELS_DIR="$ROOT_DIR/results/saved_models"
OUTPUT_ROOT="$ROOT_DIR/results/final_comparisons"
BASE_DIR_NAME="article_only_tuned_20260706_072439"
BASE_RUN="$SAVED_MODELS_DIR/$BASE_DIR_NAME"

THRESHOLD_EVAL="${THRESHOLD_EVAL:-0.5}"
AVERAGING_TYPE="${AVERAGING_TYPE:-micro}"
TOP_N="${TOP_N:-20}"

# Activate project virtualenv (matches run_all_configs.sh).
# shellcheck disable=SC1091
source "$ROOT_DIR/venv/bin/activate"

# Pairs of "<config_name>:<saved_model_dir_name>".
# config_name follows the canonical names in config._config_map() so the
# comparison output is named after the variant being evaluated.
runs=(
  #'wikipedia2vec_entities_all_langs:wikipedia2vec_entities_all_langs_20260429_103320'
  #'wikidata_description_entities:wikidata_description_entities_20260429_024239'
  # only entities models 
  'wpentities_pmm_entity_only:wpentities_pmm_tuned_only_20260610_221202'
  'wikidata_desc_entity_only:wikidata_description_entity_only_mean_20260608_014214'
  'wiki_intro_entity_only:wikipedia_intro_entity_only_mean_20260608_074455'
  'wiki_article_entity_only:wikipedia_article_entity_only_mean_20260607_220806'
  'wiki2vec_entity_only:wikipedia2vec_entity_only_mean_20260607_170649'
  'wpentities_entity_only:wp_entity_only_mean_20260607_125221'
  # mentions weighted mean
  'wpentities_pmm_mean_mentions:wpentities_pmm_mention_weighted_mean_20260708_064727'
  'wikidata_desc_:wikidata_description_mention_weighted_mean_20260708_110904'
  'wiki_intro_mean_mentions:wikipedia_intro_mention_weighted_mean_20260708_034728'
  'wiki_article_mean_mentions:wikipedia_article_mention_weighted_mean_20260708_131929'
  'wiki2vec_mean_mentions:wikipedia2vec_mention_weighted_mean_20260708_063141'
  'wpentities_mean_mentions:wpentities_mention_weighted_mean_20260605_072623'
  # attention
  #'wpentities_pmm_att:wpentities_pmm_attention_20260601_011554'
  'wikidata_desc_att:wikidata_description_attention_20260705_204132'
  'wiki_intro_att:wikipedia_intro_attention_20260703_181655'
  'wiki_article_att:wikipedia_article_entities_attention_20260611_080142'
  'wiki2vec_att:w2vec_attention_20260611_063024'
  'wpentities_att:best_wpentities_attention_hpo_20260705_230519'
  # best
  'wpentities_all_langs_fallback_att(best):wpentities_all_langs_fallback_attention_20260706_083729'
  #'wikipedia2vec_entities:wikipedia2vec_entities_20260424_152431'
  #'wpentities_mention_weighted_sum:wpentities_mention_weighted_sum_20260420_044121'
  #'wpentities_relevance_weighted_sum:wpentities_relevance_weighted_sum_20260420_022247'
  #'wpentities_weighted_mean:wpentities_weighted_mean_20260420_025827'
  #'wpentities_en_nl:best_wpentities_en_nl_20260419_231207'
  #'wpentities_nl:best_wpentities_nl_20260419_230851'
  #'wpentities_all_langs:best_wpentities_all_langs_20260419_230826'
  #'wpentities:best_wpentities_20260419_195209'
)

check_run_dir() {
  local label="$1"
  local run_dir="$2"
  if [ ! -d "$run_dir" ]; then
    echo "[ERROR] $label: directory not found: $run_dir" >&2
    return 1
  fi
  local missing=0
  for fname in predictions.pkl eval_corpus.pkl; do
    if [ ! -f "$run_dir/$fname" ]; then
      echo "[ERROR] $label: missing $fname in $run_dir" >&2
      missing=1
    fi
  done
  return "$missing"
}

if ! check_run_dir 'base' "$BASE_RUN"; then
  echo "Base run is invalid; aborting." >&2
  exit 1
fi

echo "Base run: $BASE_RUN"
echo "Output root: $OUTPUT_ROOT"
echo "Threshold: $THRESHOLD_EVAL  Averaging: $AVERAGING_TYPE"
echo

failures=()
for entry in "${runs[@]}"; do
  cfg_name="${entry%%:*}"
  dir_name="${entry##*:}"
  current_run="$SAVED_MODELS_DIR/$dir_name"
  comparison_name="${cfg_name}_vs_${BASE_DIR_NAME}"

  echo "=== ${comparison_name} ==="
  if ! check_run_dir "$cfg_name" "$current_run"; then
    failures+=("$cfg_name")
    continue
  fi

  if ! python3 -m iptc_entity_pipeline.evaluation.comparison \
      --current-run "$current_run" \
      --base-run "$BASE_RUN" \
      --config-name "$comparison_name" \
      --threshold-eval "$THRESHOLD_EVAL" \
      --averaging-type "$AVERAGING_TYPE" \
      --top-n "$TOP_N" \
      --output-root "$OUTPUT_ROOT"; then
    echo "[ERROR] comparison failed for $cfg_name" >&2
    failures+=("$cfg_name")
  fi
  echo
done

if [ "${#failures[@]}" -gt 0 ]; then
  echo "Comparisons failed for: ${failures[*]}" >&2
  exit 1
fi

echo "All comparisons completed successfully."
