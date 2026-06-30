#!/bin/bash
# Package ratio-sweep metrics for scp back to local.
#
# Run on Hyak after judging:
#   bash scripts/package_em_bad_ratio_joke_sweep.sh
#
# The archive is written on /gscratch, not home, to avoid home quota failures.

set -euo pipefail

COUNTS="${COUNTS:-1 2 3 5}"
SAMPLE_N="${SAMPLE_N:-5}"
JOKE_SAMPLE_N="${JOKE_SAMPLE_N:-5}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/gscratch/jamiemmt/claizhan/subliminal-mitigate/outputs/em_qwen25_7b_bad_medical_vs_benign_medical_joke}"
SWEEP_ROOT="$OUTPUT_ROOT/bad_ratio_sweep_joke"
EXISTING_4_ROOT="$OUTPUT_ROOT/majority_bad_medical_union_4bad_1good"
ARCHIVE="${ARCHIVE:-$SWEEP_ROOT/em_bad_ratio_joke_sweep_results.tgz}"

cd "$OUTPUT_ROOT"
shopt -s nullglob

files=()
for bad_count in $COUNTS; do
  ratio_rel="bad_ratio_sweep_joke/${bad_count}bad_1good"
  files+=( "$ratio_rel/narrow_medical_s${SAMPLE_N}_a100_1gpu"/metrics_medical_*_ratio.* )
  files+=( "$ratio_rel/narrow_medical64_s${SAMPLE_N}_a100_1gpu"/metrics_medical64_*_ratio.* )
  files+=( "$ratio_rel/joke_suffix_s${JOKE_SAMPLE_N}_a100_1gpu"/metrics_joke_suffix_ratio.* )
  files+=( "$ratio_rel/eval64_s${SAMPLE_N}_a100_1gpu"/metrics_*_ratio.* )
done

# Include the existing 4B:1G point so the local plotter can build the full curve.
files+=( "majority_bad_medical_union_4bad_1good/narrow_medical_joke_5way_s${SAMPLE_N}_a100_1gpu"/metrics_medical_judged_with_5way.* )
files+=( "majority_bad_medical_union_4bad_1good/narrow_medical_joke_5way_s${SAMPLE_N}_a100_1gpu"/metrics_medical_nojudge_with_5way.* )
files+=( "majority_bad_medical_union_4bad_1good/narrow_medical64_joke_5way_s${SAMPLE_N}_a100_1gpu"/metrics_medical64_*_ratio.* )
files+=( "majority_bad_medical_union_4bad_1good/joke_suffix_5way_s${JOKE_SAMPLE_N}_a100_1gpu"/metrics_joke_suffix_with_5way.* )
files+=( "majority_bad_medical_union_4bad_1good/eval64_joke_5way_s${SAMPLE_N}_a100_1gpu"/metrics_judged_with_5way.* )
files+=( "majority_bad_medical_union_4bad_1good/eval64_joke_5way_s${SAMPLE_N}_a100_1gpu"/metrics_nojudge_with_5way.* )

if [[ "${#files[@]}" -eq 0 ]]; then
  echo "No metric files found to package." >&2
  exit 2
fi

mkdir -p "$(dirname "$ARCHIVE")"
tar -czf "$ARCHIVE" "${files[@]}"
ls -lh "$ARCHIVE"
tar -tzf "$ARCHIVE" | sed -n '1,80p'
echo "Archive: $ARCHIVE"
