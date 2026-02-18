#!/usr/bin/env bash

set -euo pipefail

mkdir -p nonpatched

# apply patches
for f in *.analysis.jsonl.gz
do
  zcat "${f}" \
  | python3 -m geneea.mediacats.tools.patchCategoriesInMetadata \
      --taxonomy iptc \
      --patches ../corpora-iptc-patches/*.patch.tsv \
      --category-column cat \
      --score-column score \
  | pv -Wtbla \
  | gzip -c \
  > "${f%.analysis.jsonl.gz}.patched.analysis.jsonl.gz"
  cp "${f}" nonpatched/
  mv "${f%.analysis.jsonl.gz}.patched.analysis.jsonl.gz" "${f}"
done
