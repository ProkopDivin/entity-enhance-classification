#!/bin/bash
set -euo pipefail

# Check if the directory path is provided

if [ $# -ne 2 ]; then
 
  echo "Usage: $0 <output_directory> <patches_directory>"


  exit 1

fi

# Get the output directory from the argument
output_dir="$1"
patches_dir="$2"

# Ensure the directory exists or create it
mkdir -p "$output_dir"

# Change to the output directory

cd "$output_dir"

echo "Mafra train"
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/cs_mafra/train/cs_mafra.train_all.analysis.json.gz | gzip > cs_mafra_iptc.train_all.analysis.jsonl.gz
zcat cs_mafra_iptc.train_all.analysis.jsonl.gz | wc -l
echo
echo "Mafra dev"
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/cs_mafra/dev/cs_mafra.dev_all.analysis.json.gz | gzip > cs_mafra_iptc.dev_all.analysis.jsonl.gz
zcat cs_mafra_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
echo "Mafra test"
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/cs_mafra/test/cs_mafra.test_all.analysis.json.gz | gzip > cs_mafra_iptc.test_all.analysis.jsonl.gz
zcat cs_mafra_iptc.test_all.analysis.jsonl.gz | wc -l
echo

csvcut -c id /data/corpora/media_corpus/public/de_dpa/train/de_dpa.train_small.csv | tail -n+2 > ids-dpa-train_small.csv
csvcut -c id /data/corpora/media_corpus/public/de_dpa/test/de_dpa.test_small.csv | tail -n+2 > ids-dpa-test_small.csv
csvcut -c id /data/corpora/media_corpus/public/de_dpa/dev/de_dpa.dev_small.csv | tail -n+2 > ids-dpa-dev_small.csv
cat "$patches_dir/patched_documents_2023_07_19.ids.tsv" ids-dpa-train_small.csv | sort -u > ids-de_dpa-train_small-plus-patched.tsv
cat "$patches_dir/patched_documents_2023_07_19.ids.tsv" ids-dpa-test_small.csv | sort -u > ids-de_dpa-test_small-plus-patched.tsv
cat "$patches_dir/patched_documents_2023_07_19.ids.tsv" ids-dpa-dev_small.csv | sort -u > ids-de_dpa-dev_small-plus-patched.tsv

echo "DPA train"
zcat /data/corpora/media_corpus/public/de_dpa/train/de_dpa.train_all.analysis.json.gz | python -m geneea.corpora.tools.getSubsetByIds --ids ids-de_dpa-train_small-plus-patched.tsv | gzip > de_dpa_iptc.train_smallpp.analysis.jsonl.gz
zcat de_dpa_iptc.train_smallpp.analysis.jsonl.gz | wc -l
echo
echo "DPA dev"
zcat /data/corpora/media_corpus/public/de_dpa/dev/de_dpa.dev_all.analysis.json.gz | python -m geneea.corpora.tools.getSubsetByIds --ids ids-de_dpa-dev_small-plus-patched.tsv | gzip > de_dpa_iptc.dev_smallpp.analysis.jsonl.gz
zcat de_dpa_iptc.dev_smallpp.analysis.jsonl.gz | wc -l
echo
echo "DPA test"
zcat /data/corpora/media_corpus/public/de_dpa/test/de_dpa.test_all.analysis.json.gz | python -m geneea.corpora.tools.getSubsetByIds --ids ids-de_dpa-test_small-plus-patched.tsv | gzip > de_dpa_iptc.test_smallpp.analysis.jsonl.gz
zcat de_dpa_iptc.test_smallpp.analysis.jsonl.gz | wc -l
echo

cp /data/corpora/media_corpus/public/de_apa/train/de_apa.train_all.analysis.json.gz de_apa.train_all.analysis.jsonl.gz
echo "APA train"
zcat de_apa.train_all.analysis.jsonl.gz | wc -l
echo
cp /data/corpora/media_corpus/public/de_apa/dev/de_apa.dev_all.analysis.json.gz de_apa.dev_all.analysis.jsonl.gz
echo "APA dev"
zcat de_apa.dev_all.analysis.jsonl.gz | wc -l
echo
cp /data/corpora/media_corpus/public/de_apa/test/de_apa.test_all.analysis.json.gz de_apa.test_all.analysis.jsonl.gz
echo "APA test"
zcat de_apa.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_bbc/train/en_bbc.train_all.analysis.json.gz | gzip > en_bbc_iptc.train_all.analysis.jsonl.gz
echo "BBC train"
zcat en_bbc_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_bbc/dev/en_bbc.dev_all.analysis.json.gz | gzip > en_bbc_iptc.dev_all.analysis.jsonl.gz
echo "BBC dev"
zcat en_bbc_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_bbc/test/en_bbc.test_all.analysis.json.gz | gzip > en_bbc_iptc.test_all.analysis.jsonl.gz
echo "BBC test"
zcat en_bbc_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_mediahuis/train/en_mediahuis.train_all.analysis.json.gz | gzip > en_mediahuis_iptc.train_all.analysis.jsonl.gz
echo "EN Mediahuis train"
zcat en_mediahuis_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_mediahuis/test/en_mediahuis.test_all.analysis.json.gz | gzip > en_mediahuis_iptc.test_all.analysis.jsonl.gz
echo "EN Mediahuis test"
zcat en_mediahuis_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep -P '"(rts_iptc_2022|iptc_2022)"' /data/corpora/media_corpus/public/fr_rts/train/fr_rts.train_all.analysis.json.gz | gzip > fr_rts_iptc.train_all.analysis.jsonl.gz
echo "FR RTS train"
zcat fr_rts_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep -P '"(rts_iptc_2022|iptc_2022)"' /data/corpora/media_corpus/public/fr_rts/dev/fr_rts.dev_all.analysis.json.gz | gzip > fr_rts_iptc.dev_all.analysis.jsonl.gz
echo "FR RTS dev"
zcat fr_rts_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep -P '"(rts_iptc_2022|iptc_2022)"' /data/corpora/media_corpus/public/fr_rts/test/fr_rts.test_all.analysis.json.gz | gzip > fr_rts_iptc.test_all.analysis.jsonl.gz
echo "FR RTS test"
zcat fr_rts_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep 'rts_feedback_iptc' /data/corpora/media_corpus/public/fr_rts_instagram/train/fr_rts_instagram.train_all.analysis.json.gz | gzip > fr_rts_instagram_iptc.train_all.analysis.jsonl.gz
echo "FR RTS-Instagram train"
zcat fr_rts_instagram_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep 'rts_feedback_iptc' /data/corpora/media_corpus/public/fr_rts_instagram/dev/fr_rts_instagram.dev_all.analysis.json.gz | gzip > fr_rts_instagram_iptc.dev_all.analysis.jsonl.gz
echo "FR RTS-Instagram dev"
zcat fr_rts_instagram_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep 'rts_feedback_iptc' /data/corpora/media_corpus/public/fr_rts_instagram/test/fr_rts_instagram.test_all.analysis.json.gz | gzip > fr_rts_instagram_iptc.test_all.analysis.jsonl.gz
echo "FR RTS-Instagram test"
zcat fr_rts_instagram_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/nl_mediahuis/train/nl_mediahuis.train_all.analysis.json.gz | gzip > nl_mediahuis_iptc.train_all.analysis.jsonl.gz
echo "NL Mediahuis train"
zcat nl_mediahuis_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/nl_mediahuis/test/nl_mediahuis.test_all.analysis.json.gz | gzip > nl_mediahuis_iptc.test_all.analysis.jsonl.gz
echo "NL Mediahuis test"
zcat nl_mediahuis_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_cnn/train/en_cnn.train_all.analysis.json.gz | gzip > en_cnn_iptc.train_all.analysis.jsonl.gz
echo "CNN train"
zcat en_cnn_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_cnn/dev/en_cnn.dev_all.analysis.json.gz | gzip > en_cnn_iptc.dev_all.analysis.jsonl.gz
echo "CNN dev"
zcat en_cnn_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_cnn/test/en_cnn.test_all.analysis.json.gz | gzip > en_cnn_iptc.test_all.analysis.jsonl.gz
echo "CNN test"
zcat en_cnn_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/de_eurosport/train/de_eurosport.train_all.analysis.json.gz | gzip > de_eurosport_iptc.train_all.analysis.jsonl.gz
echo "DE Eurosport train"
zcat de_eurosport_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/de_eurosport/dev/de_eurosport.dev_all.analysis.json.gz | gzip > de_eurosport_iptc.dev_all.analysis.jsonl.gz
echo "DE Eurosport dev"
zcat de_eurosport_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/de_eurosport/test/de_eurosport.test_all.analysis.json.gz | gzip > de_eurosport_iptc.test_all.analysis.jsonl.gz
echo "DE Eurosport test"
zcat de_eurosport_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/en_eurosport/train/en_eurosport.train_all.analysis.json.gz | gzip > en_eurosport_iptc.train_all.analysis.jsonl.gz
echo "EN Eurosport train"
zcat en_eurosport_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/en_eurosport/dev/en_eurosport.dev_all.analysis.json.gz | gzip > en_eurosport_iptc.dev_all.analysis.jsonl.gz
echo "EN Eurosport dev"
zcat en_eurosport_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/en_eurosport/test/en_eurosport.test_all.analysis.json.gz | gzip > en_eurosport_iptc.test_all.analysis.jsonl.gz
echo "EN Eurosport test"
zcat en_eurosport_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/es_eurosport/train/es_eurosport.train_all.analysis.json.gz | gzip > es_eurosport_iptc.train_all.analysis.jsonl.gz
echo "ES Eurosport train"
zcat es_eurosport_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/es_eurosport/dev/es_eurosport.dev_all.analysis.json.gz | gzip > es_eurosport_iptc.dev_all.analysis.jsonl.gz
echo "ES Eurosport dev"
zcat es_eurosport_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/es_eurosport/test/es_eurosport.test_all.analysis.json.gz | gzip > es_eurosport_iptc.test_all.analysis.jsonl.gz
echo "ES Eurosport test"
zcat es_eurosport_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/nl_eurosport/train/nl_eurosport.train_all.analysis.json.gz | gzip > nl_eurosport_iptc.train_all.analysis.jsonl.gz
echo "NL Eurosport train"
zcat nl_eurosport_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/nl_eurosport/dev/nl_eurosport.dev_all.analysis.json.gz | gzip > nl_eurosport_iptc.dev_all.analysis.jsonl.gz
echo "NL Eurosport dev"
zcat nl_eurosport_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/nl_eurosport/test/nl_eurosport.test_all.analysis.json.gz | gzip > nl_eurosport_iptc.test_all.analysis.jsonl.gz
echo "NL Eurosport test"
zcat nl_eurosport_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/fi_stt/train/fi_stt.train_medium.analysis.json.gz | gzip > fi_stt_iptc.train_medium.analysis.jsonl.gz
echo "FI STT train (medium)"
zcat fi_stt_iptc.train_medium.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/fi_stt/dev/fi_stt.dev_medium.analysis.json.gz | gzip > fi_stt_iptc.dev_medium.analysis.jsonl.gz
echo "FI STT dev (medium)"
zcat fi_stt_iptc.dev_medium.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022_auto"' /data/corpora/media_corpus/public/fi_stt/test/fi_stt.test_medium.analysis.json.gz | gzip > fi_stt_iptc.test_medium.analysis.jsonl.gz
echo "FI STT test (medium)"
zcat fi_stt_iptc.test_medium.analysis.jsonl.gz | wc -l
echo

cp /data/corpora/media_corpus/public/nl_eventdna/train/nl_eventdna.train_all.analysis.json.gz nl_eventdna.train_all.analysis.jsonl.gz
echo "NL Eventdna train"
zcat nl_eventdna.train_all.analysis.jsonl.gz | wc -l
echo
cp /data/corpora/media_corpus/public/nl_eventdna/dev/nl_eventdna.dev_all.analysis.json.gz nl_eventdna.dev_all.analysis.jsonl.gz
echo "NL Eventdna dev"
zcat nl_eventdna.dev_all.analysis.jsonl.gz | wc -l
echo
cp /data/corpora/media_corpus/public/nl_eventdna/test/nl_eventdna.test_all.analysis.json.gz nl_eventdna.test_all.analysis.jsonl.gz
echo "NL Eventdna test"
zcat nl_eventdna.test_all.analysis.jsonl.gz | wc -l
echo

zgrep -P '\["mediahuis"\]' /data/corpora/media_corpus/public/nl_noordhollandsdagblad/train/nl_noordhollandsdagblad.train_all.analysis.json.gz | gzip > nl_noordhollandsdagblad.train_all.analysis.jsonl.gz
echo "NL Noordhollandsdagblad train"
zcat nl_noordhollandsdagblad.train_all.analysis.jsonl.gz | wc -l
echo
zgrep -P '\["mediahuis"\]' /data/corpora/media_corpus/public/nl_noordhollandsdagblad/dev/nl_noordhollandsdagblad.dev_all.analysis.json.gz | gzip > nl_noordhollandsdagblad.dev_all.analysis.jsonl.gz
echo "NL Noordhollandsdagblad dev"
zcat nl_noordhollandsdagblad.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep -P '\["mediahuis"\]' /data/corpora/media_corpus/public/nl_noordhollandsdagblad/test/nl_noordhollandsdagblad.test_all.analysis.json.gz | gzip > nl_noordhollandsdagblad.test_all.analysis.jsonl.gz
echo "NL Noordhollandsdagblad test"
zcat nl_noordhollandsdagblad.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_dw/train/en_dw.train_all.analysis.json.gz | gzip > en_dw_iptc.train_all.analysis.jsonl.gz
echo "EN DW train"
zcat en_dw_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_dw/dev/en_dw.dev_all.analysis.json.gz | gzip > en_dw_iptc.dev_all.analysis.jsonl.gz
echo "EN DW dev"
zcat en_dw_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/en_dw/test/en_dw.test_all.analysis.json.gz | gzip > en_dw_iptc.test_all.analysis.jsonl.gz
echo "EN DW test"
zcat en_dw_iptc.test_all.analysis.jsonl.gz | wc -l
echo

zgrep '"iptc_2022"' /data/corpora/media_corpus/public/nl_nu/train/nl_nu.train_all.analysis.json.gz | gzip > nl_nu_iptc.train_all.analysis.jsonl.gz
echo "NL NU train"
zcat nl_nu_iptc.train_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/nl_nu/dev/nl_nu.dev_all.analysis.json.gz | gzip > nl_nu_iptc.dev_all.analysis.jsonl.gz
echo "NL NU dev"
zcat nl_nu_iptc.dev_all.analysis.jsonl.gz | wc -l
echo
zgrep '"iptc_2022"' /data/corpora/media_corpus/public/nl_nu/test/nl_nu.test_all.analysis.json.gz | gzip > nl_nu_iptc.test_all.analysis.jsonl.gz
echo "NL NU test"
zcat nl_nu_iptc.test_all.analysis.jsonl.gz | wc -l
echo
