for f in *.analysis.jsonl.gz
do
    echo -e "$f"
    python3 ~/train/topic/training/analysisToCatlibCsv.py -i $f
    echo
done
