#!/bin/bash

# Script to add entities to all all-corpora-* CSV files in specified directories
# Outputs files with entities_ prefix

# Directories to process
DIRECTORIES=(
    "chrono-corpora-global"
    "chrono-corpora-per-dataset"
    "origin-corpora"
    "origin-corpora-filtred"
)

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENTITIES_SCRIPT="${SCRIPT_DIR}/add_entities_to_csv.py"

# Check if the Python script exists
if [ ! -f "$ENTITIES_SCRIPT" ]; then
    echo "Error: add_entities_to_csv.py not found at $ENTITIES_SCRIPT" >&2
    exit 1
fi

# Process each directory
for dir in "${DIRECTORIES[@]}"; do
    # Check if directory exists
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory '$dir' does not exist, skipping..." >&2
        continue
    fi

    echo "Processing directory: $dir"

    # Find entities.tsv file in this directory
    ENTITIES_FILE="${dir}/entities.tsv"
    if [ ! -f "$ENTITIES_FILE" ]; then
        echo "  Warning: entities.tsv not found in $dir, skipping this directory..." >&2
        continue
    fi

    # Find all files starting with all-corpora-*.csv
    find "$dir" -maxdepth 1 -name "all-corpora-*.csv" -type f | while read -r csv_file; do
        # Get filename without path
        filename=$(basename "$csv_file")
        
        # Create output filename with entities_ prefix
        output_file="${dir}/entities_${filename}"

        echo "  Processing: $filename"
        echo "    Input:  $csv_file"
        echo "    Output: $output_file"

        # Run the Python script
        python3 "$ENTITIES_SCRIPT" "$csv_file" "$ENTITIES_FILE" -o "$output_file"

        if [ $? -eq 0 ]; then
            echo "    ✓ Success"
        else
            echo "    ✗ Error processing $filename" >&2
        fi
        echo
    done

    echo "Completed processing directory: $dir"
    echo
done

echo "Done processing all directories!"

