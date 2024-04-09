#!/bin/bash

# Define an associative array with checkpoint names as keys and URLs as values
declare -A checkpoints=(
    ["SD+FT+TI"]="https://obj.umiacs.umd.edu/chopnlearn/checkpoints/SD%2BFT%2BTI.zip"
    ["SD+FT"]="https://obj.umiacs.umd.edu/chopnlearn/checkpoints/SD%2BFT.zip"
    ["SD+TI"]="https://obj.umiacs.umd.edu/chopnlearn/checkpoints/SD%2BTI.zip"
    ["classifier"]="https://obj.umiacs.umd.edu/chopnlearn/checkpoints/object_state_classifier_checkpoint.zip"
)

# Directory where the checkpoints will be extracted
extract_dir="checkpoints/"

# Create the directory if it doesn't exist
mkdir -p "$extract_dir"

# Function to download and extract a single checkpoint
download_extract() {
    local checkpoint_name="$1"
    local url="${checkpoints[$checkpoint_name]}"
    
    if [ -z "$url" ]; then
        echo "Checkpoint '$checkpoint_name' not found."
        exit 1
    fi

    echo "Downloading $checkpoint_name checkpoint from $url"
    wget -O "$checkpoint_name.zip" "$url" && \
    echo "Extracting $checkpoint_name.zip to $extract_dir" && \
    unzip "$checkpoint_name.zip" -d "$extract_dir" && \
    echo "$checkpoint_name extracted successfully." && \
    rm "$checkpoint_name.zip"
}

# Main logic to parse script arguments and handle actions
case "$1" in
    --all)
        for checkpoint_name in "${!checkpoints[@]}"; do
            download_extract "$checkpoint_name"
        done
        ;;
    *)
        download_extract "$1"
        ;;
esac
