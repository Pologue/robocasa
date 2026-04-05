#!/bin/bash

# Quick start script for RoboCasa prefix segment generation
# Usage: bash scripts/dataset_scripts/quickstart_prefix_segments.sh <outputs_dir> <dataset_path> <num_segments> [num_workers] [task]

set -e

function show_help() {
    echo "RoboCasa Prefix Segment Generation Quick Start"
    echo ""
    echo "Usage: bash quickstart_prefix_segments.sh <outputs_dir> <dataset_path> <num_segments> [num_workers] [task]"
    echo ""
    echo "Arguments:"
    echo "  outputs_dir    Path to outputs directory containing affordance annotations"
    echo "  dataset_path   Path to LeRobot dataset root"
    echo "  num_segments   Number of video segments (e.g., 4)"
    echo "  num_workers    (Optional) Number of parallel workers (default: 8)"
    echo "  task           (Optional) Process only specific task (default: all)"
    echo ""
    echo "Example:"
    echo "  bash quickstart_prefix_segments.sh ./outputs /data/lerobot_dataset 4 8"
    echo "  bash quickstart_prefix_segments.sh ./outputs /data/lerobot_dataset 3 4 DeliverStraw"
    echo ""
}

if [[ $# -lt 3 ]]; then
    show_help
    exit 1
fi

OUTPUTS_DIR="$1"
DATASET_PATH="$2"
NUM_SEGMENTS=$3
NUM_WORKERS=${4:-8}
TASK=${5:-}

# Verify paths
if [[ ! -d "$OUTPUTS_DIR" ]]; then
    echo "Error: outputs_dir not found: $OUTPUTS_DIR"
    exit 1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
    echo "Error: dataset_path not found: $DATASET_PATH"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate conda environment
echo "Activating conda environment..."
source /home/pologue/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

cd "$PROJECT_ROOT"

echo "=========================================="
echo "RoboCasa Prefix Segment Generation"
echo "=========================================="
echo "Outputs directory:  $OUTPUTS_DIR"
echo "Dataset path:       $DATASET_PATH"
echo "Number of segments: $NUM_SEGMENTS"
echo "Workers:            $NUM_WORKERS"
if [[ -n "$TASK" ]]; then
    echo "Task filter:        $TASK"
fi
echo "=========================================="
echo ""

# Step 1: Generate segment metadata
echo "[Step 1/3] Generating segment metadata..."
if [[ -n "$TASK" ]]; then
    python "$SCRIPT_DIR/generate_prefix_segments.py" \
        --outputs-dir "$OUTPUTS_DIR" \
        --num-segments "$NUM_SEGMENTS" \
        --task "$TASK" \
        --num-workers "$NUM_WORKERS" \
        --verbose
else
    python "$SCRIPT_DIR/generate_prefix_segments.py" \
        --outputs-dir "$OUTPUTS_DIR" \
        --num-segments "$NUM_SEGMENTS" \
        --num-workers "$NUM_WORKERS" \
        --verbose
fi

echo ""
echo "[Step 1/3] ✓ Segment metadata generated"
echo ""

# Step 2: Generate video files
echo "[Step 2/3] Generating prefix videos..."
if [[ -n "$TASK" ]]; then
    python "$SCRIPT_DIR/generate_prefix_videos.py" \
        --dataset "$DATASET_PATH" \
        --outputs-dir "$OUTPUTS_DIR" \
        --num-segments "$NUM_SEGMENTS" \
        --task "$TASK" \
        --num-workers "$NUM_WORKERS" \
        --verbose
else
    python "$SCRIPT_DIR/generate_prefix_videos.py" \
        --dataset "$DATASET_PATH" \
        --outputs-dir "$OUTPUTS_DIR" \
        --num-segments "$NUM_SEGMENTS" \
        --num-workers "$NUM_WORKERS" \
        --verbose
fi

echo ""
echo "[Step 2/3] ✓ Videos generated"
echo ""

# Step 3: Generate evaluation manifest
echo "[Step 3/3] Generating evaluation manifest..."
MANIFEST_PATH="${OUTPUTS_DIR}/../vlm_eval_manifest_${NUM_SEGMENTS}seg.jsonl"
python "$SCRIPT_DIR/generate_vlm_eval_manifest.py" \
    manifest \
    --outputs-dir "$OUTPUTS_DIR" \
    --num-segments "$NUM_SEGMENTS" \
    --output "$MANIFEST_PATH"

echo ""
echo "[Step 3/3] ✓ Manifest generated: $MANIFEST_PATH"
echo ""

# Print statistics
echo "[Info] Dataset statistics:"
python "$SCRIPT_DIR/generate_vlm_eval_manifest.py" \
    stats \
    --outputs-dir "$OUTPUTS_DIR" \
    --num-segments "$NUM_SEGMENTS"

echo ""
echo "=========================================="
echo "✓ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Output manifest: $MANIFEST_PATH"
echo "Ready for VLM evaluation!"
