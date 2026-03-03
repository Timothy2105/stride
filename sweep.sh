#!/bin/bash

# Sweep script to run STRIDE on multiple D4RL tasks
set -e

TASKS=(
    "D4RL/pen/human-v2"
    "D4RL/door/human-v2"
    "D4RL/hammer/human-v2"
    "D4RL/relocate/human-v2"
)

# Use smoke-test if passed as an argument
SMOKE_TEST=""
if [[ "$1" == "--smoke-test" ]]; then
    SMOKE_TEST="--smoke-test"
    echo "Running in smoke-test mode..."
fi

for TASK in "${TASKS[@]}"; do
    echo "================================================================"
    echo "Running STRIDE for task: $TASK"
    echo "================================================================"
    
    python experiments/run_all.py --task "$TASK" $SMOKE_TEST
    
    TASK_SLUG=$(echo "$TASK" | tr '/' '_')
    python experiments/plot_results.py --results "results/${TASK_SLUG}_results.json"
    
    echo "Finished task: $TASK"
    echo ""
done

echo "All tasks in sweep completed."
