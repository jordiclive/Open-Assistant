#!/bin/bash

dir="/fsx/home-jordiclive/peft_models/"

while true; do
    # Find and delete directories containing "global_step"
    find "$dir" -type d -name "*global_step*" -exec rm -rf {} +

    # Find and delete files ending with ".pt"
    find "$dir" -type f -name "*.pt" -exec rm -f {} +

    # Sleep for 45 minutes
    sleep 2700
done
