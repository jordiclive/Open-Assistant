#!/bin/bash

while true; do
  # Find all folders containing "global_step" in the name recursively in /fsx/home-jordiclive
  find /fsx/home-jordiclive -type d -name "*global_step*" -print0 |
  # Delete each folder found
  while read -d $'\0' folder; do
    echo "Deleting $folder"
    rm -rf "$folder"
  done

  # Sleep for an hour
  sleep 3600
done
