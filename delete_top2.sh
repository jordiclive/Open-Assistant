dir="/mnt/data/jordiclive/falcon/sft_falcon_ckpts"

while true; do
    # Find directories and sort by step number, keeping the two highest
    dirs_to_delete=$(find "$dir" -mindepth 1 -maxdepth 1 -type d -regex ".*/checkpoint-[0-9]*" | sort -t '-' -k2 -n | head -n -2)
    for folder in $dirs_to_delete; do
        echo "Deleting: $folder"
        rm -rf "$folder"
    done

    # Sleep for 45 minutes
    sleep 2700
done
