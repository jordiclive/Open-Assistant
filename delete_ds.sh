dir="/mnt/data/jordiclive/falcon/sft_falcon_ckpts"

while true; do
    # Find directories and sort by step number, keeping the two highest
    dirs_to_delete=$(find "$dir" -type d -regex ".*/global_step[0-9]*" | sort -t '_' -k2 -n | head -n -2)
    for folder in $dirs_to_delete; do
        echo "Deleting: $folder"
        rm -rf "$folder"
    done

    # Sleep for 45 minutes
    sleep 1000
done
