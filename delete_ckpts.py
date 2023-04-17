import os
import fnmatch
import time

def delete_unwanted_files(directory, keep_files):
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            if filename not in keep_files:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def delete_everything(directory, keep_files):
    keep_files = ['added_tokens.json']
    for root, _, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            if filename not in keep_files:
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")



if __name__ == "__main__":
    search_directory = "/fsx/home-jordiclive/66B_checkpoints"
    keep_files = [
        "added_tokens.json",
        "tokenizer_config.json",
        "config.json",
        "pytorch_model.bin",
        "trainer_state.json",
        "generation_config.json",
        "special_tokens_map.json",
        "training_args.bin",
        "tokenizer.model"
    ]

    while True:
        print("Running delete_unwanted_files...")
        delete_unwanted_files(search_directory, keep_files)
        
        #todo remove
        delete_everything(search_directory)
        print("Waiting for 2 hours before the next run...")
        time.sleep(60 * 60)
