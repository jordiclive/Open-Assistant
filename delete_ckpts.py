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

# def delete_everything(directory, keep_files):
#     keep_files = ['added_tokens.json']
#     for root, _, filenames in os.walk(directory):
#         for filename in fnmatch.filter(filenames, '*'):
#             if filename not in keep_files:
#                 file_path = os.path.join(root, filename)
#                 try:
#                     os.remove(file_path)
#                     print(f"Deleted: {file_path}")
#                 except Exception as e:
#                     print(f"Error deleting {file_path}: {e}")

import os
import glob
import time

def find_pytorch_model_bin_files(search_directory):
    files = []
    for root, _, filenames in os.walk(search_directory):
        for filename in filenames:
            if filename == "pytorch_model.bin":
                files.append(os.path.join(root, filename))
    return files

def delete_old_pytorch_model_bin_files(files):
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for file in files[1:]:
        try:
            os.remove(file)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")




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

        print("Searching for pytorch_model.bin files...")
        pytorch_model_bin_files = find_pytorch_model_bin_files(search_directory)
        print("Deleting old pytorch_model.bin files...")
        delete_old_pytorch_model_bin_files(pytorch_model_bin_files)

        print("Waiting for 2 hours before the next run...")
        #todo remove
        # delete_everything(search_directory)
        print("Waiting for 2 hours before the next run...")
        time.sleep(60 * 60)
