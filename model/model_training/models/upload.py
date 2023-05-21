from huggingface_hub import HfApi, create_repo

def upload_adapter(repo_id,folder_path):
    create_repo(repo_id)
    api = HfApi()
    api.upload_folder(

        folder_path=folder_path,

        path_in_repo=".",

        repo_id=repo_id,

        repo_type="model",

    )


if __name__ == '__main__':
    upload_adapter("jordiclive/adapter_ckpt_1500")