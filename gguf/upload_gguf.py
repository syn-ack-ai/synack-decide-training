from huggingface_hub import HfApi
api = HfApi()
r = "SkyPanther/synack-decide-26b-a4b-GGUF"
api.create_repo(r, private=True, exist_ok=True)
api.upload_large_folder(repo_id=r, folder_path="models/gguf-upload", repo_type="model")
print("GGUF_UPLOAD_DONE")
