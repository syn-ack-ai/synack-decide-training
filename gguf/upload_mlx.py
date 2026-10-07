from huggingface_hub import HfApi
api = HfApi()
r = "SkyPanther/synack-decide-26b-a4b-mlx-8bit"
api.create_repo(r, private=True, exist_ok=True)
api.upload_large_folder(repo_id=r, folder_path="models/systemone-v3-mlx8", repo_type="model")
print("MLX_UPLOAD_DONE")
