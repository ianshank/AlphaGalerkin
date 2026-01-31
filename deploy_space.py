"""Deployment script for AlphaGalerkin HF Space."""

import os

from huggingface_hub import HfApi, login


def deploy() -> None:
    """Deploy AlphaGalerkin demo to Hugging Face Spaces."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    print("Logging in with token...")
    login(token=token)

    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    print(f"Authenticated as: {username}")

    repo_id = f"{username}/alphagalerkin-demo"
    print(f"Target Repo: {repo_id}")

    try:
        # Create or get repo
        url = api.create_repo(
            repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True, private=False
        )
        print(f"Repo ready at: {url}")

        # Upload folder
        print("Uploading files...")
        api.upload_folder(
            folder_path="hf_space",
            repo_id=repo_id,
            repo_type="space",
            path_in_repo=".",
            ignore_patterns=[".git", "__pycache__"],
        )
        print("Upload complete!")
        print(f"Check your space at: https://huggingface.co/spaces/{repo_id}")

    except Exception as e:
        print(f"Deployment failed: {e}")


if __name__ == "__main__":
    deploy()
