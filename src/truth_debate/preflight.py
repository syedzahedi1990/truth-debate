from __future__ import annotations

import socket
import sys
import urllib.request
from pathlib import Path
from typing import Any


def run_preflight(cfg: dict[str, Any]) -> None:
    print("Python:", sys.version.replace("\n", " "))
    _check_torch()
    _check_network("huggingface.co", 443)
    _check_hf_metadata(str(cfg["model"]["name"]))
    print("Preflight OK.")


def download_model(cfg: dict[str, Any], local_dir: str | Path | None = None) -> str:
    from huggingface_hub import snapshot_download

    model_name = str(cfg["model"]["name"])
    kwargs: dict[str, Any] = {}
    if local_dir is not None:
        kwargs["local_dir"] = str(local_dir)
    path = snapshot_download(repo_id=model_name, **kwargs)
    print(f"Downloaded {model_name} to {path}")
    return str(path)


def _check_torch() -> None:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("PyTorch is not importable. Run scripts/bootstrap_vast.sh first.") from exc
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))


def _check_network(host: str, port: int) -> None:
    try:
        with socket.create_connection((host, port), timeout=10):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Cannot reach {host}:{port}. The machine/container has no outbound internet route. "
            "On Vast.ai, rent or restart an instance with public internet enabled, or pre-cache the model "
            "and set model.name to the local model directory."
        ) from exc
    print(f"Network: reachable {host}:{port}")


def _check_hf_metadata(model_name: str) -> None:
    url = f"https://huggingface.co/{model_name}/resolve/main/config.json"
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=15) as response:
            print(f"Hugging Face metadata: HTTP {response.status}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch Hugging Face metadata for {model_name}. "
            "Check outbound networking and HF_TOKEN for gated models."
        ) from exc
