"""
Moltbot Model Downloader
========================
Downloads required GGUF models from HuggingFace.
"""

import os
import sys
import argparse
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError
import hashlib

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Model definitions
MODELS = {
    "forecaster": {
        "name": "Qwen2.5-7B-Instruct Q5_K_M",
        "filename": "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "size_gb": 5.44,
        "sha256": None
    },
    "coder": {
        "name": "Qwen2.5-Coder-7B-Instruct Q5_K_M",
        "filename": "Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf",
        "url": "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q5_K_M.gguf",
        "size_gb": 5.44,
        "sha256": None
    },
    "reasoning": {
        "name": "Qwen3-14B Q4_K_M",
        "filename": "Qwen3-14B-Q4_K_M.gguf",
        "url": "https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf",
        "size_gb": 9.0,
        "sha256": None
    }
}


def format_size(bytes_size: int) -> str:
    """Format bytes to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


class DownloadProgress:
    """Progress reporter for downloads."""

    def __init__(self, filename: str, total_size: float):
        self.filename = filename
        self.total_size = total_size
        self.downloaded = 0
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        self.downloaded = block_num * block_size
        if total_size > 0:
            percent = int(100 * self.downloaded / total_size)
        else:
            percent = int(100 * self.downloaded / (self.total_size * 1024**3))

        if percent != self.last_percent:
            self.last_percent = percent
            bar_len = 40
            filled = int(bar_len * percent / 100)
            bar = '=' * filled + '-' * (bar_len - filled)
            print(f"\r  [{bar}] {percent}% - {format_size(self.downloaded)}", end='', flush=True)


def download_model(model_key: str, models_dir: Path, force: bool = False) -> bool:
    """Download a model."""
    if model_key not in MODELS:
        print(f"Error: Unknown model '{model_key}'")
        print(f"Available: {', '.join(MODELS.keys())}")
        return False

    model = MODELS[model_key]
    output_path = models_dir / model['filename']

    print(f"\n{'='*60}")
    print(f"Model: {model['name']}")
    print(f"Size: ~{model['size_gb']} GB")
    print(f"File: {output_path}")
    print(f"{'='*60}")

    # Check if exists
    if output_path.exists() and not force:
        print(f"Already downloaded. Use --force to re-download.")
        return True

    # Create directory
    models_dir.mkdir(parents=True, exist_ok=True)

    # Download
    print(f"\nDownloading from HuggingFace...")
    print(f"URL: {model['url']}")

    try:
        progress = DownloadProgress(model['filename'], model['size_gb'])
        urlretrieve(model['url'], output_path, reporthook=progress)
        print()  # Newline after progress bar
    except URLError as e:
        print(f"\nDownload failed: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nDownload cancelled.")
        if output_path.exists():
            output_path.unlink()
        return False

    # Verify checksum if available
    if model.get('sha256'):
        print("Verifying checksum...")
        sha256 = hashlib.sha256()
        with open(output_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        if sha256.hexdigest() != model['sha256']:
            print("Checksum mismatch! File may be corrupted.")
            return False
        print("Checksum OK.")

    print(f"\nDownloaded: {output_path}")
    print(f"Size: {format_size(output_path.stat().st_size)}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download Moltbot models")
    parser.add_argument(
        'models',
        nargs='*',
        default=['all'],
        help="Models to download: forecaster, coder, or all (default: all)"
    )
    parser.add_argument(
        '--models-dir',
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory to save models"
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Re-download even if file exists"
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help="List available models"
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    if args.list:
        print("\nAvailable models:")
        for key, model in MODELS.items():
            print(f"  {key}: {model['name']} (~{model['size_gb']} GB)")
        return

    # Determine which models to download
    if 'all' in args.models:
        to_download = list(MODELS.keys())
    else:
        to_download = args.models

    print(f"\nMoltbot Model Downloader")
    print(f"Models to download: {', '.join(to_download)}")
    print(f"Target directory: {args.models_dir}")

    total_size = sum(MODELS[m]['size_gb'] for m in to_download if m in MODELS)
    print(f"Total download size: ~{total_size:.1f} GB")

    # Confirm
    if not args.yes:
        response = input("\nContinue? [Y/n] ").strip().lower()
        if response and response != 'y':
            print("Cancelled.")
            return

    # Download each model
    success = True
    for model_key in to_download:
        if not download_model(model_key, args.models_dir, args.force):
            success = False

    if success:
        print(f"\n{'='*60}")
        print("All models downloaded successfully!")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("Some downloads failed. Check errors above.")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
