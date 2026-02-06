"""
Moltbot HuggingFace Integration
===============================
Search and download GGUF models from HuggingFace Hub.
"""

import os
import re
import logging
import shutil
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from huggingface_hub import HfApi, hf_hub_download, list_repo_files

logger = logging.getLogger(__name__)


@dataclass
class GGUFModelInfo:
    """Information about a GGUF model file."""
    repo_id: str
    filename: str
    size_bytes: Optional[int] = None
    quantization: Optional[str] = None

    @property
    def size_gb(self) -> Optional[float]:
        if self.size_bytes:
            return self.size_bytes / (1024 ** 3)
        return None


@dataclass
class SearchResult:
    """Result from a HuggingFace model search."""
    repo_id: str
    author: str
    model_name: str
    downloads: int
    likes: int
    tags: list
    gguf_files: list


class HuggingFaceClient:
    """Client for interacting with HuggingFace Hub."""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        self.api = HfApi(token=self.token)

    def search_gguf_models(
        self,
        query: str,
        limit: int = 10,
        min_downloads: int = 0
    ) -> list:
        """
        Search for GGUF models on HuggingFace.

        Args:
            query: Search query (e.g., "Qwen 7B GGUF")
            limit: Maximum number of results
            min_downloads: Minimum download count filter

        Returns:
            List of SearchResult objects with GGUF file info
        """
        logger.info(f"Searching HuggingFace for: {query}")

        try:
            models = self.api.list_models(
                search=query,
                filter="gguf",
                sort="downloads",
                limit=limit * 3
            )

            results = []
            for model in models:
                if len(results) >= limit:
                    break

                downloads = getattr(model, 'downloads', 0) or 0
                if downloads < min_downloads:
                    continue

                try:
                    gguf_files = self._get_gguf_files(model.id)
                    if not gguf_files:
                        continue

                    author = getattr(model, 'author', None) or model.id.split("/")[0]
                    likes = getattr(model, 'likes', 0) or 0
                    tags = getattr(model, 'tags', []) or []

                    results.append(SearchResult(
                        repo_id=model.id,
                        author=author,
                        model_name=model.id.split("/")[-1],
                        downloads=downloads,
                        likes=likes,
                        tags=tags,
                        gguf_files=gguf_files
                    ))
                except Exception as e:
                    logger.debug(f"Skipping {model.id}: {e}")
                    continue

            logger.info(f"Found {len(results)} GGUF model repos")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def _get_gguf_files(self, repo_id: str) -> list:
        """Get list of GGUF files in a repository."""
        try:
            files = list_repo_files(repo_id, token=self.token)
            gguf_files = []

            for f in files:
                if f.endswith(".gguf"):
                    quant = self._extract_quantization(f)
                    gguf_files.append(GGUFModelInfo(
                        repo_id=repo_id,
                        filename=f,
                        quantization=quant
                    ))

            return gguf_files
        except Exception as e:
            logger.debug(f"Failed to list files in {repo_id}: {e}")
            return []

    def _extract_quantization(self, filename: str) -> Optional[str]:
        """Extract quantization level from filename."""
        match = re.search(r'(Q[0-9]+_[A-Z0-9_]+|IQ[0-9]+_[A-Z]+)', filename, re.IGNORECASE)
        return match.group(1).upper() if match else None

    def download_model(
        self,
        repo_id: str,
        filename: str,
        local_dir: Path,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """
        Download a GGUF model file from HuggingFace.

        Args:
            repo_id: Repository ID (e.g., "bartowski/Qwen2.5-7B-Instruct-GGUF")
            filename: GGUF file name within the repo
            local_dir: Directory to save the file
            progress_callback: Optional callback(downloaded_bytes, total_bytes)

        Returns:
            Path to the downloaded file
        """
        logger.info(f"Downloading {filename} from {repo_id}")

        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        final_path = local_dir / Path(filename).name

        if final_path.exists():
            logger.info(f"File already exists: {final_path}")
            return final_path

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=self.token
        )

        downloaded_path = Path(downloaded_path)
        if downloaded_path != final_path:
            if downloaded_path.exists():
                shutil.move(str(downloaded_path), str(final_path))

        logger.info(f"Downloaded to {final_path}")
        return final_path

    def get_model_info(self, repo_id: str, filename: str) -> Optional[GGUFModelInfo]:
        """Get detailed info about a specific GGUF file."""
        try:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            url = hf_hub_url(repo_id, filename)
            metadata = get_hf_file_metadata(url, token=self.token)

            return GGUFModelInfo(
                repo_id=repo_id,
                filename=filename,
                size_bytes=metadata.size,
                quantization=self._extract_quantization(filename)
            )
        except Exception as e:
            logger.debug(f"Failed to get metadata: {e}")
            return None
