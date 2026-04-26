from huggingface_hub import hf_hub_download
from omegaconf import DictConfig
from pathlib import Path
import skops.io as sio
import shutil
import os


class Inference:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg.inference
        self.model_dir = Path(self.cfg.paths.model_dir)
        self.model_path = self.model_dir / self.cfg.paths.model_name
        self.preprocessor_path = self.model_dir / self.cfg.paths.preprocessor_name
        self.repo_id = os.getenv("HF_REPO_ID", self.cfg.huggingface.repo_id)

    def fire(self):
        """Download artifacts from HF if not present, return loaded model + preprocessor."""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._download_if_missing(self.cfg.huggingface.model_name, self.model_path)
        self._download_if_missing(
            self.cfg.huggingface.preprocessor_name, self.preprocessor_path
        )
        return self._load()

    def _download_if_missing(self, filename: str, target_path: Path) -> None:
        if not target_path.exists():
            print(f"{filename} not found locally, downloading from Hugging Face...")
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                token=os.getenv("HF_TOKEN"),
            )
            shutil.copy(downloaded, target_path)

    def _load(self) -> dict:
        print("Loading model and preprocessor...")
        return {
            "model": sio.load(
                self.model_path, trusted=sio.get_untrusted_types(file=self.model_path)
            ),
            "preprocessor": sio.load(
                self.preprocessor_path,
                trusted=sio.get_untrusted_types(file=self.preprocessor_path),
            ),
        }
