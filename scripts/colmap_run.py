"""COLMAP pipeline orchestration script."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneWorkspace:
    """Resolved paths used throughout the COLMAP pipeline."""

    scene: str
    scene_root: Path
    images_dir: Path
    colmap_root: Path
    database_path: Path
    reconstruction_dir: Path
    dense_root: Path
    undistorted_dir: Path
    text_output_dir: Path


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """Configuration flags for COLMAP feature extraction."""

    use_gpu: bool = True
    max_image_size: Optional[int] = None
    camera_model: Optional[str] = None
    single_camera: bool = False


@dataclass(frozen=True)
class MatchingConfig:
    """Configuration for COLMAP feature matching."""

    matcher: str = "exhaustive"
    sequential_overlap: int = 5


class ColmapPipeline:
    """High-level orchestrator for running COLMAP pipelines."""

    def __init__(
        self,
        data_root: Path,
        *,
        colmap_path: str = "colmap",
        feature_config: FeatureExtractionConfig | None = None,
        matching_config: MatchingConfig | None = None,
    ) -> None:
        self.data_root = data_root
        self.colmap_path = colmap_path
        self.feature_config = feature_config or FeatureExtractionConfig()
        self.matching_config = matching_config or MatchingConfig()
        self._colmap_checked = False

    def run(
        self,
        scene: str,
        *,
        overwrite: bool = False,
        exports: Optional[Iterable[str]] = None,
    ) -> None:
        """Execute the full COLMAP sparse reconstruction pipeline."""
        workspace = self._scene_workspace(scene)
        exports = tuple(exports or ())

        _LOGGER.info("Running COLMAP pipeline for scene '%s'", scene)
        self._prepare_scene(workspace, overwrite=overwrite)
        self._run_feature_extraction(workspace)
        self._run_feature_matching(workspace)
        self._run_sparse_reconstruction(workspace)

        if exports:
            self._export_results(workspace, exports)

    def _prepare_scene(self, workspace: SceneWorkspace, *, overwrite: bool) -> None:
        """Validate input scene assets and prepare workspace."""
        if not workspace.scene_root.exists():
            raise FileNotFoundError(
                f"Scene '{workspace.scene}' not found under {self.data_root}."
            )

        if not workspace.images_dir.exists():
            raise FileNotFoundError(
                f"Scene '{workspace.scene}' is missing an images/ directory."
            )

        if not any(workspace.images_dir.iterdir()):
            raise RuntimeError(
                f"Scene '{workspace.scene}' has no images to process in {workspace.images_dir}."
            )

        workspace.colmap_root.mkdir(parents=True, exist_ok=True)

        if overwrite:
            if workspace.database_path.exists():
                workspace.database_path.unlink()
            if workspace.reconstruction_dir.exists():
                shutil.rmtree(workspace.reconstruction_dir)
            if workspace.undistorted_dir.exists():
                shutil.rmtree(workspace.undistorted_dir)
            if workspace.dense_root.exists():
                shutil.rmtree(workspace.dense_root)
            if workspace.text_output_dir.exists():
                shutil.rmtree(workspace.text_output_dir)

        workspace.colmap_root.mkdir(parents=True, exist_ok=True)
        workspace.dense_root.mkdir(parents=True, exist_ok=True)
        workspace.reconstruction_dir.mkdir(parents=True, exist_ok=True)

    def _run_feature_extraction(self, workspace: SceneWorkspace) -> None:
        """Run COLMAP feature extraction for the provided scene."""
        _LOGGER.info("[COLMAP] Feature extraction")
        cfg = self.feature_config
        command = [
            "feature_extractor",
            "--database_path",
            str(workspace.database_path),
            "--image_path",
            str(workspace.images_dir),
        ]
        if not cfg.use_gpu:
            command.extend(["--SiftExtraction.use_gpu", "0"])
        if cfg.max_image_size is not None:
            command.extend(["--SiftExtraction.max_image_size", str(cfg.max_image_size)])
        if cfg.camera_model is not None:
            command.extend(["--ImageReader.camera_model", cfg.camera_model])
        if cfg.single_camera:
            command.extend(["--ImageReader.single_camera", "1"])
        self._run_colmap_command(command)

    def _run_feature_matching(self, workspace: SceneWorkspace) -> None:
        """Run COLMAP feature matching for the provided scene."""
        cfg = self.matching_config
        matcher = cfg.matcher.lower()
        if matcher not in {"exhaustive", "sequential"}:
            raise ValueError(
                "Unsupported matcher '%s'. Expected 'exhaustive' or 'sequential'."
                % cfg.matcher
            )

        _LOGGER.info("[COLMAP] Feature matching (%s)", matcher)
        command = [
            f"{matcher}_matcher",
            "--database_path",
            str(workspace.database_path),
        ]
        if matcher == "sequential":
            command.extend(
                ["--SequentialMatching.overlap", str(max(cfg.sequential_overlap, 1))]
            )
        self._run_colmap_command(command)

    def _run_sparse_reconstruction(self, workspace: SceneWorkspace) -> None:
        """Execute sparse reconstruction and bundle adjustment."""
        _LOGGER.info("[COLMAP] Sparse reconstruction")
        command = [
            "mapper",
            "--database_path",
            str(workspace.database_path),
            "--image_path",
            str(workspace.images_dir),
            "--output_path",
            str(workspace.reconstruction_dir),
        ]
        self._run_colmap_command(command)

    def _export_results(
        self, workspace: SceneWorkspace, outputs: Iterable[str]
    ) -> None:
        """Export COLMAP outputs for downstream NeRF consumption."""
        export_set = {output.lower() for output in outputs}
        model_dir = self._locate_model_dir(workspace.reconstruction_dir)

        if {"sparse", "text"} & export_set:
            _LOGGER.info("[COLMAP] Exporting sparse model to text format")
            workspace.text_output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "model_converter",
                "--input_path",
                str(model_dir),
                "--output_path",
                str(workspace.text_output_dir),
                "--output_type",
                "TXT",
            ]
            self._run_colmap_command(command)

        if "images" in export_set:
            _LOGGER.info("[COLMAP] Undistorting images")
            workspace.undistorted_dir.mkdir(parents=True, exist_ok=True)
            command = [
                "image_undistorter",
                "--image_path",
                str(workspace.images_dir),
                "--input_path",
                str(model_dir),
                "--output_path",
                str(workspace.undistorted_dir),
                "--output_type",
                "COLMAP",
            ]
            self._run_colmap_command(command)

    def _run_colmap_command(self, command: Sequence[str]) -> None:
        """Run a COLMAP subcommand, ensuring the binary exists."""
        self._ensure_colmap_available()
        full_command = [self.colmap_path, *command]
        _LOGGER.debug("Executing command: %s", " ".join(full_command))
        subprocess.run(full_command, check=True)

    def _ensure_colmap_available(self) -> None:
        if self._colmap_checked:
            return
        if shutil.which(self.colmap_path) is None:
            raise FileNotFoundError(
                f"COLMAP executable '{self.colmap_path}' not found in PATH. "
                "Provide the correct path with --colmap-path."
            )
        self._colmap_checked = True

    def _locate_model_dir(self, reconstruction_dir: Path) -> Path:
        """Locate the COLMAP sparse model directory within the workspace."""
        direct_model = reconstruction_dir / "cameras.bin"
        if direct_model.exists():
            return reconstruction_dir

        candidates = sorted(p for p in reconstruction_dir.iterdir() if p.is_dir())
        for candidate in candidates:
            if (candidate / "cameras.bin").exists():
                return candidate

        raise FileNotFoundError(
            "No COLMAP model was found after reconstruction. Expected cameras.bin "
            f"under {reconstruction_dir}."
        )

    def _scene_workspace(self, scene: str) -> SceneWorkspace:
        """Resolve workspace paths for the requested scene."""
        scene_root = self.data_root / scene
        return SceneWorkspace(
            scene=scene,
            scene_root=scene_root,
            images_dir=scene_root / "images",
            colmap_root=scene_root / "colmap",
            database_path=scene_root / "colmap" / "database.db",
            reconstruction_dir=scene_root / "colmap" / "sparse",
            dense_root=scene_root / "dense",
            undistorted_dir=scene_root / "dense" / "images",
            text_output_dir=scene_root / "sparse",
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for COLMAP pipeline execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="Name of the scene folder under data/")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory containing scene assets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run pipeline even if outputs already exist.",
    )
    parser.add_argument(
        "--exports",
        nargs="*",
        default=("sparse", "images"),
        help=(
            "Subset of COLMAP outputs to export (choices: sparse, text, images). "
            "Defaults to sparse and images."
        ),
    )
    parser.add_argument(
        "--colmap-path",
        default="colmap",
        help="Path to the COLMAP executable.",
    )
    parser.add_argument(
        "--feature-no-gpu",
        action="store_true",
        help="Disable GPU usage during feature extraction.",
    )
    parser.add_argument(
        "--feature-max-image-size",
        type=int,
        default=None,
        help="Optional maximum image size for SIFT extraction.",
    )
    parser.add_argument(
        "--feature-camera-model",
        default=None,
        help="Override the COLMAP camera model (e.g. PINHOLE, SIMPLE_RADIAL).",
    )
    parser.add_argument(
        "--feature-single-camera",
        action="store_true",
        help="Force COLMAP to use a shared camera across all images.",
    )
    parser.add_argument(
        "--matcher",
        choices=("exhaustive", "sequential"),
        default="exhaustive",
        help="Feature matcher strategy to use.",
    )
    parser.add_argument(
        "--sequential-overlap",
        type=int,
        default=5,
        help="Number of neighbouring frames to match when using sequential matcher.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g. DEBUG, INFO, WARNING).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for the COLMAP pipeline script."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s: %(message)s")

    feature_config = FeatureExtractionConfig(
        use_gpu=not args.feature_no_gpu,
        max_image_size=args.feature_max_image_size,
        camera_model=args.feature_camera_model,
        single_camera=args.feature_single_camera,
    )
    matching_config = MatchingConfig(
        matcher=args.matcher,
        sequential_overlap=args.sequential_overlap,
    )

    pipeline = ColmapPipeline(
        args.data_root,
        colmap_path=args.colmap_path,
        feature_config=feature_config,
        matching_config=matching_config,
    )
    pipeline.run(
        args.scene,
        overwrite=args.overwrite,
        exports=args.exports,
    )


if __name__ == "__main__":
    main()
