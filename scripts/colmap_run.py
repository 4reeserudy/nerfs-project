"""COLMAP pipeline orchestration script."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional


class ColmapPipeline:
    """High-level orchestrator for running COLMAP pipelines."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def run(self, scene: str, *, overwrite: bool = False) -> None:
        """Execute the full COLMAP sparse reconstruction pipeline."""
        raise NotImplementedError

    def _prepare_scene(self, scene: str) -> None:
        """Validate input scene assets and prepare workspace."""
        raise NotImplementedError

    def _run_feature_extraction(self, scene: str) -> None:
        """Run COLMAP feature extraction for the provided scene."""
        raise NotImplementedError

    def _run_feature_matching(self, scene: str) -> None:
        """Run COLMAP feature matching for the provided scene."""
        raise NotImplementedError

    def _run_sparse_reconstruction(self, scene: str) -> None:
        """Execute sparse reconstruction and bundle adjustment."""
        raise NotImplementedError

    def _export_results(self, scene: str, outputs: Iterable[str]) -> None:
        """Export COLMAP outputs for downstream NeRF consumption."""
        raise NotImplementedError


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
        help="Subset of COLMAP outputs to export.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """CLI entry point for the COLMAP pipeline script."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
