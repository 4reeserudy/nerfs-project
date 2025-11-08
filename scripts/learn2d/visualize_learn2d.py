"""
visualize_learn2d.py

Utility script to generate visualizations for Learn2D runs.
- Progress grids: show evolution over epochs for a single run
- Final comparisons: compare final outputs across multiple runs (commented out by default)
- PSNR plots: plot PSNR curves for a run (commented out by default)

Run:
    python -m nerf.learn2d.visualize_learn2d
"""

from pathlib import Path
from nerf.learn2d.viz import make_progress_grid_by_epoch, tile_images_with_labels


# ============================================================
# 🔧 CONFIGURATION — EDIT THESE TO MATCH YOUR RUNS
# ============================================================

# Example runs for the *fox* image
RUNS = [
    Path("results/learn2d/hammock/L4_W128"),
    Path("results/learn2d/hammock/L4_W512"),
    Path("results/learn2d/hammock/L10_W128"),
    Path("results/learn2d/hammock/L10_W512"),
]

# Epoch checkpoints you saved for snapshots
SNAP_EPOCHS = [10, 300, 1000]


# ============================================================
# 1. GENERATE PROGRESS GRIDS FOR EACH RUN
# ============================================================

def generate_progress_grids():
    for run in RUNS:
        out = run / "progress_grid.png"
        print(f"[Progress] {run.name} → {out.name}")
        make_progress_grid_by_epoch(
            run_dir=run,
            epochs=SNAP_EPOCHS,
            out_path=out,
            include_final=True,
        )
    print("[Done] Progress grids saved.")


# ============================================================
# 2. FINAL SIDE-BY-SIDE COMPARISON ACROSS RUNS
# ============================================================

def compare_finals():
    image_paths = [r / "recon_final.png" for r in RUNS]
    labels = [r.name for r in RUNS]   # shorter labels

    out = Path("results/learn2d/hammock/finals_compare.png")
    tile_images_with_labels(
        image_paths=image_paths,
        labels=labels,
        grid=(1, len(image_paths)),  # horizontal row
        out_path=out,
        pad=8,
        bg=(0.15, 0.15, 0.15),
    )
    print(f"[Done] Final comparison saved: {out}")

# To enable this: uncomment the line in main():
# compare_finals()


# ============================================================
# 3. PSNR CURVE PLOT
# ============================================================

def plot_psnr_curves():
    import json
    import matplotlib.pyplot as plt

    for run in RUNS:
        log_file = run / "run_log.json"
        if not log_file.exists():
            print(f"[Skip] No run_log.json in {run}")
            continue

        log = json.loads(log_file.read_text())
        plt.figure()
        plt.plot(log["iters"], log["train_psnr"], label="train")
        if "val_psnr" in log:
            plt.plot(log["iters"], log["val_psnr"], label="val")

        plt.xlabel("Iteration")
        plt.ylabel("PSNR (dB)")
        plt.title(run.name)
        plt.legend()
        plt.tight_layout()

        out = run / "psnr_curve.png"
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"[PSNR] {run.name} → {out}")

# To enable this: uncomment the line in main():
# plot_psnr_curves()


# ============================================================
# MAIN
# ============================================================

def main():
    generate_progress_grids()
    compare_finals()
    plot_psnr_curves()


if __name__ == "__main__":
    main()
