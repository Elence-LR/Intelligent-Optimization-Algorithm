from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd


COLOR_MAP = {
    "ACO": "#e63946",
    "GA": "#457b9d",
    "PSO": "#2a9d8f",
    "DE": "#f4a261",
}


def _plot_route(cities, route, title: str, out_path: Path) -> None:
    ordered = cities[route]
    closed = ordered.tolist() + [ordered[0].tolist()]
    closed = pd.DataFrame(closed, columns=["x", "y"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(cities[:, 0], cities[:, 1], c="#1d3557", s=36, zorder=3)
    ax.plot(closed["x"], closed["y"], c="#e76f51", linewidth=1.8, zorder=2)

    for idx, (x, y) in enumerate(cities):
        ax.text(x + 0.8, y + 0.8, str(idx), fontsize=7)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def save_route_frames_and_gif(cities, route_history, out_dir: Path, algo_name: str) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for i, route in enumerate(route_history, start=1):
        img_path = frames_dir / f"iter_{i:03d}.png"
        title = f"{algo_name} - Iteration {i}"
        _plot_route(cities, route, title, img_path)
        image_paths.append(img_path)

    gif_path = out_dir / f"{algo_name.lower()}_evolution.gif"
    with imageio.get_writer(gif_path, mode="I", duration=0.15, loop=0) as writer:
        for img_path in image_paths:
            writer.append_data(imageio.imread(img_path))

    final_path = out_dir / f"{algo_name.lower()}_best_route.png"
    _plot_route(cities, route_history[-1], f"{algo_name} - Final Best Route", final_path)


def plot_all_convergence(results: dict[str, dict], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for name, data in results.items():
        ax.plot(
            range(1, len(data["best_history"]) + 1),
            data["best_history"],
            label=f"{name} (best={data['best_distance']:.2f})",
            linewidth=2.0,
            color=COLOR_MAP.get(name),
        )

    ax.set_xlabel("Iteration / Generation")
    ax.set_ylabel("Best Tour Distance")
    ax.set_title("TSP Optimization Convergence Comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_comparison.png", dpi=160)
    plt.close(fig)


def create_algorithm_summary(results: dict[str, dict], out_dir: Path) -> None:
    rows = []
    for name, data in results.items():
        history = data["best_history"]
        rows.append(
            {
                "algorithm": name,
                "best_distance": round(data["best_distance"], 4),
                "initial_best": round(history[0], 4),
                "improvement": round(history[0] - data["best_distance"], 4),
                "iterations": len(history),
            }
        )

    df = pd.DataFrame(rows).sort_values("best_distance").reset_index(drop=True)
    df.to_csv(out_dir / "algorithm_summary.csv", index=False, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(df["algorithm"], df["best_distance"], color=[COLOR_MAP.get(x, "#999") for x in df["algorithm"]])
    ax.set_title("Best Distance by Algorithm")
    ax.set_ylabel("Distance")
    ax.grid(axis="y", alpha=0.25)

    for i, v in enumerate(df["best_distance"]):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "best_distance_bar.png", dpi=160)
    plt.close(fig)
