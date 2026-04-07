from __future__ import annotations

import argparse
from pathlib import Path

from src.algorithms import (
    AntColonyOptimizer,
    DifferentialEvolutionTSP,
    GeneticAlgorithmTSP,
    ParticleSwarmTSP,
)
from src.tsp_utils import create_distance_matrix, generate_cities
from src.visualization import (
    create_algorithm_summary,
    plot_all_convergence,
    save_route_frames_and_gif,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSP optimization comparison")
    parser.add_argument("--cities", type=int, default=30, help="Number of cities")
    parser.add_argument("--iters", type=int, default=150, help="Iterations per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cities = generate_cities(n_cities=args.cities, seed=args.seed)
    dist = create_distance_matrix(cities)

    algorithms = {
        "ACO": AntColonyOptimizer(distance_matrix=dist, n_iterations=args.iters, seed=args.seed),
        "GA": GeneticAlgorithmTSP(distance_matrix=dist, n_generations=args.iters, seed=args.seed),
        "PSO": ParticleSwarmTSP(distance_matrix=dist, n_iterations=args.iters, seed=args.seed),
        "DE": DifferentialEvolutionTSP(distance_matrix=dist, n_generations=args.iters, seed=args.seed),
    }

    all_results: dict[str, dict] = {}

    for name, algo in algorithms.items():
        print(f"Running {name}...")
        result = algo.run()
        all_results[name] = result

        algo_out = out_dir / name.lower()
        algo_out.mkdir(parents=True, exist_ok=True)

        save_route_frames_and_gif(
            cities=cities,
            route_history=result["best_route_history"],
            out_dir=algo_out,
            algo_name=name,
        )

    plot_all_convergence(all_results, out_dir=out_dir)
    create_algorithm_summary(all_results, out_dir=out_dir)

    print("\n=== Final best distances ===")
    for name, result in all_results.items():
        print(f"{name}: {result['best_distance']:.4f}")


if __name__ == "__main__":
    run()
