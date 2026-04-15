"""
TSP 多算法对比实验入口脚本。

本脚本负责：
- 解析命令行参数（城市数量、迭代次数、随机种子、输出目录）
- 生成随机城市坐标并构造距离矩阵
- 依次运行 4 种智能优化算法（ACO/GA/PSO/DE），收集每代全局最优结果
- 为每种算法保存路径演化帧与 GIF，并在实验结束后汇总对比图与统计表

算法核心实现位于 src/algorithms.py；城市生成与距离矩阵位于 src/tsp_utils.py；
可视化与输出生成位于 src/visualization.py。
"""

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
    # 命令行参数解析：
    # - cities: 城市数量（问题规模）
    # - iters : 每个算法的迭代/代数（不同算法内部命名可能不同，这里统一由脚本传入）
    # - seed  : 随机种子（保证可复现的城市分布与算法初始化）
    # - out   : 输出目录（保存图表、表格、GIF 等）
    parser = argparse.ArgumentParser(description="TSP optimization comparison")
    parser.add_argument("--cities", type=int, default=30, help="Number of cities")
    parser.add_argument("--iters", type=int, default=150, help="Iterations per algorithm")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--aco-beta", type=float, default=5.0, help="Beta parameter for ACO")
    return parser.parse_args()


def run() -> None:
    # 1) 读取参数并创建输出目录
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) 生成城市坐标并构造距离矩阵
    # cities 通常是形状为 (n_cities, 2) 的二维坐标数组（x, y）
    # dist 为 (n_cities, n_cities) 的对称矩阵，其中 dist[i, j] 表示城市 i 到城市 j 的距离
    cities = generate_cities(n_cities=args.cities, seed=args.seed)
    dist = create_distance_matrix(cities)

    # 3) 实例化算法
    # 注意：不同算法内部使用的“迭代/代数”参数命名不同：
    # - ACO / PSO：更常见用 n_iterations
    # - GA / DE ：更常见用 n_generations
    # 本脚本用同一个 args.iters 统一控制对比实验的计算预算
    algorithms = {
        "ACO": AntColonyOptimizer(
            distance_matrix=dist,
            n_iterations=args.iters,
            beta=args.aco_beta,
            seed=args.seed,
        ),
        "GA": GeneticAlgorithmTSP(distance_matrix=dist, n_generations=args.iters, seed=args.seed),
        "PSO": ParticleSwarmTSP(distance_matrix=dist, n_iterations=args.iters, seed=args.seed),
        "DE": DifferentialEvolutionTSP(distance_matrix=dist, n_generations=args.iters, seed=args.seed),
    }

    # all_results 用于收集所有算法的输出结果，便于在实验结束后统一绘图和汇总
    # 约定 result 至少包含：
    # - best_distance：最终全局最优路径长度
    # - best_route_history：每一代/迭代的全局最优路径（用于生成演化动画与收敛曲线）
    all_results: dict[str, dict] = {}

    # 4) 逐个运行算法并保存该算法的可视化输出
    for name, algo in algorithms.items():
        print(f"Running {name}...")
        # algo.run() 返回算法运行过程的记录与最终最优解
        result = algo.run()
        all_results[name] = result

        # 每种算法单独建一个输出子目录，避免文件相互覆盖
        algo_out = out_dir / name.lower()
        algo_out.mkdir(parents=True, exist_ok=True)

        # 根据每代最优路径历史生成帧序列与 GIF，用于展示“最优路径如何随迭代逐步改进”
        save_route_frames_and_gif(
            cities=cities,
            route_history=result["best_route_history"],
            out_dir=algo_out,
            algo_name=name,
        )

    # 5) 跨算法汇总输出：收敛曲线对比图与结果汇总表等
    plot_all_convergence(all_results, out_dir=out_dir)
    create_algorithm_summary(all_results, out_dir=out_dir)

    # 6) 在控制台打印各算法最终最优路径长度，便于快速对比
    print("\n=== Final best distances ===")
    for name, result in all_results.items():
        print(f"{name}: {result['best_distance']:.4f}")


if __name__ == "__main__":
    run()
