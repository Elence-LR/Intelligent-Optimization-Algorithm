# Intelligent-Optimization-Algorithm

## 智能优化算法课程作业：TSP 多算法对比

本项目以旅行商问题（TSP）为例，对比 4 种课堂常见智能优化算法：

- 蚁群优化算法（ACO）
- 遗传算法（GA）
- 粒子群优化算法（PSO，随机键编码）
- 差分进化算法（DE，随机键编码）

## 1. 环境准备（Conda）

在项目目录执行：

```bash
conda create -y -p ./.conda python=3.11
conda install -y -p ./.conda numpy pandas matplotlib seaborn scipy imageio tqdm
conda activate ./.conda
```

也可以使用：

```bash
conda env create -f environment.yml
conda activate intel_tsp_compare
```

## 2. 运行实验

```bash
python run_experiment.py --cities 30 --iters 150 --seed 42 --out outputs
```

## 3. 输出说明

运行后将生成：

- `outputs/convergence_comparison.png`：4 算法收敛曲线对比
- `outputs/best_distance_bar.png`：最终最优路径长度柱状图
- `outputs/algorithm_summary.csv`：实验结果表格
- `outputs/multi_seed_summary.csv`：多随机种子统计结果
- `outputs/aco/aco_evolution.gif`、`outputs/ga/ga_evolution.gif`、`outputs/pso/pso_evolution.gif`、`outputs/de/de_evolution.gif`：每代最优路径演化动画
- 各算法目录下 `*_best_route.png`：最终最优路径图

## 4. 实现说明

- ACO/GA 直接在排列空间优化路径。
- PSO/DE 使用随机键编码：每个个体是连续向量，按向量排序结果映射成城市访问顺序。
- 所有算法都记录每一代（迭代）的当前全局最优路径和最优路径长度，用于可视化。

## 5. 可扩展方向

- 增加多个随机种子重复实验，比较统计显著性
- 对不同城市规模（30/50/100）做复杂度与质量对比
- 增加更多算子（如 GA 的 2-opt 局部搜索）
