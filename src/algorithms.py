from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .tsp_utils import keys_to_route, route_distance


@dataclass
class AntColonyOptimizer:
    distance_matrix: np.ndarray
    n_ants: int = 40
    n_iterations: int = 150
    alpha: float = 1.0
    beta: float = 5.0
    evaporation_rate: float = 0.5
    q: float = 100.0
    seed: int = 42

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        n = self.distance_matrix.shape[0]
        pheromone = np.ones((n, n), dtype=float)
        heuristic = 1.0 / self.distance_matrix

        best_distance = np.inf
        best_route = None
        best_history = []
        route_history = []

        for _ in range(self.n_iterations):
            ant_routes = []
            ant_distances = []

            for _ in range(self.n_ants):
                start = rng.integers(0, n)
                route = [start]
                unvisited = set(range(n))
                unvisited.remove(start)

                while unvisited:
                    current = route[-1]
                    candidates = np.array(list(unvisited), dtype=int)
                    tau = pheromone[current, candidates] ** self.alpha
                    eta = heuristic[current, candidates] ** self.beta
                    probs = tau * eta
                    probs /= probs.sum()
                    nxt = int(rng.choice(candidates, p=probs))
                    route.append(nxt)
                    unvisited.remove(nxt)

                route_arr = np.array(route, dtype=int)
                dist = route_distance(route_arr, self.distance_matrix)
                ant_routes.append(route_arr)
                ant_distances.append(dist)

                if dist < best_distance:
                    best_distance = dist
                    best_route = route_arr.copy()

            pheromone *= 1.0 - self.evaporation_rate
            for route, dist in zip(ant_routes, ant_distances):
                delta = self.q / (dist + 1e-12)
                for i in range(n):
                    a = route[i]
                    b = route[(i + 1) % n]
                    pheromone[a, b] += delta
                    pheromone[b, a] += delta

            best_history.append(float(best_distance))
            route_history.append(best_route.copy())

        return {
            "best_distance": float(best_distance),
            "best_route": best_route,
            "best_history": best_history,
            "best_route_history": route_history,
        }


@dataclass
class GeneticAlgorithmTSP:
    distance_matrix: np.ndarray
    pop_size: int = 120
    n_generations: int = 150
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    elite_size: int = 2
    seed: int = 42

    def _tournament_select(self, population: np.ndarray, fitness: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = 4
        ids = rng.integers(0, population.shape[0], size=k)
        best_id = ids[np.argmin(fitness[ids])]
        return population[best_id].copy()

    def _ordered_crossover(self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(p1)
        c1, c2 = sorted(rng.integers(0, n, size=2))
        child = -np.ones(n, dtype=int)
        child[c1:c2] = p1[c1:c2]

        p2_filtered = [x for x in p2 if x not in child[c1:c2]]
        fill_idx = [i for i in range(n) if child[i] == -1]
        for i, gene in zip(fill_idx, p2_filtered):
            child[i] = gene
        return child

    def _swap_mutation(self, individual: np.ndarray, rng: np.random.Generator) -> None:
        i, j = rng.integers(0, len(individual), size=2)
        individual[i], individual[j] = individual[j], individual[i]

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        n = self.distance_matrix.shape[0]

        population = np.array([rng.permutation(n) for _ in range(self.pop_size)], dtype=int)
        best_distance = np.inf
        best_route = None
        best_history = []
        route_history = []

        for _ in range(self.n_generations):
            fitness = np.array([route_distance(ind, self.distance_matrix) for ind in population], dtype=float)
            sorted_idx = np.argsort(fitness)

            if fitness[sorted_idx[0]] < best_distance:
                best_distance = float(fitness[sorted_idx[0]])
                best_route = population[sorted_idx[0]].copy()

            new_population = [population[i].copy() for i in sorted_idx[: self.elite_size]]

            while len(new_population) < self.pop_size:
                p1 = self._tournament_select(population, fitness, rng)
                p2 = self._tournament_select(population, fitness, rng)

                if rng.random() < self.crossover_rate:
                    c1 = self._ordered_crossover(p1, p2, rng)
                    c2 = self._ordered_crossover(p2, p1, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                if rng.random() < self.mutation_rate:
                    self._swap_mutation(c1, rng)
                if rng.random() < self.mutation_rate:
                    self._swap_mutation(c2, rng)

                new_population.extend([c1, c2])

            population = np.array(new_population[: self.pop_size], dtype=int)
            best_history.append(float(best_distance))
            route_history.append(best_route.copy())

        return {
            "best_distance": float(best_distance),
            "best_route": best_route,
            "best_history": best_history,
            "best_route_history": route_history,
        }


@dataclass
class ParticleSwarmTSP:
    distance_matrix: np.ndarray
    n_particles: int = 80
    n_iterations: int = 150
    w: float = 0.7
    c1: float = 1.5
    c2: float = 1.5
    seed: int = 42

    def _objective(self, x: np.ndarray) -> float:
        route = keys_to_route(x)
        return route_distance(route, self.distance_matrix)

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        n = self.distance_matrix.shape[0]

        positions = rng.uniform(0.0, 1.0, size=(self.n_particles, n))
        velocities = rng.uniform(-0.1, 0.1, size=(self.n_particles, n))

        pbest_pos = positions.copy()
        pbest_scores = np.array([self._objective(p) for p in positions], dtype=float)

        gbest_idx = int(np.argmin(pbest_scores))
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_score = float(pbest_scores[gbest_idx])
        gbest_route = keys_to_route(gbest_pos)

        best_history = []
        route_history = []

        for _ in range(self.n_iterations):
            r1 = rng.random(size=(self.n_particles, n))
            r2 = rng.random(size=(self.n_particles, n))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (pbest_pos - positions)
                + self.c2 * r2 * (gbest_pos - positions)
            )
            positions += velocities
            positions = np.clip(positions, 0.0, 1.0)

            scores = np.array([self._objective(p) for p in positions], dtype=float)
            improved = scores < pbest_scores
            pbest_scores[improved] = scores[improved]
            pbest_pos[improved] = positions[improved]

            current_best_idx = int(np.argmin(pbest_scores))
            if pbest_scores[current_best_idx] < gbest_score:
                gbest_score = float(pbest_scores[current_best_idx])
                gbest_pos = pbest_pos[current_best_idx].copy()
                gbest_route = keys_to_route(gbest_pos)

            best_history.append(float(gbest_score))
            route_history.append(gbest_route.copy())

        return {
            "best_distance": float(gbest_score),
            "best_route": gbest_route,
            "best_history": best_history,
            "best_route_history": route_history,
        }


@dataclass
class DifferentialEvolutionTSP:
    distance_matrix: np.ndarray
    pop_size: int = 100
    n_generations: int = 150
    f: float = 0.7
    cr: float = 0.9
    seed: int = 42

    def _objective(self, x: np.ndarray) -> float:
        route = keys_to_route(x)
        return route_distance(route, self.distance_matrix)

    def run(self) -> dict:
        rng = np.random.default_rng(self.seed)
        n = self.distance_matrix.shape[0]

        pop = rng.uniform(0.0, 1.0, size=(self.pop_size, n))
        scores = np.array([self._objective(ind) for ind in pop], dtype=float)

        best_idx = int(np.argmin(scores))
        best_score = float(scores[best_idx])
        best_vec = pop[best_idx].copy()
        best_route = keys_to_route(best_vec)

        best_history = []
        route_history = []

        for _ in range(self.n_generations):
            for i in range(self.pop_size):
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = rng.choice(candidates, size=3, replace=False)
                mutant = pop[a] + self.f * (pop[b] - pop[c])
                mutant = np.clip(mutant, 0.0, 1.0)

                cross_mask = rng.random(n) < self.cr
                if not np.any(cross_mask):
                    cross_mask[rng.integers(0, n)] = True
                trial = np.where(cross_mask, mutant, pop[i])

                trial_score = self._objective(trial)
                if trial_score < scores[i]:
                    pop[i] = trial
                    scores[i] = trial_score

            current_idx = int(np.argmin(scores))
            if scores[current_idx] < best_score:
                best_score = float(scores[current_idx])
                best_vec = pop[current_idx].copy()
                best_route = keys_to_route(best_vec)

            best_history.append(float(best_score))
            route_history.append(best_route.copy())

        return {
            "best_distance": float(best_score),
            "best_route": best_route,
            "best_history": best_history,
            "best_route_history": route_history,
        }
