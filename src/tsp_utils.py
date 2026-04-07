from __future__ import annotations

import numpy as np


def generate_cities(n_cities: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 100, size=(n_cities, 2))


def create_distance_matrix(cities: np.ndarray) -> np.ndarray:
    diff = cities[:, None, :] - cities[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e-12)
    return dist


def route_distance(route: np.ndarray, distance_matrix: np.ndarray) -> float:
    shifted = np.roll(route, -1)
    return float(np.sum(distance_matrix[route, shifted]))


def keys_to_route(keys: np.ndarray) -> np.ndarray:
    return np.argsort(keys)
