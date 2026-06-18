from __future__ import annotations

from typing import Any

import numpy as np

from utils import calculate_laplacian_matrix


def evaluate_original_lap_smooth_cost(
    retargeter: Any,
    q_current: np.ndarray,
    q_previous: np.ndarray,
    target_laplacian: np.ndarray,
    adj_list: list[list[int]],
    obj_pts_local: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluate raw, unweighted nonlinear laplacian and smooth terms."""
    _, p_OC_dict, _ = retargeter._calc_manipulator_jacobians(
        q_current,
        links=retargeter.laplacian_match_links,
        obj_frame=(retargeter.object_name != "ground"),
    )
    robot_link_keys = list(retargeter.laplacian_match_links.keys())
    robot_pts_local = np.array([p_OC_dict[k] for k in robot_link_keys])
    vertices = np.vstack([robot_pts_local, obj_pts_local])

    laplacian_matrix = calculate_laplacian_matrix(vertices, adj_list)
    current_laplacian = laplacian_matrix @ vertices
    residual = current_laplacian - target_laplacian

    lap_cost = float(np.sum(residual * residual))

    dq_active = q_current[retargeter.q_a_indices] - q_previous[retargeter.q_a_indices]
    smooth_cost = float(np.sum(dq_active * dq_active))

    return lap_cost, smooth_cost, lap_cost + smooth_cost
