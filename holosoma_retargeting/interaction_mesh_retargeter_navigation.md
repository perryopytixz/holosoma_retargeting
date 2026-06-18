# `InteractionMeshRetargeter` Code Navigation

Target file:
`holosoma_retargeting/src/interaction_mesh_retargeter.py`

Primary role:
`InteractionMeshRetargeter` is the core retargeting solver class used by the
Holosoma example entry points. It takes framewise human/object data, builds an
interaction-mesh target, and solves a constrained local convex subproblem
within an SQP-style inner loop for each frame.

## High-Level Structure

```text
External callers
├─ examples/robot_retarget.py
└─ examples/parallel_robot_retarget.py
          |
          v
InteractionMeshRetargeter
├─ 1. Initialization layer
│  ├─ __init__
│  │  role: load models, define optimization-variable slice, set weights,
│  │        limits, tolerances, and feature switches
│  ├─ _init_foot_lock
│  │  role: normalize explicit frame-window foot-lock settings
│  └─ _init_self_collision
│     role: expand configured self-collision body pairs into MuJoCo geom pairs
│
├─ 2. Main solving layer
│  └─ retarget_motion
│     role: outer frame loop over the whole motion sequence
│     pipeline:
│     ├─ prepare current-frame human/object geometry
│     ├─ build interaction mesh
│     ├─ compute adjacency list and target Laplacian coordinates
│     └─ call iterate(...)
│        └─ iterate
│           role: inner SQP-style loop for one frame
│           └─ solve_single_iteration
│              role: build and solve one local convex subproblem
│              components:
│              ├─ Laplacian linearization from robot keypoint Jacobians
│              ├─ foot sticking / foot lock constraints
│              ├─ non-penetration constraints
│              ├─ self-collision constraints
│              ├─ joint-limit constraints
│              ├─ trust-region constraint
│              └─ objective = Laplacian match
│                            + nominal tracking
│                            + Q_diag regularization
│                            + temporal smoothness
│
├─ 3. Geometry and Jacobian layer
│  ├─ _calc_manipulator_jacobians
│  │  role: compute current link positions and Jacobians for the selected
│  │        robot keypoints, optionally expressed in object frame
│  │  depends on:
│  │  └─ _calc_contact_jacobian_from_point
│  │     role: point translational Jacobian wrt generalized-position rate
│  │     depends on:
│  │     └─ _build_transform_qdot_to_qvel_fast
│  │        role: convert qdot-based Jacobians to MuJoCo's qvel convention,
│  │              including free-joint quaternion handling
│  │
│  ├─ _update_jacobians_and_phis_from_q
│  │  role: build non-penetration linearization terms
│  │  pipeline:
│  │  ├─ _prefilter_pairs_with_mj_collision
│  │  │  role: broad-phase candidate collision filtering
│  │  └─ _compute_jacobian_for_contact_relative
│  │     role: signed-distance directional Jacobian for a geom pair
│  │
│  └─ _compute_self_collision_constraints
│     role: self-collision distance/Jacobian terms for configured body pairs
│
├─ 4. Visualization and debug layer
│  ├─ _setup_visualization
│  ├─ draw_mesh_from_geom
│  ├─ draw_mesh_pair_with_contact
│  ├─ draw_q
│  ├─ draw_keypoints
│  ├─ visualize_motion
│  ├─ visualize_tetrahedra
│  └─ _draw_self_collision_geoms
│
└─ 5. Small utilities
   ├─ _is_foot_locked_in_window
   ├─ _world_to_body_frame
   ├─ _get_geometry_name
   └─ _get_robot_link_positions
```

## Mental Model

```text
For each frame:
human/object data
-> target interaction mesh
-> target Laplacian coordinates
-> repeated local linearization around current q
-> solve a constrained convex subproblem
-> update q
-> store retargeted motion for this frame
```

## Reading Order

Recommended first pass:

1. `__init__`
2. `retarget_motion`
3. `iterate`
4. `solve_single_iteration`
5. `_calc_manipulator_jacobians`
6. `_update_jacobians_and_phis_from_q`

Recommended second pass:

1. `_compute_self_collision_constraints`
2. `_calc_contact_jacobian_from_point`
3. `_build_transform_qdot_to_qvel_fast`
4. visualization/debug functions as needed

## Function Groups by Purpose

### Motion-Level Control

- `retarget_motion`: frame loop, target-mesh construction, result storage
- `iterate`: repeated local solves for one frame
- `solve_single_iteration`: single local convex solve

### Problem Modeling

- `_calc_manipulator_jacobians`: keypoint positions and Jacobians
- `_update_jacobians_and_phis_from_q`: non-penetration linearization
- `_compute_self_collision_constraints`: self-collision linearization
- `_is_foot_locked_in_window`: foot-lock activation logic

### MuJoCo Differential Kinematics Plumbing

- `_build_transform_qdot_to_qvel_fast`
- `_calc_contact_jacobian_from_point`
- `_compute_jacobian_for_contact_relative`
- `_prefilter_pairs_with_mj_collision`

### Visualization

- `_setup_visualization`
- `draw_*`
- `visualize_motion`
- `visualize_tetrahedra`

## What To Ignore On First Read

Skip these on the first pass unless the immediate task is debugging rendering or
collision geometry:

- visualization setup and all `draw_*` functions
- `visualize_motion`
- `visualize_tetrahedra`
- `_draw_self_collision_geoms`

The main algorithm can be understood without them.
