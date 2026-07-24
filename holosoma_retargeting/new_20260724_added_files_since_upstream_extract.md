# 上游代码抽取后新增的文件

对比基线：

```text
d011afc Extract holosoma_retargeting from amazon-far/holosoma
```

本清单记录相对 `d011afc` 的 Git 状态为 `A`（新增）的文件。清单不包含仅被
修改的上游文件。提交新增文件后，可使用以下命令重新生成 Git 已跟踪部分：

```bash
git diff --diff-filter=A --name-only d011afc..HEAD
```

## 环境与文档

- `holosoma_retargeting/environment.yml`
- `holosoma_retargeting/new_20260724_added_files_since_upstream_extract.md`
- `holosoma_retargeting/new_20260724_retarget_lafan_to_holosoma_data_note.md`

## 保留的对比与说明文件

- `holosoma_retargeting/compare_variants_viser.py`
- `holosoma_retargeting/examples/pipeline.ipynb`
- `holosoma_retargeting/examples/robot_retarget个人注释版.py`
- `holosoma_retargeting/examples/robot_retarget原版.py`
- `holosoma_retargeting/src/interaction_mesh_retargeter_个人注释版.py`
- `holosoma_retargeting/src/interaction_mesh_retargeter原版.py`

## Sandbox 配置与共享支持代码

- `holosoma_retargeting/config_types/sandbox_retargeting.py`
- `holosoma_retargeting/src/original_cost_components.py`

## Sandbox 运行入口

- `holosoma_retargeting/examples/robot_retarget_sandbox_experimental.py`
- `holosoma_retargeting/examples/robot_retarget_sandbox_extra_keypoints_experimental.py`
- `holosoma_retargeting/examples/robot_retarget_sandbox_extra_keypoints_offsets_experimental.py`
- `holosoma_retargeting/examples/robot_retarget_sandbox_removed_experimental.py`

## Sandbox 重定向算法实现

- `holosoma_retargeting/src/interaction_mesh_retargeter_sandbox_experimental.py`
- `holosoma_retargeting/src/interaction_mesh_retargeter_sandbox_extra_keypoints_experimental.py`
- `holosoma_retargeting/src/interaction_mesh_retargeter_sandbox_extra_keypoints_offsets_experimental.py`
- `holosoma_retargeting/src/interaction_mesh_retargeter_sandbox_removed_experimental.py`

## 实验控制与分析

- `holosoma_retargeting/examples/run_sandbox_many_data.ipynb`
- `holosoma_retargeting/examples/multi_laplacian_hessian_spectrum_diagnosis.ipynb`

## LAFAN 数据适配

- `holosoma_retargeting/data_conversion/convert_lafan_bvh_to_holosoma.py`

本清单共记录 22 个文件，其中包含本清单文件本身。
