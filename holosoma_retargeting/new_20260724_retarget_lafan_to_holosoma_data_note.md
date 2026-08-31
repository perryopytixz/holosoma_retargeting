# Holosoma 外部 LAFAN 数据说明与适配记录

## 1. 文档目的

`Retarget/` 与 `holosoma_retargeting/` 是两个独立的 retargeting 项目，分别具有自己的算法、
数据 loader、坐标约定和结果格式。当前研究以 Holosoma retargeting 为主，`Retarget/`
提供参考代码和外部 LAFAN 动作数据。

本文记录 Holosoma 当前使用的数据、从 `Retarget/` 导入的三个 LAFAN 动作、第一次数据
适配失败的原因，以及当前采用的修复流程。这里的转换把 Retarget LAFAN 源数据适配成
Holosoma LAFAN loader 使用的米制、Y-up、固定 22 关节顺序 NPY。

## 2. Holosoma 当前数据

### 2.1 原生 Holosoma 数据

`demo_data/OMOMO_new/` 包含两个物体交互动作：
`sub10_largebox_049.pt` 和 `sub3_largebox_003.pt`。PT 文件保存 InterMimic 张量；
Holosoma 从中提取 52 个 SMPL-H 人体关节和物体位姿，并通过 `smplh` loader 读取。

`demo_data/climb/` 包含五个攀爬动作：`mocap_climb_seq_0` 至
`mocap_climb_seq_4`。每个任务目录包含人体关节位置 NPY，以及箱体场景使用的 XML、URDF
和 OBJ。Holosoma 的 `mocap` loader 读取 NPY，并按当前实现每四帧取一帧。

这些数据直接使用 Holosoma 原生 loader。

### 2.2 从 Retarget 导入的 LAFAN 数据

当前导入 `aiming1_subject1`、`dance1_subject2` 和 `dance2_subject5`。三条动作的可用
源文件不同。

#### aiming1_subject1

`Retarget/bvh_npz/lafan1/aiming1_subject1.bvh` 是完整原始 BVH，包含 7184 帧、
30.0003 FPS、cm/Y-up 数据、完整 22 关节层级和真实 Toe 轨迹。当前 aiming 使用该 BVH。

同名 `Retarget/bvh_npz/aiming1_subject1.npz` 有 14367 帧，`positions` 为
`(14367, 22, 3) float64`，帧率为 59.9988 FPS，单位和上轴为 m/Z-up。
LeftToe/RightToe 全帧为零，骨骼连接只有 15 条，Toe rotation 也没有有效运动。该 NPZ
保留用于问题溯源，当前 Holosoma 适配不读取它。

#### dance1_subject2

`Retarget/bvh_npz/lafan1/dance1_subject2.bvh` 是 0 bytes 空文件，不能解析。
`Retarget/bvh_npz/dance1_subject2.npz` 包含 3945 帧、30.0003 FPS、cm/Y-up 数据和
有效 Toe，当前 dance1 使用该 NPZ。

#### dance2_subject5

`Retarget/bvh_npz/lafan1/dance2_subject5.bvh` 是 0 bytes 空文件，不能解析。
`Retarget/bvh_npz/dance2_subject5.npz` 包含 6771 帧、30.0003 FPS、cm/Y-up 数据和
有效 Toe，当前 dance2 使用该 NPZ。

因此当前十条不同动作由两个 OMOMO、五个 climb 和三个导入 LAFAN 组成。Notebook 中同一
动作可以运行多个算法变体，task 数量会大于动作数量。

## 3. 为什么需要数据适配

`Retarget/` 的算法通过 [`../../Retarget/bvh_loader.py`](../../Retarget/bvh_loader.py)
读取自定义源 NPZ。该 NPZ 包含 `positions`、`rotations`、`joint_names`、
`connections`、`frame_time` 和 `frame_count` 等字段。

Holosoma 的 LAFAN 分支通过
[`examples/robot_retarget_sandbox_experimental.py`](examples/robot_retarget_sandbox_experimental.py)
读取 `<task>.npy`。该 NPY 只保存 `(T, 22, 3)` 人体全局关节位置，关节含义由
`LAFAN_DEMO_JOINTS` 的固定列顺序确定。

当前适配将选定的 BVH 或源 NPZ 转换为：

```text
shape: (T, 22, 3)
dtype: float32
unit: meter
up axis: Y
joint order: LAFAN_DEMO_JOINTS
horizontal origin: first-frame Hips XZ
```

生成文件位于：

```text
generated_data/retarget_lafan/
```

Holosoma loader 随后把 Y-up NPY 转换为求解器使用的 Z-up 坐标。源 NPZ 是人体动作中间
数据；Holosoma retarget 产生的结果 NPZ 保存机器人 `qpos`、实际人体关节和代价。两者
扩展名相同，但语义不同。

## 4. 第一次适配失败的原因

第一次失败包含一处 Holosoma 代码错误、一条外部源数据问题和一处初版适配代码错误。

### 4.1 Holosoma 换轴代码使用了反射

旧版 [`src/utils.py`](src/utils.py) 使用：

```text
[x, y, z] -> [x, z, y]
```

该变换交换 Y/Z 轴，矩阵行列式为 `-1`，会反射骨架并改变坐标手性。反射后的骨架会影响
人体朝向估计、左右关系和 interaction mesh 几何。该错误会影响所有进入此 LAFAN loader
的动作，包括格式正常的 dance 数据。

### 4.2 aiming NPZ 与 dance NPZ 不属于同一种数据格式

dance1 和 dance2 NPZ 是约 30 FPS、cm/Y-up 数据，Toe 有效。aiming NPZ 是约
60 FPS、m/Z-up 数据，Toe 全零且部分骨骼连接缺失。

m/Z-up/60 FPS 表示 aiming NPZ 的格式与当前接口不兼容；Toe 和骨骼连接缺失属于源数据
内容不完整。已知单位、上轴和帧率可以转换，缺失的真实 Toe 轨迹不能从该 NPZ 无损恢复。

### 4.3 初版适配代码没有验证和分流

旧版 [`examples/run_sandbox_many_data.ipynb`](examples/run_sandbox_many_data.ipynb)
中的 LAFAN NPZ preparation cell 把三份 NPZ 全部当成 cm/Y-up/30 FPS 数据，按
`LAFAN_DEMO_JOINTS` 重排后统一乘 `0.01`。

该逻辑使 aiming 的米制位置再次缩小 100 倍，又把已经是 Z-up 的位置交给 Y-up loader
换轴。约 60 FPS 的序列仍被标记为 30 FPS，零 Toe 继续进入人体落地、脚接触判断和
Laplacian landmark 计算。

因此第一次失败的责任划分为：

- `[x, z, y]` 反射属于 Holosoma 换轴代码错误；
- aiming NPZ 的格式不兼容和 Toe 信息缺失属于外部源数据问题；
- 未在接口边界验证和选择正确源文件属于初版适配代码错误。

## 5. 当前修复

### 5.1 使用 proper rotation

[`src/utils.py`](src/utils.py) 当前使用：

```text
[x, y, z] -> [x, -z, y]
```

该矩阵行列式为 `+1`，是绕 X 轴的 proper rotation，保持坐标手性。

### 5.2 按动作选择可用源

[`examples/run_sandbox_many_data.ipynb`](examples/run_sandbox_many_data.ipynb) 当前优先
选择同名非空 BVH；BVH 缺失或为空时，使用通过验证的同名源 NPZ。

aiming 使用完整 BVH。dance1 和 dance2 因原始 BVH 为空，使用有效 NPZ。

写入 Holosoma NPY 前检查 FPS、尺度、上轴、Toe、坐标手性和关节顺序。这些检查用于阻止
60 FPS 被标记为 30 FPS、米制数据重复缩放、Z-up 数据重复换轴、零 Toe 进入接触计算，
以及左右关节或坐标手性错误。

### 5.3 从完整 BVH 重建 aiming

[`data_conversion/convert_lafan_bvh_to_holosoma.py`](data_conversion/convert_lafan_bvh_to_holosoma.py)
解析 BVH hierarchy、offset 和 motion channel，将 Euler rotation 转为 quaternion，
通过前向运动学重建 22 个全局关节位置，再按 `LAFAN_DEMO_JOINTS` 重排。

该路径从完整 BVH 恢复真实 Toe 轨迹，避免给不完整 aiming NPZ 人工补 Toe。

## 6. 当前生成数据

正式 Holosoma LAFAN 输入为：

```text
generated_data/retarget_lafan/aiming1_subject1.npy
generated_data/retarget_lafan/dance1_subject2.npy
generated_data/retarget_lafan/dance2_subject5.npy
```

对应 shape 分别为 `(7184, 22, 3)`、`(3945, 22, 3)` 和 `(6771, 22, 3)`。三份数据均为
`float32`、米制、Y-up、首帧 Hips XZ 居中。

每个生成文件旁的 `.conversion.json` 记录源文件路径、大小、时间戳、帧数、帧率、坐标
语义和转换版本。aiming 的 raw-BVH metadata 还记录源文件 SHA-256、输入/输出关节顺序和
验证指标。Regular 与 removed Holosoma retargeter 读取相同的生成 NPY。

## 7. 实现位置

[`src/utils.py`](src/utils.py) 定义正确的 Y-up 到 Z-up proper rotation。

[`data_conversion/convert_lafan_bvh_to_holosoma.py`](data_conversion/convert_lafan_bvh_to_holosoma.py)
负责完整 BVH 解析、前向运动学、重排、验证和 NPY/metadata 生成。

[`examples/run_sandbox_many_data.ipynb`](examples/run_sandbox_many_data.ipynb) 负责源文件
发现、BVH 优先选择、NPZ 回退验证、任务生成和 Holosoma sandbox 运行。

`Retarget/bvh_npz/` 下的源 BVH 和 NPZ 保持原状。当前适配生成的数据写入 Holosoma 的
`generated_data/retarget_lafan/`。

## 8. 复现

在 `holosoma_retargeting/holosoma_retargeting/` 目录中单独重建 aiming 输入：

```bash
conda run -n robot python data_conversion/convert_lafan_bvh_to_holosoma.py \
    ../../Retarget/bvh_npz/lafan1/aiming1_subject1.bvh \
    --output-dir generated_data/retarget_lafan \
    --force
```

完整的数据发现、dance NPZ 适配和 Holosoma sandbox 任务由
`examples/run_sandbox_many_data.ipynb` 统一执行。运行前应重启 Jupyter kernel，使当前
源选择和验证逻辑生效。

## 9. 数据侧限制

aiming NPZ 保留用于溯源，不作为当前输入。dance1 和 dance2 缺少可解析的原始 BVH，因此
继续依赖现有 NPZ。

新增 LAFAN 数据必须先确认单位、上轴、帧率、关节顺序和 Toe 有效性，再生成 Holosoma
输入。后续 Holosoma 求解实验状态记录在 `TODO.md` 和对应实验 notebook 中。
