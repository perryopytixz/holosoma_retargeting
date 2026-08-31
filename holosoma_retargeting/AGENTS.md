# Holosoma 重定向说明

本说明适用于当前目录及其子目录，并补充仓库根目录的 `AGENTS.md`。

## 原始问题与文献出处

本项目讨论的 OmniRetarget 原始优化问题以以下论文为主要来源：

Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel,
Carmelo Sferrazza, C. Karen Liu, Rocky Duan, and Guanya Shi,
“OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body
Loco-Manipulation and Scene Interaction,” arXiv:2509.26633v2 [cs.RO], 2025.

仓库内论文文件：
`../../Retarget/Yang et al. - 2025 - OmniRetarget interaction-preserving data generation for humanoid whole-body loco-manipulation and s.pdf`。

讨论“原始问题”时，以论文定义的逐帧非线性约束优化模型为准。讨论论文算法时，还要单独说明论文对原始问题构造的 Sequential SOCP。讨论 Holosoma 实现时，以具体源文件实际组装的目标、约束、决策变量和更新规则为准。

## OmniRetarget 论文的原始思想

### Interaction mesh

论文使用 interaction mesh 表示人体或机器人、物体和环境之间的空间关系。网格顶点由人体或机器人关键点、物体表面采样点和环境采样点组成，网格拓扑通过 Delaunay 四面体剖分得到。

顶点 \(p_{t,i}\) 的均匀权重 Laplacian 坐标定义为

\[
\mathcal L(p_{t,i})
=p_{t,i}-\sum_{j\in\mathcal N(i)}w_{ij}p_{t,j},
\qquad
w_{ij}=\frac{1}{|\mathcal N(i)|}.
\]

人体示范网格提供源 Laplacian 坐标，机器人、同一物体和同一环境构成目标网格。机器人重定向通过减小两组 Laplacian 坐标之间的差异，保留人体各部位与物体、地形之间的相对空间和接触关系。

### 论文定义的逐帧非线性问题

机器人配置 \(q_t\) 包含浮动基座的位置、四元数姿态和关节角。机器人关键点 \(p_{t,i}^{\mathrm{target}}(q_t)\) 由非线性正向运动学计算。论文公式 (3) 定义的逐帧目标问题为

\[
\begin{aligned}
q_t^\star=\arg\min_{q_t}\quad
&\sum_i
\left\|
\mathcal L\!\left(p_{t,i}^{\mathrm{source}}\right)
-\mathcal L\!\left(p_{t,i}^{\mathrm{target}}(q_t)\right)
\right\|_2^2
+\left\|q_t-q_{t-1}\right\|_Q^2 \\
\text{s.t.}\quad
&\phi_j(q_t)\geq 0,\qquad \forall j,\\
&q_{\min}\leq q_t\leq q_{\max},\\
&v_{\min}\,\Delta t
\leq q_t-q_{t-1}
\leq v_{\max}\,\Delta t,\\
&p_t^F=p_{t-1}^F,qquad \forall\text{ stance foot}.
\end{aligned}
\]

其中，第一项是 interaction-mesh Laplacian deformation，第二项是时间平滑项，\(\phi_j\) 是碰撞对的有符号距离。硬约束分别表达非穿透、配置边界、速度边界和支撑脚固定。该问题含有非线性正向运动学、非线性碰撞距离和四元数变量，因此是非凸约束优化问题。

### 论文采用的 Sequential SOCP

论文没有在每一帧直接求解上述非线性问题。第 \(t\) 帧以前一帧解初始化：

\[
\bar q_0=q_{t-1}^\star.
\]

在第 \(n\) 次内层迭代中，以 \(\bar q_n\) 为线性化点，对目标 Laplacian 坐标、碰撞距离和足部位置进行一阶展开。论文公式 (15) 的局部子问题为

\[
\begin{aligned}
\Delta q_n^\star=\arg\min_{\Delta q_n}\quad
&\left\|
L^{\mathrm{source}}
-\left(J_L^n\Delta q_n+\bar L_n^{\mathrm{target}}\right)
\right\|_2^2 \\
&+\left\|\bar q_n+\Delta q_n-q_{t-1}\right\|_Q^2\\
\text{s.t.}\quad
&J_j^n\Delta q_n+\phi_j(\bar q_n)\geq0,qquad\forall j,\\
&q_{\min}\leq\bar q_n+\Delta q_n\leq q_{\max},\\
&v_{\min}\Delta t
\leq\bar q_n+\Delta q_n-q_{t-1}
\leq v_{\max}\Delta t,\\
&p_t^F(\bar q_n)+J_F^n\Delta q_n=p_{t-1}^F,
\qquad\forall\text{ stance foot},\\
&\|\Delta q_n\|_2\leq\varepsilon.
\end{aligned}
\]

求解后执行

\[
\bar q_{n+1}=\bar q_n+\Delta q_n^\star.
\]

论文将这一过程称为 customized SQP-style solver 和 Sequential SOCP。trust region 半径取 \(\varepsilon=0.2\)，每帧最多执行 10 次内层迭代。线性化 Laplacian residual 的平方产生 Gauss-Newton 二次项

\[
H_{\mathrm{GN}}=(J_L^n)^\top J_L^n.
\]

这个矩阵是局部目标的 Gauss-Newton 近似。对于带对角权重 \(W\) 的一般 residual 目标，精确 Hessian 为

\[
\nabla^2\!\left(\frac{1}{2}\lVert e(q)\rVert_W^2\right)
=J_e(q)^\top WJ_e(q)
+\sum_{i=1}^{m}w_i e_i(q)\nabla^2e_i(q).
\]

Sequential SOCP 保留第一项，省略包含 residual 二阶导数的求和项；约束也只保留一阶模型。论文局部子问题是对论文原始非线性问题的近似。

## 当前 Holosoma sandbox 做了什么

当前主要实验实现是 `src/interaction_mesh_retargeter_sandbox_experimental.py`。它总体沿用论文的逐帧、逐次局部凸近似结构，但具体计算由当前代码定义。

### Interaction mesh 与 Laplacian 局部模型

每一帧使用人体映射关键点和物体采样点构造 Delaunay interaction mesh。物体交互任务在物体局部坐标系中构造网格。源网格生成邻接关系和目标 Laplacian 坐标。机器人关键点与物体点组成当前目标网格。

在当前迭代配置 \(\bar q_n\) 处，代码计算机器人关键点位置 Jacobian \(J_V^n\)，并构造

\[
J_L^n=(L\otimes I_3)J_V^n.
\]

代码直接在配置增量 \(dqa\) 上使用线性化 Laplacian residual：

\[
r_L(dqa)
=J_L^n dqa
+L^{\mathrm{target}}(\bar q_n)
-L^{\mathrm{source}}.
\]

局部 Laplacian 目标为

\[
\left\|W_L^{1/2}r_L(dqa)\right\|_2^2.
\]

因此当前代码仍然使用 Gauss-Newton 曲率 \(2(J_L^n)^\top W_LJ_L^n\)。sandbox 直接写出 reduced residual，删除了原版代码中代数等价的辅助 `lap_var` 变量。根据任务和配置，局部目标还可以包含 nominal tracking、手工 \(Q_{\mathrm{diag}}\) 正则项和时间 smoothness 项。

### 当前代码实际组装的约束

当前 sandbox 的局部约束包括：

- 支撑脚 XY 位置相对前一帧的容差区间；
- 可选 foot-lock 时间窗中的足部 Z 位置约束；
- 当前碰撞候选对的线性化非穿透约束，并允许配置的 penetration tolerance；
- 可选自碰撞约束；
- 活跃配置变量的关节边界；
- \(\|dqa\|_2\leq\texttt{step\_size}\) 的二阶锥 trust region。

当前 sandbox 没有组装论文公式 (3d)/(15e) 中显式的 \(v_{\min},v_{\max}\) 速度边界。论文中的三维支撑脚等式在当前代码中主要实现为带容差的 XY 区间，并可通过单独的 foot-lock 配置增加 Z 约束。描述约束时必须按当前代码表述，不能直接复述论文公式。

### 数值工具、变量和更新

论文正文说明使用 Drake 自动微分处理 \(S^3\) 上的旋转微分。当前 Holosoma sandbox 使用 MuJoCo 计算关于 generalized velocity 的关键点 Jacobian，再通过代码构造的 \(qdot\) 到 \(qvel\) 映射转换成配置坐标 Jacobian。局部问题由 CVXPY 建模并交给 Clarabel 求解。

当前决策变量是选定 `q_a_indices` 上的配置坐标增量。求解后执行加法更新，并对浮动基座四元数归一化：

\[
q_{\mathrm{active}}^+
=q_{\mathrm{active}}+dqa^\star,
\qquad
r^+=\frac{r^+}{\|r^+\|_2}.
\]

当前 sandbox 第一帧最多执行 50 次内层迭代，后续帧最多执行 10 次，并以局部目标值的 `np.isclose` 判定提前停止。每次内层更新后重新计算运动学、Laplacian Jacobian 和非线性碰撞几何。

## 求解器语义

必须将以下三个层次分开：

1. 论文公式 (3) 定义的逐帧非线性目标问题。
2. 论文公式 (15) 或 Holosoma 代码实际构造的局部 Gauss-Newton/约束线性化子问题。
3. 局部问题求解后的配置更新、四元数处理和内层迭代。

一次局部 QP 或 SOCP 求解只得到当前线性化模型的解。它不能表述为直接求得了原始非线性重定向问题的解。局部子问题为二次形式，也不能据此把实现称为精确二阶 Newton 方法。检查代码后，根据实际计算使用“Gauss-Newton 局部模型”“约束线性化 QP/SOCP”或“逐次局部凸近似”等术语。

## 必须区分的内容

解释、审计或修改 retargeter 时，必须明确说明以下内容：

1. 预期满足的非线性目标问题及其目标函数和约束。
2. 局部子问题实际构造的目标模型，包括每个目标项是被精确保留、线性化后平方形成 Gauss-Newton 项，还是被简化为一阶项。
3. 局部子问题实际构造的约束模型，包括精确仿射边界、非线性约束的一阶线性化、碰撞候选对选择和 trust region。
4. 决策变量所在的空间，以及该变体使用的四元数参数化。
5. 求解后的配置更新规则、非线性可行性复查、停止条件和内层迭代次数。

不同变体具有相同的类名或方法名，不足以支持它们采用相同算法的结论。必须检查对应文件中实际构造的目标、约束和更新。

## 论文与代码比较

比较 Holosoma 代码与 OmniRetarget 或其他参考方法时，必须区分以下对象：

- 参考论文定义的非线性优化问题；
- 参考论文采用的局部近似和序列求解方法；
- 所选 Holosoma 变体预期求解的非线性问题；
- 该变体实际构造的局部 QP 或 SOCP；
- 局部求解后实际执行的数值配置更新。

描述算法修改时，必须说明修改影响非线性目标问题、局部近似、更新与迭代方法中的哪些层次。
