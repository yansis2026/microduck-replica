# 原地跳舞任务（Dance）

任务 ID：`Mjlab-Dance-Flat-MicroDuck-HD1910`

稳稳站在地面，**上下起伏、左右交替地小蹲**，中间回到站立位，头也随着左右明显摆动，
像一个带劲的原地律动：

```text
站立 → 下沉小蹲（左膝屈得狠、右膝次之） → 站立 → 下沉小蹲（镜像） → 站立 → …
```

与劈叉这类"单向/往返"技能不同，这是一个**周期性、原地、左右交替**的动作。
执行器、传感器、终止、求解器与观测/动作契约完全复用劈叉任务
（61D 观测、14 维动作，`[twist(3), head(4), body(6)]` 命令块）。

## 时序与关键帧

数据在 [dance_keyframes.py](../src/mjlab_microduck/tasks/dance_keyframes.py)。
周期 `DANCE_PERIOD_S = 2.0 s`，回合 `DANCE_EPISODE_LENGTH_S = 8.0 s`（4 个周期，400 步，整除）。

记 `i = 0..8`（9 个 waypoint，`phase = i/8`），左右相位 `w = sin(2π·i/8)`：
`w` 在两端与半周期为 0（站立），1/4 处为 +1（沉左胯），3/4 处为 −1（沉右胯）。

**每个 sink 是一次小蹲**：两腿的膝盖都屈（大腿和小腿折叠、腿变短），骨盆随之**下沉约 1 cm**；
承重一侧的膝屈得更狠（约 **34°**），另一侧只屈约 **17°**（相对伸直），所以两条腿一深一浅地
交替，身体上下**明显地起落**——这就是要的"激情"幅度。脚掌始终**平贴地面、不移动**，骨盆
向承重侧侧倾约 **3°**。主要可见动作是**上下起伏 + 头左右摆动**：头随节奏左右摆（不是抖动，
这次幅度更大），腿左右交替屈/伸。基座（骨盆）向承重腿一侧做侧倾 `base_roll`（侧倾是双脚
钉地的必然结果，不是关节命令），头摆动 `head_roll`：

```text
base_roll = -BASE_ROLL_MAX · w      head_roll = +HEAD_ROLL_MAX · w
BASE_ROLL_MAX = 0.050292 rad（~2.9°）   HEAD_ROLL_MAX = 0.18 rad（~10.3°）
```

| i | phase | w = sin | base_roll | head_roll | 姿态 | 躯干高 |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 0.000 | 0.0000 | 0.0000 | STAND | 0.11718 m |
| 1 | 0.125 | +0.707 | −0.0356 | +0.1273 | 左膝深屈（过渡下蹲） | 0.11001 m |
| 2 | 0.250 | +1.000 | −0.0503 | +0.1800 | **左膝最屈、右膝次之（蹲得最低）** | 0.10704 m |
| 3 | 0.375 | +0.707 | −0.0356 | +0.1273 | 左膝深屈（过渡下蹲） | 0.11001 m |
| 4 | 0.500 | 0.000 | 0.0000 | 0.0000 | STAND | 0.11718 m |
| 5 | 0.625 | −0.707 | +0.0356 | −0.1273 | 右膝深屈（过渡下蹲） | 0.11001 m |
| 6 | 0.750 | −1.000 | +0.0503 | −0.1800 | **右膝最屈、左膝次之（蹲得最低）** | 0.10704 m |
| 7 | 0.875 | −0.707 | +0.0356 | −0.1273 | 右膝深屈（过渡下蹲） | 0.11001 m |
| 8 | 1.000 | 0.000 | 0.0000 | 0.0000 | STAND（= i0，闭合） | 0.11718 m |

- 左右交替，中间（i4）回到站立；i0 与 i8 同为 STAND，相位时钟 `p % 1` 时循环**无缝**。
- 每个 sink 是**小蹲**：骨盆下沉约 **10 mm**，两只膝都屈，承重膝屈 **~34°**、另一侧屈 **~17°**，
  所以对侧腿相对更长 **~8 mm**（一深一浅、左右交替）。
- 骨盆向承重侧倾 **±2.9°**，承重侧髋部比另一侧低约 **2 mm**；上下起伏是这次的主要"激情"来源。
- 头随律动左右**摆动** **±10.3°**（平滑正弦，比之前幅度大近一倍，不是抖动）。
- `w > 0`（左膝屈得狠）的姿态是 `w < 0`（右膝屈得狠）姿态的**精确镜像**：左右腿对调且各关节角取反。

### 平脚是数值 IK 解出来的（不是矢状和恒等）

一个常见的直觉是"弯膝的同时让 `hip_pitch + knee + ankle` 保持不变"就能让脚掌水平。
**这是错的**：膝一弯，各 pitch 关节的轴就不再平行，用这种"矢状和恒等"会让脚掌翘起、踩到
脚边缘（这正是之前被否决的动作）。所以 9 个 waypoint 的关节角**不是**解析公式，而是用
`scipy.optimize.least_squares` 数值解出来的：

- **把承重腿的膝盖屈到目标**（`0.60 rad ≈ 34°`），另一条腿的膝盖屈 `0.30 rad ≈ 17°`，
- 其余未知量（基座侧倾 `φ`、基座高度 `z`、每条腿的 `hip_yaw/hip_roll/hip_pitch/ankle`）
  一起解，使两脚脚掌法线朝正上方（`sole_up ≈ 1`）且脚底最低点**钉在地面上**，
- 先只**稳解 sink** 这一组姿态（`φ ≈ 2.9°`），再用 0.707（=sin45°）从 STAND 线性缩放得到
  ease 姿态——保证 waypoint 单调平滑、不会跳到别的局部解。

`hip_yaw` 结果被压在 `±YAW_BOUND = ±0.04 rad`（~2.3°）以内，脚掌只轻微原地转动。实测：
9 个 waypoint 脚底离地 ≤ 2.3 mm、`sole_up ≥ 0.997`、水平漂移 ≤ 1.6 mm。结果：脚掌
**贴地不动**，两条膝都明显屈（承重侧更深）、髋-踝距离缩短；对侧腿相对更长，骨盆随双膝下蹲
而明显起落。

> 为什么侧倾只有 ~2.9°：下沉主要走了**双膝小蹲**（矢状面屈膝），而不是靠骨盆侧倾把胯压下去。
> 双脚钉地、骨盆是刚体，两腿屈得不一样多时**仍会**带来一点骨盆侧倾；~2.9° 是个好平衡——
> 既让身体明显上下起落（~1 cm），又不会左右乱晃。

**基座 roll 不是关节命令**：它作用在自由基座上。训练时这个侧倾由物理 + 双脚着地奖励自然
产生（与劈叉一致）；CPU 检查/预览脚本直接把 `DANCE_BASE_ROLL_WAYPOINTS` 施加到基座来复现。

### 骨盆随小蹲明显起落

每个 sink 双膝下蹲，骨盆随之**下沉约 1 cm**（这就是"上下幅度"）。`DANCE_HEIGHT_WAYPOINTS`
是实测的**对称三次起伏**：站立 `0.11718 m` → 过渡 `0.11001 m` → 沉胯 `0.10704 m` → 回到
站立。这是一个明显但受控的垂直律动，不是身高被硬压。

## 检查与预览（纯 CPU，无需 GPU / mjlab）

```bash
python scripts/check_split_pose.py --timeline dance
python scripts/check_split_pose.py --timeline dance --render out/dance
```

检查脚本对跳舞时间线会施加每个 waypoint 存储的 `base_roll` 与高度，逐 waypoint 报告脚掌
sole 法线的 z 分量（`sole_up`，≈1 表示脚掌水平）、两脚离地量、脚掌水平漂移、髋/躯干/头离地。
通过条件：两只脚 `sole_up ≥ 0.99`（脚掌水平，没踩到边）、脚底 ±3 mm、脚掌水平漂移 ≤ 6 mm、
躯干/头部不触地。当前结果：9 个 waypoint 全部通过，`sole_up ≥ 0.997`、脚底 ≤ 2.3 mm、
水平漂移 ≤ 1.6 mm、骨盆侧倾 ≤ 2.9°、骨盆起落幅度 ~10 mm。

本地可视化（只需 CPU MuJoCo + Pillow），相位会**循环**播放（`p % 1`），逐帧对关节、
`base_roll` 与高度做线性插值，脚掌保持贴地：

```bash
python scripts/preview_split_sim.py --mode kinematic --timeline dance --out out/dance_preview
```

输出 `out/dance_preview/dance_kinematic.gif` 与 `dance_kinematic_strip.png`（帧上标注
t/phase/trunk_z）。它播放的是**策略被要求学会的目标动作**，不是训练后的策略；`--mode pd`
可作开环伺服对照。

## 训练

需要 Linux + NVIDIA GPU，或配置了 GPU 的 WSL2（BAM 要求 Python < 3.13，训练需要 CUDA）：

```bash
cd software/training
uv sync --frozen --python 3.12

# 先用少量环境检查链路
uv run --frozen train Mjlab-Dance-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 \
  --agent.logger tensorboard

# 正式训练
uv run --frozen train Mjlab-Dance-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 4000 \
  --agent.logger tensorboard
```

日志目录：`logs/rsl_rl/microduck_dance/`。

TensorBoard 重点：

- `split_pose_legs` 与 `split_pose_neck` 呈**周期性**波动（每 100 步一个周期，左右交替）；
- `split_height` 呈**周期性**起伏：站立 ~0.117 m、沉胯时 ~0.107 m（每个 sink 下沉约 1 cm，
  左右各一次）；
- `split_feet_grounded` 全程为正；所有惩罚项 ≤ 0；
- 回合约 8 s、4 个周期。

测试（在 Linux/WSL 的 GPU 环境）：

```bash
uv run --frozen --with pytest python -m pytest -q \
  tests/test_dance_cfg.py tests/test_dance_geometry.py
```

## 奖励与终止

复用劈叉任务的相位奖励，只把跟踪目标换成舞蹈时间线，并去掉"静止保持"项：

| 奖励项 | 权重 | 说明 |
|---|---|---|
| `split_pose_legs` | 8.0 | 腿部 10 关节按相位跟踪（沉胯/伸腿） |
| `split_pose_neck` | 1.5 | 颈/头跟踪（含 `head_roll`，故头部左右摆也被跟踪） |
| `split_pose_l1` | 0.5 | 远距离时的恒定梯度 |
| `split_height` | 2.0 | 躯干高度按相位起伏（站立 0.117 m ↔ 沉胯 0.107 m） |
| `split_feet_grounded` | 1.5 | `start_phase = 0`，全程奖励双脚着地 |
| `upright` | 1.0 | 躯干直立（std 0.5，~7° 侧倾几乎不惩罚） |
| `body_ang_vel` | -0.05 | 躯干角动量 |
| `action_rate_l2` | -0.05 | 动作平滑 |
| `joint_torques_l2` | -2e-3 | 力矩正则 |
| `angular_momentum` | -0.01 | 角动量正则 |
| `trunk_impact_penalty` / `head_impact_penalty` | -2.0 | 躯干/头部触地受力惩罚 |

> 没有 `split_hold_stillness`：连续律动没有静止段，静止奖励会与律动相冲突。

终止沿用劈叉/速度任务：`trunk_ground_contact`、`head_ground_contact`、`nan_state`，
以及通用 `fell_over`。符号约定见 `AGENTS.md`（代价函数返回正值，故权重为负）。

## 运行时接口

与所有 microduck 任务相同，`twist` 命令槽是**可部署的相位时钟**：

```text
cmd_twist = [cos(2πp), sin(2πp), 0]
```

`p` 从 0 线性增长，**周期 2 s**，并在模块 1 内**循环**（与劈叉"不回绕"不同，这正是连续
舞蹈需要的）：

- 按下"跳舞"按钮时把 `p` 置 0 并开始推进；
- 每 `p` 增长 1.0 即完成一次"沉左 → 站立 → 沉右 → 站立"；
- 只要持续推进，机器人就一直循环跳舞。

观测与动作维度与其它 microduck 策略一致（61D 观测、14 维动作）。
模型导出仍用 `scripts/export.py`，输出 `policy.onnx`（含观测归一化）。

## 导出与部署

导出命令（按任务 ID 取）：

```bash
uv run --frozen export Mjlab-Dance-Flat-MicroDuck-HD1910 \
  --checkpoint logs/rsl_rl/microduck_dance/<run>/model_XXXX.pt \
  --output exports/dance/policy.onnx
```

> 注意：手机控制台的 Studio 目前只有 7 种固定动作槽位
> （walk / sitstand / ground_pick / kick_left / kick_right / roulade / stand），
> **没有 dance 槽位**，因此即使 `manifest.json` / `deployment-contract.json` 填写正确也无法导入。
> 需要先在 Studio 增加 dance 槽位才能部署到实机。
