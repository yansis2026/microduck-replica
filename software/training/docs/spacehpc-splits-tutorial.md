# 星海智算（SpaceHPC）劈叉训练 · 小白教程

从零开始，在 **星海智算 <https://www.spacehpc.com/>** 上训练 Microduck 的**前劈叉**
（左腿在前）任务 `Mjlab-Splits-Flat-MicroDuck-HD1910`，最后导出可以拷走的 `policy.onnx`。

> 本教程的命令同样适用于劈叉展示任务 `Mjlab-SplitsShow-Flat-MicroDuck-HD1910`
> （劈叉 → 保持 5 s → 原路起身站稳），把任务 ID 替换即可；时间线与奖励见
> [splits-show-task.md](splits-show-task.md)。

本教程使用你自己的仓库和分支（**不是**上游 `master`，上游还没有劈叉代码）：

```text
仓库：https://github.com/yansis2026/microduck-replica.git
分支：feat/splits-training
```

> 全程只需照着复制命令即可。遇到报错先看文末「常见问题」。

---

## 0. 先搞清楚几个词（30 秒）

| 词 | 意思 |
|---|---|
| **iteration（迭代）** | 训练一步。一次迭代 = 几千个虚拟机器人各走 24 个控制步，然后神经网络更新一次。 |
| `num-envs` | 同时开多少个虚拟机器人。越大 GPU 越吃满、显存也越吃紧。 |
| `max-iterations` | 训练多少步。6000 是建议值，曲线走平就可以提前停。 |
| **checkpoint（检查点）** | 训练过程中保存的权重文件 `model_XXXX.pt`。 |
| `policy.onnx` | 训练完导出的神经网络，给鸭子用。 |

一句话：**先在 GPU 上对着 4096 个虚拟鸭子练 6000 轮，挑一个好的 `model_*.pt`，导出成 `policy.onnx`。**

---

## 1. 在星海智算创建实例

### 1.1 注册 / 登录

1. 打开 <https://www.spacehpc.com/> 注册并登录控制台。
2. 新用户通常有礼包，先看清满减券门槛和有效期。

### 1.2 创建 GPU 实例

1. 进入「创建实例」页面。
2. **卡型选 `RTX 4090 24G × 1 卡`**。本任务 4096 环境 24G 足够；A100 80G 只快 1.5–2 倍、
   价格却是 3–4 倍，不建议。
3. **镜像**：选带 **PyTorch 2.x** 的镜像即可。镜像里 Python 不是 3.12 也没关系，后面的 `uv`
   会自己装一个 3.12。
4. **计费方式**（省钱关键）：
   - 第一次只装环境 / 传代码：先用**无卡模式**开机（约 0.2 元/h），把最耗时的依赖装好。
   - 正式训练：用 **4090 按量**（按秒计费，关机即停费，但关机后 GPU 不预留）。
   - 打算连跑几天：再考虑 **包日 / 包月**（折算约 1.78 元/h，且预留 GPU）。
5. 等状态变成「运行中」，页面会给出 **SSH 连接信息**（主机、端口、用户名、密码）。

> 参考价格：4090 按量约 **2.97 元/h**；包月折算约 **1.78 元/h**。价格随分区浮动，以下单页为准。

---

## 2. 连上服务器

用你电脑的终端（Windows 可用 PowerShell / Git Bash；Mac/Linux 用自带终端）执行
星海智算页面给的 SSH 命令，形如：

```bash
ssh -p 12345 root@1.2.3.4
```

第一次连接会问 `yes/no`，输入 `yes` 回车，然后输入密码（输入时屏幕不显示，正常）。

连上后，后面的命令都在服务器里执行。

---

## 3. 安装 uv 并拉代码

### 3.1 安装 uv（Python 包管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

`uv --version` 能打印版本号就成功。

### 3.2 拉你的劈叉分支

```bash
cd ~
git clone -b feat/splits-training https://github.com/yansis2026/microduck-replica.git
cd microduck-replica/software/training
```

> 注意 `-b feat/splits-training`：**一定要拉这个分支**，上游 `master` 还没有劈叉任务代码。

### 3.3 安装训练依赖

```bash
uv python install 3.12
uv sync --frozen --python 3.12
```

这一步会下载 PyTorch / CUDA / MuJoCo 等，**比较久**（视网络 10–30 分钟），耐心等。
后续所有命令都用 `uv run --frozen ...` 在同一个环境里跑。

> 如果卡在从 GitHub 拉 `better-actuator-models`，重试一次；仍失败可尝试给 git 配代理或
> 换网络后重跑 `uv sync --frozen --python 3.12`。

---

## 4. 冒烟测试前：先做纯 CPU 姿态自检

这一步**不需要 GPU**，确认关键帧几何是否合法（双脚着地、躯干/头不触地）：

```bash
uv run --frozen python scripts/check_split_pose.py
```

必须看到最后一行：

```text
OK: all waypoints are kinematically feasible.
```

如果这里失败，**不要继续训练**——先把报错发出来，说明关键帧有问题。

可选：生成一段不看训练策略的动画，直观看看「要练成什么动作」（需要 Pillow，依赖里已包含）：

```bash
uv run --frozen python scripts/preview_split_sim.py --mode kinematic --out out/split_preview
```

产物：`out/split_preview/split_kinematic.gif` 和 `split_kinematic_strip.png`。

---

## 5. 冒烟测试（64 环境 × 5 轮）

只验证「能加载、能步进、能存盘」，几十秒到几分钟：

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard
```

能正常结束、不报错，并且在 `logs/rsl_rl/microduck_splits/` 下生成了目录，就说明链路通了。

> 如果提示参数名不认识，把 `--agent.max-iterations` 换成 `--agent.max_iterations` 再试。
> 5 轮远远学不会动作，它只是「通电测试」。

---

## 6. 正式训练（4096 环境 × 6000 轮）

### 6.1 用 tmux，断线也不中断

直接 `ssh` 跑训练，一旦网络断开训练就会死。用 `tmux`：

```bash
tmux new -s train
```

进入 tmux 后执行正式训练：

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 6000 \
  --agent.logger tensorboard
```

启动后：
- 按 `Ctrl-B`，松开，再按 `D`：**退出 tmux，但训练继续跑**。
- 下次连回服务器后 `tmux attach -t train` 回到训练界面。
- 想看日志又不想进 tmux：`ls -t logs/rsl_rl/microduck_splits/*/`。

> 没有 tmux 也可用：
> `nohup uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.logger tensorboard > train.log 2>&1 &`
> 然后 `tail -f train.log`。

### 6.2 显存不够怎么办

报 `CUDA out of memory` 时，**降环境数**，不要盲目升卡：

```text
4096 → 2048 → 1024
```

环境数低于 ~1024 时 PPO 噪声会变大，尽量保持在 2048 以上。

---

## 7. 可视化训练曲线

训练日志在 `logs/rsl_rl/microduck_splits/`。在服务器里启动 TensorBoard：

```bash
uv run --frozen tensorboard --logdir logs/rsl_rl/microduck_splits --port 6006
# 如果提示找不到 tensorboard，用：
# uv run --frozen python -m tensorboard.main --logdir logs/rsl_rl/microduck_splits --port 6006
```

然后在**你自己的电脑**上开一个 SSH 隧道（另开一个终端，用星海智算给的端口）：

```bash
ssh -p 12345 -L 6006:localhost:6006 root@1.2.3.4
```

保持这个 SSH 窗口不关，浏览器打开 <http://localhost:6006> 就能看曲线。

> 也可以直接用平台自带的 **JupyterLab 终端 / 端口转发**，效果一样。

### 7.1 重点看这些曲线

| 指标 | 期望 |
|---|---|
| `Episode_Reward/split_pose_legs` | **持续上升**（腿部跟踪关键帧） |
| `Episode_Reward/split_height` | 跟随 0.115 → 0.114 → 0.095 → 0.063 |
| `Episode_Reward/split_feet_grounded` | 上升（双脚着地） |
| `Episode_Termination/trunk_ground_contact` | **下降**（躯干触地会被终止） |
| `Episode_Termination/head_ground_contact` | **下降** |
| 所有惩罚项 | 按约定应 ≤ 0 |

### 7.2 什么时候可以停

不必跑满 6000 轮。满足下面几点就可以挑检查点：

- `split_pose_legs` 和 `split_height` 连续几百轮不再明显上升；
- 躯干/头部触地终止接近 0；
- 挑一个「奖励高、终止少」的 `model_XXXX.pt`。

> 如果奖励一直上不去、终止率很高：先确认第 4 步姿态自检仍通过，再看环境数是否太小 / 是否
> 显存 OOM 被降级。

---

## 8. 导出 ONNX

训练会在 `logs/rsl_rl/microduck_splits/<某次运行>/` 下保存 `model_XXXX.pt`。先找到最新一个：

```bash
ls -t logs/rsl_rl/microduck_splits/*/model_*.pt | head -5
```

导出（把 `$CKPT` 换成你选中的检查点，通常是最后一轮 `model_6000.pt`）：

```bash
CKPT=$(ls -t logs/rsl_rl/microduck_splits/*/model_*.pt | head -1)

uv run --frozen python scripts/export.py Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --checkpoint-file "$CKPT" \
  --num-envs 1 \
  --onnx-file policy.onnx
```

成功会打印 `Written /.../policy.onnx`。

说明：
- **观测归一化已经自动写进 ONNX 图里**，不需要你手动处理。
- ONNX 只是一张网络图；它和训练环境的观测顺序、动作顺序一致（见下一节）。

---

## 9. 让鸭子真正用上：ONNX ≠ 能直接导入

**这是最容易踩坑的地方，务必看完。**

训练导出的 `policy.onnx` 是**策略网络**，但 Studio 的「模型管理 → 导入模型」不接受单独一个
ONNX。它要求一个**部署包**（ZIP / tar.gz / tgz），至少要包含 3 个文件：

```text
policy.onnx
manifest.json
deployment-contract.json
```

（可选再放 `deployment-execution.json`、`deployment-runtime.json`、`model-info.json`。）

> **重要事实：本训练仓库的 `export.py` 目前只生成 `policy.onnx`，不会自动生成
> `manifest.json` / `deployment-contract.json`。** 这两个文件描述「关节顺序、Home 姿态、
> 动作缩放、观测项、HD1910 驱动参数」等运行时约定，属于**部署**环节，需要你（或维护者）
> 按下面的规则补上，并通过在本机运行的 `robotd validate-policy` 预检才算兼容。

### 9.1 部署包会被检查什么（来自 `software/studio/studio/model_package.py`）

`manifest.json`：

- `onnx_sha256`：必须等于 `policy.onnx` 的 SHA256；
- `task`：必须等于 `deployment-contract.json` 里的 `task`；
- 可选 `name`（显示名）、`training`（训练信息）。

`deployment-contract.json` 必须满足：

- `schema` = `"microduck-deployment-contract/v1"`
- `task_adapter` ∈ `{ "hd1910-v1", "hd1910-v2", "hd1910-official0151" }`
- `input_dim` = `61`，`output_dim` = `14`，`obs_normalizer` = `"baked_in_onnx"`
- `task`：非空字符串（本任务填 `Mjlab-Splits-Flat-MicroDuck-HD1910`）
- `model.joints`：恰好 14 项，每个关节一条 `{ "name": ..., "home_rad": <有限数> }`
- `onnx_metadata.joint_names`：逗号连接的 14 个关节名（顺序必须与 `policy.onnx` 一致）
- `model.actuators`：恰好 1 项，含
  `parameter_file.sha256 = "ef2d51adfb1cc0831b9b02ca19aa9176fceecdf725148d084378d7d9a64afceb"`
  和整数 `kp_fw`（1–32）
- `actions.joint_pos`：对象
- `observations.terms`：对象

14 个关节的顺序（与训练/`splits_keyframes.py` 的 `JOINT_ORDER` 完全一致）：

```text
left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
neck_pitch, head_pitch, head_yaw, head_roll,
right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle
```

参考实现：`software/studio/tests/test_model_package.py` 里有一份**走路模型**的真实合同
`CONTRACT`，可以照着改 `task` 和关节 Home，再实际验证。
把三个文件放同一目录后打包：

```bash
zip splits_deploy.zip policy.onnx manifest.json deployment-contract.json
# 没有 zip 时：
# tar -czf splits_deploy.tar.gz policy.onnx manifest.json deployment-contract.json
```

`manifest.json` 里的 `onnx_sha256` 可这样算：

```bash
sha256sum policy.onnx
```

### 9.2 真正的兼容性由 `robotd validate-policy` 决定

Studio 导入时会调用本机 `robotd validate-policy` 预检 ONNX。**归档层检查通过不等于真机能用**：
`task_adapter`、Home 姿态、动作/观测映射必须和劈叉策略的实际运行方式匹配。

劈叉任务的运行时接口（来自 [splits-task.md](splits-task.md)）：

- 观测 **61 维**、动作 **14 维**；
- 复用 `twist` 命令槽作为相位时钟：`cmd_twist = [cos(2πp), sin(2πp), 0]`，周期 **9 s**；
- 按钮触发时 `p` 从 0 开始推进，`p ≥ 0.48` 进入保持段。

> **诚实提示**：本分支只提供**训练**代码。劈叉的运行时适配（`task_adapter` 该用哪个、
> 命令槽如何接按钮）不在本分支内，需要以你们的 `robotd` 实际支持的适配为准。如果导入时
> `validate-policy` 因合同/适配不符被拒，把它的报错原样发出来，按报错补合同或先做运行时适配，
> **不要**把「解压成功」当成「能用」。

### 9.3 现状：Studio 还没有「劈叉」槽位（会直接装不进去）

Studio 模型库只认 **7 个固定槽位**（`software/studio/studio/models.py`）：

```text
walk, sitstand, ground_pick, kick_left, kick_right, roulade, stand
```

**没有 splits / stretch 槽。** 而槽位是由部署合同的 `task` 前缀硬推断的
（`software/studio/studio/model_package.py` 的 `_recommended_slot`）：`Mjlab-Splits-...`
不匹配任何前缀，返回 `None`；安装时检测到槽位不在允许列表里会直接拒绝，报：

```text
无法识别这个模型的动作用途，暂不能安装
```

> 在合同里手写 `"slot": "ground_pick"` 也**没用**：非踢球任务的 `slot` 字段会被忽略，
> 槽位仍由 `task` 前缀决定。

也就是说：**`policy.onnx` 本身没问题，但当前这套 Studio 无法把「劈叉」装上真机。**
要在真机验证劈叉，需要运行时侧先支持：

- 新增 `splits` / `stretch` 技能槽；
- `robotd` 能按 `cmd_twist = [cos(2πp), sin(2πp), 0]`、周期 9 s 驱动该策略；
- 一个触发按键 + 持续时间（约 9 s）；
- Studio 的 `SLOTS` / `SKILLS` / `_recommended_slot` 加上对应映射。

这些**都不在当前训练分支里**。所以训练可以照本文跑，但「真机验证」这一步请先和负责
`robotd` / Studio 固件的人确认已部署固件是否已有劈叉槽；没有的话要先做这块运行时适配，
否则会卡在「导入被拒」。不要用「ONNX 能加载」当成「鸭子能跑」。

---

## 10. 把模型拷回来，然后省钱关机

1. 把 `policy.onnx`（以及你组好的部署包）从服务器下载到你电脑，例如在你电脑上执行：
   ```bash
   scp -P 12345 root@1.2.3.4:~/microduck-replica/software/training/policy.onnx .
   ```
   （端口/主机换成星海智算给你的。）
2. 训练日志 `logs/rsl_rl/microduck_splits/` 建议一起拷走，方便以后续训/换检查点。
3. 不用了就**关机**；按量实例关机即停 GPU 费用。长期不再用就**删除实例**，避免存储费。
   - 注意：按量实例关机后 **GPU 不预留**，下次开机可能要重新等卡。
   - 网盘存储约 0.005 元/GB/日，重要检查点别只留在系统盘。

---

## 11. 常见问题

| 现象 | 处理 |
|---|---|
| `uv: command not found` | 执行 `export PATH="$HOME/.local/bin:$PATH"`，或重开终端 |
| `uv sync` 卡在 git 依赖 | 重试；换网络或给 git 配代理后再 `uv sync --frozen --python 3.12` |
| `CUDA out of memory` | 降 `--env.scene.num-envs`（4096 → 2048 → 1024） |
| 参数 `--agent.max-iterations` 不认识 | 改成 `--agent.max_iterations` |
| 训练中途 SSH 断了，训练没了 | 下次一定用 `tmux` 或 `nohup` |
| `check_split_pose.py` 失败 | 不要训练，说明关键帧几何不合法，把报错发出来 |
| 奖励不涨、终止率很高 | 先跑自检；再确认环境数不过小、没有 OOM 降级 |
| TensorBoard 打不开 | 确认 SSH 隧道窗口没关，浏览器访问 `http://localhost:6006` |
| 导入模型被拒 | 缺 `manifest.json` / `deployment-contract.json`，或 `robotd validate-policy` 不通过；按报错补，见第 9 节 |

---

## 12. 最短路径（命令合订版）

```bash
# 0) 连上服务器后，装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 1) 拉你的劈叉分支
cd ~
git clone -b feat/splits-training https://github.com/yansis2026/microduck-replica.git
cd microduck-replica/software/training

# 2) 装环境（Python 3.12）
uv python install 3.12
uv sync --frozen --python 3.12

# 3) 纯 CPU 姿态自检
uv run --frozen python scripts/check_split_pose.py

# 4) 冒烟
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard

# 5) 正式训练（tmux 里，4096×6000）
tmux new -s train
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.logger tensorboard
# Ctrl-B 再按 D 退出；tmux attach -t train 回来

# 6) 看曲线（另开 SSH 隧道窗口）
uv run --frozen tensorboard --logdir logs/rsl_rl/microduck_splits --port 6006
# 本地：ssh -p <端口> -L 6006:localhost:6006 root@<主机>  → http://localhost:6006

# 7) 导出 ONNX
CKPT=$(ls -t logs/rsl_rl/microduck_splits/*/model_*.pt | head -1)
uv run --frozen python scripts/export.py Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --checkpoint-file "$CKPT" --num-envs 1 --onnx-file policy.onnx
```
