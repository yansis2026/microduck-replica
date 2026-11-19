# 云端训练教程（从零到拿到 ONNX）

本文讲清楚三件事：**训练到底在干什么**、**在哪台机器上怎么敲命令**、**大概花多少钱**。
以劈叉任务 `Mjlab-Splits-Flat-MicroDuck-HD1910` 为例，换成走路任务时只改任务 ID。

> 一句话版本：进入 `software/training`，先跑 64 环境 × 5 轮的冒烟测试，再用
> `--env.scene.num-envs 4096 --agent.max-iterations 6000` 正式训练；云上用 HF Jobs
> 最省心；国内优云智算便宜但经常没卡，星海智算的 4090 性价比高；训练完把 `exported/policy.onnx` 拷到机器人。

---

## 1. 训练在干什么（先搞懂再开机器）

强化学习训练不是“跑一个仿真视频”，而是：

1. 同时开几千个虚拟 Microduck（`num-envs`，比如 4096 个）；
2. 每个虚拟机器人按当前策略动 24 个控制步（`num_steps_per_env=24`，每步 0.02 s，共 0.48 s）；
3. 按奖励函数给这一小段动作打分（劈叉任务看腿是否贴关键帧、躯干高度、双脚是否着地等）；
4. PPO 根据分数更新神经网络；
5. 重复第 2–4 步，这就是一个 **iteration（迭代）**。

所以：

- **`num-envs` 越大**：一次收集的数据越多、GPU 越吃满，但显存也越吃紧。
- **`max-iterations` 越大**：训练越久。6000 轮是建议值，不是硬性要求；奖励曲线走平就可以提前停。
- 训练产出的是**神经网络权重**（`model_*.pt`），最后导出成 `policy.onnx` 给机器人用。

任务入口和关键帧在：

- 任务配置：[../src/mjlab_microduck/tasks/microduck_splits_env_cfg.py](../src/mjlab_microduck/tasks/microduck_splits_env_cfg.py)
- 关键帧数据：[../src/mjlab_microduck/tasks/splits_keyframes.py](../src/mjlab_microduck/tasks/splits_keyframes.py)
- 任务说明（奖励、终止、部署接口）：[splits-task.md](splits-task.md)

---

## 2. 训练前必须做的检查（10 分钟，省几小时）

### 2.1 代码就位

```bash
git clone https://github.com/fanhao375/microduck-replica.git
cd microduck-replica/software/training
```

如果已经有仓库，直接 `cd` 进去即可。后面所有命令都在 `software/training/` 下执行。

### 2.2 先做纯 CPU 的姿态检查（不需要 GPU）

```bash
python scripts/check_split_pose.py
```

看到 `OK: all waypoints are kinematically feasible.` 才能继续。它会检查每个关键帧双脚是否着地、躯干/头部是否触地。
这一步在 Windows 原生 Python 上也能跑（只需要 `mujoco`、`numpy`）。

### 2.3 确认训练依赖（需要 Linux 或配好 GPU 的 WSL2）

- Python **3.12**（项目依赖要求 `>=3.12,<3.13`）
- NVIDIA 驱动 + CUDA（项目用 CUDA 版 PyTorch）
- 已安装 [uv](https://docs.astral.sh/uv/)

> **Windows 原生 Python 3.14 不行**：既有 Python 版本限制，也没有 CUDA 训练环境。
> 要么用 WSL2，要么直接用云 GPU（推荐，见第 3 节）。

安装依赖：

```bash
uv sync --frozen --python 3.12
```

### 2.4 冒烟测试：64 个环境跑 5 轮

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard
```

这一步只验证“能加载、能步进、能保存”，**5 轮远不足以学会动作**。
日志在 `logs/rsl_rl/microduck_splits/`。

> 参数名如果提示无法识别，把 `--agent.max-iterations` 换成
> `--agent.max_iterations`（两种写法在不同版本的 CLI 里都出现过）。

---

## 3. 四种云端方案对比

| 方案 | 适合谁 | 单卡价格（约） | 优点 | 缺点 |
|---|---|---|---|---|
| **A. HF Jobs** | 想最省心、有外币卡/PayPal | L4 24G **$0.80/h**；A10G 24G **$1.50/h**；A100 80G **$2.50/h** | 一条命令提交，自动传源码、自动存检查点、自动导出 ONNX，按分钟计费 | 需要 HF 账号 + Jobs 额度；国内网络一般；无卡排队时也要等镜像 |
| **B. 优云智算 Compshare** | 国内用户、要便宜、能接受抢卡 | 4090 24G **1.92–2.15 元/h**（抢占约 **1.31**）；A100 80G **10.75 元/h**；无卡模式 **0.15 元/h** | 国内延迟低、按量/抢占便宜、数据盘持久 | **经常没资源**；需要自己装环境、自己管进程；关机后保数据仍收无卡费 |
| **C. 星海智算 SpaceHPC** | 国内用户、想要 4090 且能包日/包月 | 4090 24G **2.97 元/h**（包月折算约 **1.78**）；A100 80G **11.10 元/h**（折算约 **6.66**）；无卡模式约 **0.2 元/h** | 4090 单价低；包日/包月**预留 GPU**；按秒计费、关机即停费 | 需要自己装环境；按量关机后不预留 GPU；价格随区域浮动 |
| **D. RunPod / Vast.ai** | 有外币卡、想要备份 | 4090 **$0.34–0.89/h**；L4 **$0.39–0.59/h**；A100 80G **$1.39–1.79/h** | 按秒计费、卡型多、社区卡便宜 | 社区卡稳定性一般；同样要自己配环境 |

> 价格会变，**下单页显示为准**。下面每一步都写成可直接照做的操作。

### 方案 A：HF Jobs（推荐先用这个）

仓库已经把 HF Jobs 封装好了（见 [../scripts/hf/README.md](../scripts/hf/README.md)），
你只需要在正常训练命令后面加 `--hf-jobs`。

**一次性准备：**

1. 注册/登录 Hugging Face，创建 **fine-grained token**，务必勾选 **Jobs** 权限：
   <https://huggingface.co/settings/tokens>
2. 在本地缓存 token（二选一）：
   ```bash
   hf auth login          # 有 hf CLI 时
   # 或者
   export HF_TOKEN=hf_xxx
   ```
3. 登录 wandb（训练曲线会实时同步）。**注意：不登录 wandb 时必须在命令里加
   `--no-wandb`，否则提交脚本会直接报错退出。**
   ```bash
   wandb login
   ```
4. 确认账户有 Jobs 额度（免费额度很少，正式训练通常要充值或开 Pro）：
   <https://huggingface.co/settings/billing>

**先看它准备提交什么（不花钱、不占卡）：**

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 \
  --hf-jobs --flavor l4x1 --dry-run
```

**正式提交（会一直打印日志，Ctrl-C 只是断开查看，任务不会停）：**

```bash
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 6000 \
  --agent.logger tensorboard \
  --hf-jobs --flavor l4x1 --timeout 12h
```

常用参数：

| 参数 | 作用 |
|---|---|
| `--flavor l4x1` | 默认，最便宜；OOM 时不要升卡，先降 `num-envs` |
| `--flavor a10g-large` | 比 L4 快，价格约 2 倍 |
| `--flavor a100-large` | 80G 显存，想开更大 `num-envs` 时用 |
| `--timeout 12h` | 超时自动杀任务，防止忘记关机烧钱 |
| `--detach` | 提交后立刻返回，不盯日志 |
| `--run-name split-v1` | 自定义仓库名，方便找 |
| `--no-wandb` | 不转发 wandb（同时也不要开 wandb logger） |

**提交后会发生什么（不用你管）：**

- 源码打包上传到你的私有数据集 `<你的用户名>/mjlab-microduck-src`；
- 检查点每 60 秒自动推到私有模型仓库 `<你的用户名>/<run-name>`；
- 训练成功后自动导出 `exported/policy.onnx` 并上传；
- 终端会打印任务 URL 和日志；断线后可以用 Python API 重新看：
  ```python
  from huggingface_hub import HfApi
  api = HfApi()
  api.list_jobs()
  for line in api.fetch_job_logs(job_id="你的任务ID", follow=True):
      print(line)
  ```

**省钱要点：** 镜像拉取 + `uv sync` 通常要 10–20 分钟，这段时间也计费；所以**别反复提交小任务**，先用 64 环境本地冒烟，云端一次跑够。

### 方案 B：优云智算 Compshare（国内便宜，但要抢卡）

你给的控制台地址是创建实例页：
<https://console.compshare.cn/light-gpu/resources/create?ImageId=compshareImage-1t1vuerl80np>

这个平台的问题是**资源经常被用光**，所以策略比命令更重要：

**B1. 先用“无卡模式”把环境准备好（0.15 元/小时）**

没卡的时候也能创建实例（无卡模式），它会保留系统盘/数据盘，但没有 GPU。用它来做：

```bash
# 进入实例后
git clone https://github.com/fanhao375/microduck-replica.git
cd microduck-replica/software/training

# 先安装 uv（如果镜像里没有）
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 只下载/安装依赖，不需要 GPU
uv sync --frozen --python 3.12
```

这一步把最耗时的依赖安装做掉。**注意：无卡模式没法跑训练，只能准备环境。**

**B2. 有卡后开机，先测速再正式跑**

镜像建议选 **PyTorch 2.x + CUDA 12.x**、**Python 3.12**（或自己用 uv 建 3.12 环境）。
卡型优先级：

1. **RTX 4090 24G**（性价比最高，按量约 2 元/h）
2. 4090 48G（显存更大，约 3.3 元/h）
3. A100 80G（贵，约 10.75 元/h，只有确实需要大显存时才用）

```bash
cd microduck-replica/software/training

# 冒烟
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard

# 正式训练（用 tmux，SSH 断开也不中断）
tmux new -s train
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 6000 \
  --agent.logger tensorboard
# Ctrl-B 然后 D 退出 tmux；下次 tmux attach -t train
```

> 没有 tmux 就用 `nohup ... > train.log 2>&1 &`，然后 `tail -f train.log`。

**B3. 抢不到卡的应对（按顺序试）**

1. **错峰**：白天/工作日通常更紧张，晚上和清晨更容易出卡；
2. **换卡型**：4090 没有就试 4090 48G、A10、L4 等；
3. **换可用区**：同卡型不同可用区库存不同；
4. **用抢占式（Spot）**：4090 约 1.31 元/h，但**可能被随时回收**——必须勤存检查点、勤上传；
5. **先无卡占位准备环境**，有卡再开机（B1）；
6. **多刷创建页**，有人释放就能捡到。

**B4. 关机 ≠ 免费，数据盘会一直计费**

- 训练完先确认 `logs/rsl_rl/.../model_*.pt` 已经拷出来或上传走；
- 不用了就**释放实例**（不是只关机）；只关机且保留数据会按无卡模式约 0.15 元/h 继续计费；
- 数据盘/网盘/镜像有单独的存储费，长期存放前看清计费项。

**B5. 防丢检查点（抢占式必做）**

训练日志里会定期写 `model_*.pt`，用另一个终端定时同步到数据盘或对象存储：

```bash
# 每 5 分钟同步一次（示例）
while true; do
  rsync -a logs/rsl_rl/microduck_splits/ /root/persist/splits_logs/
  sleep 300
done
```

HF Jobs 的方案不用这一步（它自带上传）。

### 方案 C：星海智算 SpaceHPC（国内，4090 性价比高）

平台：<https://www.spacehpc.com/>

**推荐卡型：RTX 4090 24G。** 4096 个环境用 24G 显存足够；A100 80G 实际训练通常只快
1.5–2 倍，价格却是 3–4 倍，这个任务不建议上 A100。

| 卡型 | 标价 | 优惠折算价 |
|---|---:|---:|
| **RTX 4090 24G（推荐）** | **2.97 元/h** | **1.78 元/h**（包月折算） |
| RTX 4090D 24G | 2.93 元/h | 1.76 元/h |
| RTX 3090 24G（特惠） | 1.97 元/h | 1.18 元/h |
| “顶级大显存” 48G | 4.47 元/h | 2.68 元/h |
| A100 80G | 11.10 元/h | 6.66 元/h |

> 价格随区域/分区浮动，以下单页和帮助文档为准。

**推荐套餐与时长（这是本节的重点）**

| 使用场景 | 推荐套餐 | 建议时长 | 费用（约） |
|---|---|---|---|
| 只装环境、传代码 | **无卡模式**开机 | 1–2 h | 0.2–0.4 元 |
| 首次跑通 / 单次正式训练 | **4090 按量**（按秒计费） | 24–36 h 预算（13–35 GPU 小时） | 39–105 元 |
| 反复调参 / 连跑几天 | **4090 包日或包月**（折算 1.78 元/h，且预留 GPU） | 2 天–1 个月 | 约 23–65 元/次训练 |
| 确实需要大显存 | 48G 大显存或 A100 | 按实测 | 35–250 元 |

**为什么这样选：**

- 按量：灵活、关机即停费，但**关机后 GPU 不预留**，下次开机没卡就要等；
- 包日/包月：贵一点但**预留 GPU**、折算单价最低，训练超过约 24 小时就比按量划算；
- 6000 轮参考需要 13–35 GPU 小时，第一次建议按 1.5 倍留预算，别一次买太久。

**操作步骤**

1. 注册并登录控制台；新用户通常有 50 元礼包，先看清满减券门槛和有效期。
2. 创建 GPU 实例：区域选离你近的，卡型 **RTX 4090 24G × 1 卡**（平台 1 卡起租），
   计费方式按上表选。
3. 镜像选 **PyTorch 2.x**。镜像里没有 Python 3.12 也没关系，uv 会自己装。
4. 等状态变成“运行中”（通常几分钟），用 **SSH** 或 **JupyterLab 终端**连接。
5. 装环境（无卡模式也可以先做这一步）：

   ```bash
   git clone https://github.com/fanhao375/microduck-replica.git
   cd microduck-replica/software/training
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"
   uv python install 3.12
   uv sync --frozen --python 3.12
   ```

6. 先冒烟，再用 tmux/nohup 正式训练（命令与 Compshare 相同）：

   ```bash
   tmux new -s train
   uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
     --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.logger tensorboard
   # Ctrl-B 再按 D 退出；tmux attach -t train 回来
   ```

7. 训练完把 `logs/rsl_rl/microduck_splits/` 里的 `model_*.pt` 通过网盘/SCP 拷走，
   然后关机；不再用就删除实例。

**注意事项**

- 按量实例关机后 **GPU 不预留**；包日/包月才会预留。
- 网盘存储约 **0.005 元/GB/日**，比长时间挂着无卡实例更划算；重要检查点别只留在系统盘。
- 训练中途有被回收的风险（按量更明显），定时把 `model_*.pt` 同步到网盘。

### 方案 D：RunPod / Vast.ai（备用）

适合 HF、Compshare、星海智算都用不了时。流程和 Compshare 基本一样：

1. 选 **RTX 4090 24G** 或 **L4 24G**，镜像选 PyTorch 2.x + CUDA 12.x；
2. 用官方模板或自己 `git clone` + `uv sync --frozen --python 3.12`；
3. 同样先 64 环境冒烟，再 4096 环境正式训练；
4. **训练完立刻停止 Pod**，按秒计费不等于不收费。

社区卡（Community Cloud）更便宜但不稳定，抢占式同样要勤存检查点。

---

## 4. 训练中看什么、什么时候停

用 TensorBoard 看：

```bash
tensorboard --logdir logs/rsl_rl/microduck_splits
```

重点指标（劈叉任务）：

| 指标 | 期望 |
|---|---|
| `Episode_Reward/split_pose_legs` | **持续上升**（腿部跟踪关键帧） |
| `Episode_Reward/split_height` | 跟随 0.115 → 0.114 → 0.095 → 0.063 |
| `Episode_Reward/split_feet_grounded` | 上升 |
| `Episode_Termination/trunk_ground_contact` | **下降**（躯干触地会终止） |
| `Episode_Termination/head_ground_contact` | **下降** |
| 所有惩罚项 | 按约定应 ≤ 0 |

**什么时候可以停：**

- `split_pose_legs` 和 `split_height` 连续几百轮不再上升；
- 躯干/头部触地终止接近 0；
- 不必跑满 6000 轮。挑一个“奖励高且终止少”的检查点即可。

如果一开始奖励就上不去、终止率很高，先别加轮数，回看：
环境数是否太小（<1024 时 PPO 方差大）、显存是否 OOM 导致降级、关键帧几何是否仍然通过 `check_split_pose.py`。

---

## 5. 训练完：拿模型、导出、部署

### 5.1 HF Jobs 用户

训练成功后，模型仓库里会直接有：

```text
<你的用户名>/<run-name>/
  model_XXXX.pt
  exported/policy.onnx     ← 自动导出，直接可用
```

查看/下载：

```bash
# 新版命名（装了 hf CLI）
hf download <你的用户名>/<run-name> --local-dir ckpts

# 旧版命名
huggingface-cli download <你的用户名>/<run-name> --local-dir ckpts
```

### 5.2 Compshare / 星海智算 / RunPod 用户

检查点在 `logs/rsl_rl/microduck_splits/` 下，先把最新的 `model_*.pt` 拷到本地，再导出：

```bash
uv run python scripts/export.py Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --checkpoint-file model_6000.pt \
  --num-envs 1 \
  --onnx-file policy.onnx
```

导出会把观测归一化一起写进 ONNX（见 [../scripts/export.py](../scripts/export.py)）。

### 5.3 部署到机器人前确认

- 观测 **61 维**、动作 **14 维**（与其它 microduck 策略一致）；
- 命令槽用相位时钟：`cmd_twist = [cos(2πp), sin(2πp), 0]`，周期 **9 s**；
- 按下“劈叉”按钮时把 `p` 置 0 并开始推进；`p ≥ 0.48` 进入保持段；
- 关节顺序见 [splits-task.md](splits-task.md) 和 `splits_keyframes.py` 的 `JOINT_ORDER`。

---

## 6. 费用到底多少

### 6.1 先算“需要多少 GPU 小时”

1 个 iteration = `num-envs × 24` 步。4096 环境就是 **98,304 步/轮**。

实际速度**必须自己测**（不同任务、不同卡差异很大）：

```bash
# 正式训练跑 20–50 轮，看日志里每轮耗时；或用 time 包一下
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 30 --agent.logger tensorboard
```

然后：

```text
GPU 小时 ≈ 每轮秒数 × 6000 ÷ 3600  +  0.3（启动/装依赖）
```

参考量级（mjlab 官方在 RTX 4090 上跑 Go1 平地约 **8,000 步/秒**；劈叉任务接触更重，可能更慢）：

| 每轮耗时 | 6000 轮约 | 说明 |
|---|---|---|
| 8 s | 13.3 h | 很快的卡/较省的环境 |
| 12 s | 20 h | 4090 量级，常见 |
| 20 s | 33 h | 接触复杂或卡较慢 |
| 30 s | 50 h | 明显偏慢，建议降环境数或提前停 |

> 结论：**按 13–35 GPU 小时做预算**，再用实测校正。别把 6000 轮当成必须跑满。

### 6.2 换算成钱（大约）

按上面的 13–35 GPU 小时：

| 平台 / 卡 | 单价 | 13 h | 35 h |
|---|---|---:|---:|
| HF Jobs l4x1（L4 24G） | $0.80/h | ~$10 | ~$28 |
| HF Jobs a10g-large | $1.50/h | ~$20 | ~$53 |
| HF Jobs a100-large | $2.50/h | ~$33 | ~$88 |
| Compshare 4090 24G | ~2.0 元/h | ~26 元 | ~70 元 |
| Compshare 4090 抢占 | ~1.31 元/h | ~17 元 | ~46 元 |
| Compshare A100 80G | ~10.75 元/h | ~140 元 | ~376 元 |
| 星海智算 4090 24G（按量） | ~2.97 元/h | ~39 元 | ~104 元 |
| 星海智算 4090 24G（包日/包月折算） | ~1.78 元/h | ~23 元 | ~62 元 |
| 星海智算 A100 80G（折算） | ~6.66 元/h | ~87 元 | ~233 元 |
| RunPod 4090（Secure） | ~$0.74/h | ~$10 | ~$26 |
| RunPod A100 80G | ~$1.79/h | ~$23 | ~$63 |

另外加：

- 启动/拉镜像/装依赖：HF 约 10–20 分钟；Compshare 无卡准备另算 0.15 元/h；星海智算无卡模式约 0.2 元/h。
- 存储：Compshare 关机保数据约 0.15 元/h；星海智算网盘约 0.005 元/GB/日；HF 私有仓库免费额度内通常不额外收费。
- **试错成本**：第一次训练大概率要调奖励/环境数，建议预算 **1.5–2 倍**。

### 6.3 推荐预算

- **首次跑通（只求链路通）**：HF l4x1 跑 1–2 小时，约 **$1–2**；或 Compshare 无卡准备 + 1 小时 4090，约 **2–5 元**；星海智算无卡准备 + 1 小时 4090，约 **3–6 元**。
- **正式训练一版**：**$15–40**（HF）、**30–100 元**（Compshare 4090）或 **39–105 元（星海按量）/ 23–62 元（星海包日/包月折算）**。
- **想反复调参**：优先 Compshare 抢占式或星海智算包日/包月 + 勤存检查点，或 HF 的 A10G。
- **星海智算的推荐组合**：无卡模式装环境 → 4090 按量跑第一次（测速）→ 确认要连跑再转包日/包月，兼顾便宜和“关机也预留 GPU”。

---

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| Windows 上 `uv sync` 或训练失败 | 用 WSL2 或云 GPU；项目要求 Python 3.12 + CUDA |
| `CUDA out of memory` | 降 `--env.scene.num-envs`（4096 → 2048 → 1024），不要盲目升卡 |
| HF 提交报 **402 Payment Required** | Jobs 额度不足：<https://huggingface.co/settings/billing> 充值或开 Pro |
| HF 提交报 **403 Forbidden** | token 没有 Jobs 权限：重建 fine-grained token 并勾选 Jobs，再 `hf auth login` |
| HF 任务一直 `SCHEDULING` | 排队/拉镜像，属正常；超时或挂载失败脚本会自动重试 |
| Compshare 创建页没卡 | 错峰、换卡型/可用区、用抢占式、先无卡模式准备环境 |
| 星海智算关机后开机没卡 | 按量实例不预留 GPU；改用包日/包月，或换卡型/区域 |
| 星海智算想省钱装环境 | 关机后选无卡模式开机（约 0.2 元/h） |
| 训练中断但想接着用 | 保留 `logs/` 与 `model_*.pt`；本仓库未封装一键续训参数，以 `uv run train --help` 为准；更重要的是勤保存/勤上传 |
| 关节状态出现 NaN | 劈叉任务已调高 `nconmax` 和求解器迭代；仍出现就降环境数并检查关键帧 |
| 奖励不涨 | 先确认 `python scripts/check_split_pose.py` 通过；再检查终止率是否过高、环境数是否太小 |

---

## 8. 最短路径速查

```bash
# 0) 代码
cd microduck-replica/software/training

# 1) 本地纯 CPU 检查姿态（不需要 GPU）
python scripts/check_split_pose.py

# 2) 装依赖（Linux/WSL2/云）
uv sync --frozen --python 3.12

# 3) 冒烟
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 64 --agent.max-iterations 5 --agent.logger tensorboard

# 4a) HF Jobs 正式训练
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 \
  --agent.logger tensorboard --hf-jobs --flavor l4x1 --timeout 12h

# 4b) 或云主机上正式训练（tmux 里跑）
uv run --frozen train Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --env.scene.num-envs 4096 --agent.max-iterations 6000 --agent.logger tensorboard

# 5) 导出 ONNX（HF Jobs 会自动做）
uv run python scripts/export.py Mjlab-Splits-Flat-MicroDuck-HD1910 \
  --checkpoint-file model_6000.pt --num-envs 1 --onnx-file policy.onnx
```

---

### 参考

- HF Jobs 价格：<https://huggingface.co/docs/hub/en/jobs-pricing>
- 优云智算计费：<https://www.compshare.cn/docs/operation/charge/bill>
- 优云智算价格列表：<https://www.compshare.cn/price-list>
- 星海智算官网：<https://www.spacehpc.com/>
- 星海智算价格页：<https://spacehpc.com/hpc/pricing>
- 星海智算充值计费说明：<https://help.spacehpc.com/document/recharge_billing/recharge_billing.html>
- 星海智算知识库（实例/镜像/数据盘）：<https://help.spacehpc.com/document/user_data/summarize/summarize.html>
- RunPod 价格：<https://www.runpod.io/pricing>
- mjlab 性能讨论（RTX 4090 Go1 平地约 8,176 steps/s）：<https://github.com/mujocolab/mjlab/discussions/220>
