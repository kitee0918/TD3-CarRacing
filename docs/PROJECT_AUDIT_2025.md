# TD3 CarRacing 项目梳理（2025 年课业版本）

核查范围：`TD3-CarRacing.git/`、`td3_car/` 的本地文件、Git 提交与实验 CSV。时间均为 Git 记录中的北京时间（UTC+8）；文件修改时间只用于给未提交的代码排序，不等于提交时间。本文没有运行训练或重新评估模型。

## 结论

- **有效项目主体是 2025 年的改进与实验。** `TD3-CarRacing.git/` 的 Git 历史只有 2023-06-01 的上游提交；其 2025-10-29 至 2025-11-02 的大量改进在当前仓库中属于未提交文件。`td3_car/` 是 2025-11-18 至 2025-11-23 有连续 Git 提交的后续课业实验仓库。
- 2023 年仓库的 `car_racing_TD3.py` 与 `td3_car` 首次提交的同名文件文本相似度约 **96%**。这支持“从开源基线继续改进”的叙述，不能据此把基线实现归为个人从零完成。上游仓库的 `LICENSE` 是署名 Yanran Zhang 的 MIT 许可证；公开时应保留许可证和来源说明。
- 最接近项目描述的主线是 `td3_car/optimal_d_csdn.py`、`optimal_d.py`、`optimal_d_ctn.py`，配合 `runs/experiment_log.csv`、`demo_run.py`、`read_csv.py`。较早的并行训练、测速和网格搜索探索在 `TD3-CarRacing.git/` 中。
- **“约 200 Episodes 后最佳测试 Score > 800”目前不能按原句引用。** 本地 `td3_car/runs/experiment_log.csv` 在 `run_000_decay_0.990000` 的第 200 轮记录 `best_reward_so_far=865.65`，同轮 `episode_reward=659.32`；远端分支已提交的另一份同名实验 CSV 在第 200 轮记录历史最佳 **803.70**，同轮回报 **591.53**。两份记录的回放容量等配置不同；这些数值是训练历史最佳，不是独立测试分数。独立评估函数会打印结果并录视频，但未找到可核对的评估分数表。
- 上述高分脚本中 `actor_update_freq=1`，即每次 Critic 更新也更新 Actor；因此不能把该高分运行描述为使用了 **Delayed Policy Update**。其他版本（如 `optimal_d.py`、`optimal_d_ctn.py`、较早的 `car_racing_best.py`）实现了延迟更新。

## 两个文件夹的时间线

| 时间 | 文件夹与依据 | 改动/阶段 | Git 状态 |
| --- | --- | --- | --- |
| 2023-06-01 | `TD3-CarRacing.git/`，`d57bb87`、`43dfcab`、`6b500fe` | 上游初始仓库、单文件 TD3 与 MIT 许可证 | 已提交；作者为 Yanran Zhang |
| 2025-10-29 | `TD3-CarRacing.git/car_racing_TD3.py` 及 `Td3-car Racing-*.py` 的文件时间 | 将基线转向 Gymnasium/CarRacing-v3，尝试并行训练和评估脚本 | 当前只有 `car_racing_TD3.py` 未提交修改，其余新增脚本未提交 |
| 2025-10-30 至 11-02 | `TD3-CarRacing.git/` 的 `car_racing_best.py`、`td3_car_racing_train_*.py`、`td3_*benchmark.py`、`td3_gridsearch_*.py` | 并行环境、帧预处理、训练速度基准、自动评估和超参数搜索；部分网格搜索 CSV 记录超时与编码错误 | 这些 2025 文件和运行产物在该仓库均未提交 |
| 2025-11-18 | `td3_car/`，`def5433`、`0deeb8e`、`323f40b`、`08b76a0` | 新建 Gitee 项目、导入基线、加入 `optimal_d.py`；提交说明记录“只能跑200分，无中心奖励” | 已提交 |
| 2025-11-19 | `f74347d`、`9638a81`、`4213b30` | “较好的一版，除了急转弯”；预训练/网格搜索；急弯逻辑 | 已提交于不同分支 |
| 2025-11-20 | `506f2b7`、`b464d60`、`00d9b83`、`2ba967c`、`27ae200`、`75a8079`、`f8f6da4` | Gymnasium 迁移、道路/草地视觉阈值与 ROI、`optimal_d_ctn.py` 超参数搜索、油门刹车调整 | 已提交于不同分支；`demo_pause_tune_rgb.py` 是相关但未提交的本地工具 |
| 2025-11-21 | `0818481`、`d09138b`、`e865588`、`4ebd7b3` | `optimal_d_csdn_hyper.py`，多实验、噪声衰减搜索、CSV、TensorBoard、分运行目录、评估录像；ReplayBuffer 压缩；演示脚本 | 主体代码已提交；本地运行产物不全在当前 HEAD |
| 2025-11-22 至 11-23 | `5283156`、`a852f72`、`e09a2a6`、`0822636`、`64da20e`、`a369aa2`、`3fff80f` | 驶离道路惩罚、动作映射实验与回退、展示模型；`64da20e` 首次将一批 CSV、TensorBoard 文件和 checkpoint 提交至远端分支 | 当前本地 `kite_long_obsv` 停在 `e09a2a6`，**落后** `origin/kite_long_obsv` 5 个提交；不要把当前工作树与远端最新分支视为同一版本 |

补充：`5283156` 的作者时间为 2025-11-20 23:13，提交时间为 **2025-11-22 16:41**；上表按实际提交阶段归类。两个目录的 2026-10-19 文件夹时间来自搬运/拷贝，不能作为项目开发时间。

## 现有文件与项目描述的对应关系

| 描述要点 | 主要证据 | 判断 |
| --- | --- | --- |
| CNN、Actor、双 Critic、ReplayBuffer、目标网络 | `td3_car/optimal_d_csdn.py` 中 `Actor`（84 行）、`Critic`（179）、`ReplayBuffer`（292）、`Agent`（337）；`TD3-CarRacing.git/car_racing_best.py` 中也有另一套实现 | 有代码证据；基线源流应注明 |
| Clipped Double Q 与 Target Policy Smoothing | `optimal_d_csdn.py` 393–401 行：目标动作噪声裁剪、双目标 Q 取较小值 | 有代码证据 |
| Delayed Policy Update | `optimal_d_csdn.py` 415–418 行设为 `actor_update_freq=1`；`optimal_d.py` 288–289 行设为 2；`car_racing_best.py` 可配置 `POLICY_DELAY` | 机制存在于其他版本，**不适用于当前高分脚本** |
| 驶离道路奖励与驾驶行为 | `optimal_d.py` 444–475 行分阶段增加道路像素奖励并提前结束；`optimal_d_csdn.py` 的 `SkipFrame` 对驶离道路改写奖励；`5283156` 与 `64da20e` 调整惩罚强度 | 有代码与提交证据；“抄近路、保守驾驶、脱轨恢复”目前主要是观察结论，缺少整理好的逐场景量化评估 |
| 探索噪声对比 | `runs/experiment_log.csv` 包含 `run_baseline`、`run_more_brake`、`run_less_explore`、两种 decay 运行；`optimal_d_csdn.py` 844 行后定义衰减搜索 | 有实验记录，但部分运行同时改变刹车率、噪声、容量等参数，不能把差异完全归因于单一变量 |
| 种子与日志 | `optimal_d_csdn.py` 634–645、679、735–773 行；CSV、TensorBoard、checkpoint、视频都存在 | 代码支持种子记录；当前两种 decay 记录均为 seed 42，尚不足以声称完成多随机种子统计 |
| 200 轮与 800 分 | 本地 CSV 第 55 行；远端 `origin/kite_long_obsv:runs/experiment_log.csv` 第 3 行 | 均为训练历史最佳；没有独立测试 >800 的结构化证据 |

`td3_car/runs/experiment_log.csv` 还有一个实际问题：前 52 条数据为 24 列旧结构，后 16 条为 25 列新结构，却只保留了 24 列表头。直接用 `read_csv.py`/pandas 读取容易列错位；发布前应拆分或迁移成统一 schema，并保留原始文件备份。

## 建议的 GitHub 结构

以下是**目标结构**，尚未移动现有文件：

```text
td3-carracing/
├─ README.md                       # 项目定位、2025 时间线、复现命令、证据边界
├─ LICENSE                         # 保留上游 MIT 声明；说明课程改进来源
├─ NOTICE.md                       # 上游链接、团队成员与个人贡献说明
├─ requirements.txt                # 固定可运行的 Python/依赖版本
├─ .gitignore                      # 排除缓存、全部原始日志和批量 checkpoint
├─ src/td3_carracing/
│  ├─ networks.py                  # 从最终选定版本抽取 CNN、Actor、Critic
│  ├─ agent.py                     # TD3 更新：双 Q、平滑、可配置延迟更新
│  ├─ replay_buffer.py
│  ├─ env.py                       # CarRacing-v3、预处理、帧堆叠
│  └─ rewards.py                   # 驶离赛道奖励方案
├─ scripts/
│  ├─ train.py                     # 统一训练入口
│  ├─ evaluate.py                  # 固定种子的独立测试与成绩导出
│  ├─ sweep.py                     # 噪声/奖励参数搜索
│  └─ demo.py                      # 从 demo_run.py 整理
├─ configs/                        # 基线、奖励惩罚、探索噪声的显式配置
├─ experiments/
│  ├─ results.csv                  # 统一 schema 的精选结果
│  ├─ methodology.md               # 训练/评估协议、种子与参数控制
│  └─ failure_modes.md             # 急弯、脱轨恢复、探索阶段的案例与视频索引
├─ assets/                          # 只放精选演示短视频/图表
└─ legacy/                          # 2025 探索脚本索引；不混入默认运行入口
```

迁移优先级：

1. 以 **`td3_car/` 的 2025 年分支**建立主线，先对齐 `origin/kite_long_obsv` 的 5 个提交并保存当前未提交文件；保留提交作者信息。`TD3-CarRacing.git/` 可作为 2025 年 10–11 月探索阶段的来源，不必原样合并两个 `.git` 历史。
2. 选定一套能和 checkpoint 对应的模型结构、奖励逻辑与训练配置。现有 `demo_run.py` 固定加载 `run_000_decay_0.999900/model/agent_opt_500.pt`，不对应第 200 轮高分证据；展示模型需要单独核验。
3. 将 CSV 拆分并规范化，标注 **训练回报** 与 **独立评估回报**。至少用固定评估种子对选定 checkpoint 跑多局，导出每局分数、均值和失败类型；若仍要写“测试 >800”，用这份评估结果支持。
4. 若简历保留完整 TD3 三机制表述，把延迟更新作为可配置参数在最终主线中真正启用并记录对应实验；不要将当前 `actor_update_freq=1` 的成绩归因于延迟更新。
5. 保留上游 MIT 许可证和清晰署名；说明课程团队与个人负责的模块。批量 `.pt`、TensorBoard 原始事件和测试失败日志不应全部进入代码仓库，精选一个可运行 checkpoint 和少量展示素材即可。

## 可用于当前证据的项目表述

> 基于 2023 年开源 TD3 CarRacing 实现，完成 2025 年课程项目中的 CarRacing-v3 迁移、视觉预处理、奖励设计和训练实验管理；实现双 Critic、目标策略平滑、经验回放、TensorBoard/CSV 记录及噪声衰减搜索。实验日志显示某次运行在第 200 个 episode 前取得历史最佳训练回报超过 800。对急弯和驶离赛道等行为进行了案例分析；独立多种子评估与最终开源复现流程正在整理。

这段表述避免把训练历史最佳写成测试分数，也避免将尚未对应高分运行的延迟更新机制归因于成绩。
