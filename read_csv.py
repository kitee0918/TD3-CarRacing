import os
import pandas as pd
import matplotlib.pyplot as plt

# CSV 路径：与你训练时 append_experiment_log 写的一致
# runs/experiment_log.csv
CSV_PATH = os.path.join("runs", "experiment_log.csv")

# 图像输出目录
FIG_DIR = os.path.join("runs", "figures")


def load_log(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """读取实验日志 CSV。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到日志文件: {csv_path}")
    df = pd.read_csv(csv_path)

    # 简单检查字段是否存在
    required_cols = ["run_idx", "run_name", "episode", "is_final",
                     "episode_reward", "best_reward_so_far", "sigma_decay_factor"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV 中缺少必须字段: {col}")
    return df


def summarize_final_by_decay(df: pd.DataFrame) -> pd.DataFrame:
    """
    汇总：不同 sigma_decay_factor 下的最终 best_reward 表现。
    我们用 is_final == 1 的行作为每个 run 的最终结果。
    """
    final_df = df[df["is_final"] == 1].copy()

    # 防止 old log 里没有这一列
    if "sigma_decay_factor" not in final_df.columns:
        raise ValueError("CSV 中没有 sigma_decay_factor 列，请确认日志记录代码是否已更新。")

    # 按 (sigma_decay_factor) 分组，统计：
    # - mean_best_reward: 各组最终 best_reward 平均值
    # - std_best_reward : 各组最终 best_reward 标准差
    # - count_runs      : 每个衰减因子下跑了多少个 run
    group = final_df.groupby("sigma_decay_factor")["best_reward_so_far"]
    summary = pd.DataFrame({
        "mean_best_reward": group.mean(),
        "std_best_reward": group.std(),
        "count_runs": group.count()
    }).reset_index().sort_values("sigma_decay_factor")
    return summary


def plot_final_bar(summary: pd.DataFrame, save_dir: str = FIG_DIR):
    """画不同 sigma_decay_factor 的最终 best_reward 柱状图。"""
    os.makedirs(save_dir, exist_ok=True)

    x = summary["sigma_decay_factor"].values
    y = summary["mean_best_reward"].values

    plt.figure()
    plt.bar(range(len(x)), y)
    plt.xticks(range(len(x)), [f"{v:.6f}" for v in x], rotation=45, ha="right")
    plt.xlabel("sigma_decay_factor")
    plt.ylabel("mean final best_reward")
    plt.title("Final best_reward by sigma_decay_factor")

    out_path = os.path.join(save_dir, "final_best_reward_by_sigma_decay_factor.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存图像: {out_path}")


def plot_learning_curves(df: pd.DataFrame, save_dir: str = FIG_DIR):
    """
    画训练过程曲线：不同 sigma_decay_factor 下，
    best_reward_so_far 随 episode 的变化（对相同衰减因子的多个 run 取平均）。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 只看中间模型记录：is_final == 0
    mid_df = df[df["is_final"] == 0].copy()

    if "sigma_decay_factor" not in mid_df.columns:
        raise ValueError("CSV 中没有 sigma_decay_factor 列，请确认日志记录代码是否已更新。")

    # 为了更平滑，对 (sigma_decay_factor, episode) 分组，取 best_reward_so_far 的均值
    grouped = (
        mid_df
        .groupby(["sigma_decay_factor", "episode"])["best_reward_so_far"]
        .mean()
        .reset_index()
        .sort_values(["sigma_decay_factor", "episode"])
    )

    plt.figure()
    for decay, sub in grouped.groupby("sigma_decay_factor"):
        sub = sub.sort_values("episode")
        plt.plot(sub["episode"].values, sub["best_reward_so_far"].values,
                 label=f"{decay:.6f}")

    plt.xlabel("episode")
    plt.ylabel("mean best_reward_so_far")
    plt.title("Learning curves by sigma_decay_factor")
    plt.legend(title="sigma_decay_factor")
    plt.tight_layout()

    out_path = os.path.join(save_dir, "learning_curves_by_sigma_decay_factor.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存图像: {out_path}")


def main():
    df = load_log(CSV_PATH)

    print("日志前几行：")
    print(df.head())

    # 1. 汇总最终表现 & 柱状图
    summary = summarize_final_by_decay(df)
    print("\n不同 sigma_decay_factor 的最终表现：")
    print(summary)

    plot_final_bar(summary, save_dir=FIG_DIR)

    # 2. 训练过程曲线
    plot_learning_curves(df, save_dir=FIG_DIR)


if __name__ == "__main__":
    main()
