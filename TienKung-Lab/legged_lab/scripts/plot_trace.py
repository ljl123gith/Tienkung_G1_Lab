#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 play.py 导出的 trace CSV 画出：
- X 方向速度：cmd_vx vs root_vx
- Y 方向速度：cmd_vy vs root_vy
- yaw 角速度：cmd_wz vs root_wz
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def moving_average(x, k=5):
    if k <= 1:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="play.py 保存的 trace CSV 路径，例如 logs/.../play_traces/g1_dwaq_model_9400.csv",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="如果指定，则把图片保存到该路径（例如 track_vel.png），否则只弹出窗口预览。",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="对 root 速度做简单滑动平均窗口大小（步数），0/1 表示不平滑。",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    t = df["t"].to_numpy()
    cmd_vx = df["cmd_vx"].to_numpy()
    cmd_vy = df["cmd_vy"].to_numpy()
    cmd_wz = df["cmd_wz"].to_numpy()
    root_vx = df["root_vx"].to_numpy()
    root_vy = df["root_vy"].to_numpy()
    root_wz = df["root_wz"].to_numpy()

    if args.smooth > 1:
        root_vx_s = moving_average(root_vx, args.smooth)
        root_vy_s = moving_average(root_vy, args.smooth)
        root_wz_s = moving_average(root_wz, args.smooth)
    else:
        root_vx_s, root_vy_s, root_wz_s = root_vx, root_vy, root_wz

    plt.figure(figsize=(10, 6))

    # X 速度
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, cmd_vx, label="期望 X 速度 cmd_vx", color="C0", linewidth=2)
    ax1.plot(t, root_vx_s, label="实际 X 速度 root_vx", color="C1", linewidth=1)
    ax1.set_ylabel("X 速度 (m/s)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # Y 速度
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, cmd_vy, label="期望 Y 速度 cmd_vy", color="C0", linewidth=2)
    ax2.plot(t, root_vy_s, label="实际 Y 速度 root_vy", color="C1", linewidth=1)
    ax2.set_ylabel("Y 速度 (m/s)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")

    # yaw 角速度
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, cmd_wz, label="期望 yaw 角速度 cmd_wz", color="C0", linewidth=2)
    ax3.plot(t, root_wz_s, label="实际 yaw 角速度 root_wz", color="C1", linewidth=1)
    ax3.set_ylabel("yaw 角速度 (rad/s)")
    ax3.set_xlabel("时间 t (s)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left")

    plt.tight_layout()

    if args.save:
        out_path = args.save
        # 如果只给了文件名，就放到 CSV 同目录
        if not os.path.isabs(out_path):
            out_path = os.path.join(os.path.dirname(os.path.abspath(args.csv)), out_path)
        plt.savefig(out_path, dpi=300)
        print(f"已保存图片到: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()