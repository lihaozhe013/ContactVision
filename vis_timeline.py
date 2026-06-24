#!/usr/bin/env python3
"""
生成接触标签的时间轴可视化（不嵌入原视频）。

输出一个独立的 MP4 视频，显示 4 个接触标签
(left_toe / right_toe / left_heel / right_heel)
随时间变化的 0/1 状态。

用法:
  .venv/bin/python vis_timeline.py --labels <pred_npy> [--fps 30] [--output output_timeline.mp4]
  .venv/bin/python vis_timeline.py --labels pred_4.npy --video data/sample/video/4.mp4 --output timeline_4.mp4
"""

import argparse, numpy as np, cv2
from pathlib import Path

LABEL_NAMES = ['left_toe', 'right_toe', 'left_heel', 'right_heel']
LABEL_COLORS = [(0, 200, 0), (0, 200, 200), (200, 100, 0), (0, 100, 200)]
BG_COLOR = (30, 30, 30)
ACTIVE_COLOR = (0, 200, 0)
INACTIVE_COLOR = (80, 80, 80)
TEXT_COLOR = (220, 220, 220)
GRID_COLOR = (60, 60, 60)

W, H = 1280, 720
MARGIN_L = 160
MARGIN_R = 60
MARGIN_T = 60
MARGIN_B = 80
CHART_W = W - MARGIN_L - MARGIN_R
CHART_H = H - MARGIN_T - MARGIN_B
ROW_H = CHART_H // 4


def draw_timeline_frame(img, labels, frame_idx, total_frames, fps):
    img[:] = BG_COLOR
    t = frame_idx / fps

    # Title
    cv2.putText(img, f'Contact Timeline  ({t:.2f}s / {total_frames/fps:.2f}s)',
                (MARGIN_L, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT_COLOR, 2)

    # Time axis
    axis_y = H - MARGIN_B + 40
    cv2.line(img, (MARGIN_L, axis_y), (MARGIN_L + CHART_W, axis_y), GRID_COLOR, 2)

    # Time labels every N seconds
    n_ticks = max(2, min(10, total_frames // int(fps)))
    for i in range(n_ticks + 1):
        tick_frame = int(i * total_frames / n_ticks)
        tick_x = MARGIN_L + int(tick_frame / total_frames * CHART_W)
        cv2.line(img, (tick_x, axis_y - 4), (tick_x, axis_y + 4), TEXT_COLOR, 1)
        label = f'{tick_frame/fps:.1f}s'
        cv2.putText(img, label, (tick_x - 20, axis_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    # Progress indicator (current time)
    prog_x = MARGIN_L + int(frame_idx / total_frames * CHART_W) if total_frames > 0 else MARGIN_L
    cv2.line(img, (prog_x, MARGIN_T - 10), (prog_x, axis_y), (0, 150, 255), 2)

    # Draw each label row
    for row in range(4):
        y0 = MARGIN_T + row * ROW_H
        y1 = MARGIN_T + (row + 1) * ROW_H - 4
        row_mid = (y0 + y1) // 2

        # Label name
        cv2.putText(img, LABEL_NAMES[row], (15, row_mid + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LABEL_COLORS[row], 2)

        # Background row
        cv2.rectangle(img, (MARGIN_L, y0), (MARGIN_L + CHART_W, y1), (45, 45, 45), -1)

        # Draw contact bars for visible window
        visible_start = max(0, frame_idx - int(fps * 2))
        visible_end = min(total_frames, frame_idx + int(fps * 2))
        px_per_frame = CHART_W / max(total_frames, 1)

        for f in range(visible_start, visible_end):
            if f >= len(labels):
                break
            x = MARGIN_L + int(f * px_per_frame)
            bar_w = max(2, int(px_per_frame * 1.5))
            color = ACTIVE_COLOR if labels[f, row] == 1 else INACTIVE_COLOR
            cv2.rectangle(img, (x, y0 + 2), (x + bar_w, y1 - 2), color, -1)

        # Thin separator line
        if row > 0:
            cv2.line(img, (MARGIN_L, y0), (MARGIN_L + CHART_W, y0), GRID_COLOR, 1)

    # Current value indicator
    if frame_idx < len(labels):
        vals = labels[frame_idx].astype(int)
        info = '  |  '.join(f'{n}: {v}' for n, v in zip(LABEL_NAMES, vals))
        cv2.putText(img, f'[{info}]', (MARGIN_L, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)


def main():
    parser = argparse.ArgumentParser(description='Generate contact timeline visualization')
    parser.add_argument('--labels', required=True, help='Prediction .npy file (shape: T,4)')
    parser.add_argument('--video', help='Source video (used for FPS, optional)')
    parser.add_argument('--fps', type=float, default=30, help='FPS (used if no video given)')
    parser.add_argument('--output', default='timeline.mp4', help='Output video path')
    args = parser.parse_args()

    labels = np.load(args.labels).astype(int)
    total = len(labels)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    else:
        fps = args.fps

    fps = max(fps, 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))

    print(f'生成时间轴可视化: {total}帧, {fps}fps, 输出到 {args.output}')
    for i in range(total):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        draw_timeline_frame(img, labels, i, total, fps)
        writer.write(img)
        if (i + 1) % 30 == 0:
            print(f'  渲染中... {i+1}/{total}')

    writer.release()
    print(f'完成: {args.output}')


if __name__ == '__main__':
    main()
