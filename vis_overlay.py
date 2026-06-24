#!/usr/bin/env python3
"""
在视频上叠加脚步接触状态可视化（圆圈 + 信息面板）。

用法:
  .venv/bin/python vis_overlay.py \
      --video input/2.mp4 \
      --pose output/2/2_raw.npy \
      --labels output/2/2_pred_labels.npy \
      --output output/2/2_overlay.mp4
"""

import argparse, numpy as np, cv2

JOINT_INDICES = {'left_toe': 19, 'right_toe': 22, 'left_heel': 21, 'right_heel': 24}
LABEL_ORDER = ['left_toe', 'right_toe', 'left_heel', 'right_heel']
LABEL_SHORT = ['L_toe', 'R_toe', 'L_heel', 'R_heel']
ACTIVE_COLOR = (0, 220, 0)
INACTIVE_COLOR = (0, 0, 200)
MARKER_RADIUS = 12
MARKER_THICKNESS = -1
BG_COLOR = (25, 25, 25)
TEXT_COLOR = (240, 240, 240)


def main():
    parser = argparse.ArgumentParser(description='Overlay contact labels on video')
    parser.add_argument('--video', required=True)
    parser.add_argument('--pose', required=True, help='Raw keypoints .npy (T,25,3)')
    parser.add_argument('--labels', required=True, help='Prediction .npy (T,4)')
    parser.add_argument('--output', default='overlay.mp4')
    args = parser.parse_args()

    poses = np.load(args.pose).squeeze()
    labels = np.load(args.labels)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    PANEL_W = 220
    panel_x0 = w - PANEL_W - 15
    panel_y0 = 15

    frame_idx = 0
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), poses.shape[0])
    print(f'渲染 {total} 帧到 {args.output}...')

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= poses.shape[0]:
            break

        joints = poses[frame_idx]
        lbl = labels[frame_idx]

        # Draw joint markers
        for jname, jidx in JOINT_INDICES.items():
            x, y, c = joints[jidx]
            if c <= 0 or not (np.isfinite(x) and np.isfinite(y)):
                continue
            li = LABEL_ORDER.index(jname)
            color = ACTIVE_COLOR if lbl[li] == 1 else INACTIVE_COLOR
            cv2.circle(frame, (int(x), int(y)), MARKER_RADIUS, color, MARKER_THICKNESS)
            cv2.circle(frame, (int(x), int(y)), MARKER_RADIUS, (255, 255, 255), 2)

        # Info panel background
        panel_h = 28 * 4 + 50
        cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x0 + PANEL_W, panel_y0 + panel_h), BG_COLOR, -1)
        cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x0 + PANEL_W, panel_y0 + panel_h), (80, 80, 80), 1)

        t = frame_idx / fps
        cv2.putText(frame, f'Frame {frame_idx}  {t:.2f}s',
                    (panel_x0 + 10, panel_y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

        for i, (name, short) in enumerate(zip(LABEL_ORDER, LABEL_SHORT)):
            y = panel_y0 + 50 + i * 28
            val = int(lbl[i])
            color = ACTIVE_COLOR if val == 1 else INACTIVE_COLOR
            cv2.circle(frame, (panel_x0 + 20, y + 5), 6, color, -1)
            cv2.putText(frame, f'{short}: {val}', (panel_x0 + 35, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'  {frame_idx}/{total}')

    cap.release()
    writer.release()
    print(f'完成: {args.output}')


if __name__ == '__main__':
    main()
