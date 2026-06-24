#!/usr/bin/env python3
"""
一键跑通整个流程：遍历 ./input/ 下的所有视频，
依次执行 OpenPose → opjson2npy → preprocess → inference → 可视化，
所有输出放到 ./output/<视频名>/ 下。

通过 KEEP 数组选择要保留的文件。
"""

import subprocess, sys, os, glob, cv2, numpy as np
from pathlib import Path

# ============================================================
#  配置：选择要保留的输出 (True=保留, False=完成后删除)
# ============================================================
KEEP = {
    'openpose_json': False,   # OpenPose 每帧 JSON
    'raw_npy':        False,  # 原始 25 个关键点 .npy
    'processed_npy':  False,  # 预处理后下半身 .npy
    'pred_npy':       True,   # 预测标签 .npy
    'results_txt':    True,   # 时间戳 + 数组文本
    'timeline':       False,  # 时间轴可视化 .mp4
    'overlay':        True,   # 圆圈叠加可视化 .mp4
}
# ============================================================

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / 'input'
OUTPUT_DIR = ROOT / 'output'
OPENPOSE_BIN = '/home/lihaozhe/dev/openpose/build/examples/openpose/openpose.bin'
OPENPOSE_MODEL = '/home/lihaozhe/dev/openpose/models'
CHECKPOINT = ROOT / 'checkpoints' / 'best_model.pth'
VENV_PYTHON = ROOT / '.venv' / 'bin' / 'python'

LABEL_NAMES = ['left_toe', 'right_toe', 'left_heel', 'right_heel']
JOINT_INDICES = {'left_toe': 19, 'right_toe': 22, 'left_heel': 21, 'right_heel': 24}


def run_openpose(video_path, json_dir):
    print(f'  [1/5] OpenPose extracting keypoints...')
    subprocess.run([
        OPENPOSE_BIN,
        '--video', str(video_path),
        '--write_json', str(json_dir),
        '--display', '0',
        '--render_pose', '0',
        '--model_folder', OPENPOSE_MODEL,
    ], check=True)


def run_opjson2npy(json_dir, out_path):
    print(f'  [2/5] Converting JSON -> raw .npy...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.opjson2npy',
        '--input_dir', str(json_dir),
        '--output_path', str(out_path),
    ], check=True, cwd=str(ROOT))


def run_preprocess(json_dir, out_path):
    print(f'  [3/5] Preprocessing -> processed .npy...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.preprocess',
        '--input_dir', str(json_dir),
        '--output_path', str(out_path),
    ], check=True, cwd=str(ROOT))


def run_inference(input_npy, out_path):
    print(f'  [4/5] Running inference...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.inference',
        '--input_path', str(input_npy),
        '--output_path', str(out_path),
        '--model_path', str(CHECKPOINT),
    ], check=True, cwd=str(ROOT))


def write_text_result(video_path, labels_npy, out_path):
    print(f'  [5/5] Generating text results...')
    labels = np.load(labels_npy).astype(int)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    with open(out_path, 'w') as f:
        f.write(f'视频: {video_path}\n')
        f.write(f'FPS: {fps:.2f}, 帧数: {len(labels)}\n')
        f.write(f'标签顺序: {LABEL_NAMES}\n')
        f.write(f'  {"帧号":>5}  {"时间":>8}  {"L_toe":>5} {"R_toe":>5} {"L_heel":>5} {"R_heel":>5}\n')
        f.write('  ' + '-' * 42 + '\n')
        for i in range(len(labels)):
            t = i / fps
            a = labels[i]
            f.write(f'  {i:>5}  {t:>7.3f}s  {a[0]:>5} {a[1]:>5} {a[2]:>5} {a[3]:>5}\n')
    print(f'  -> {out_path}')


def gen_overlay(video_path, pose_npy, labels_npy, out_path):
    print(f'  [5/5] Generating overlay video...')
    poses = np.load(pose_npy).squeeze()
    labels = np.load(labels_npy)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), poses.shape[0])

    PANEL_W, PANEL_X0, PANEL_Y0 = 220, w - 235, 15
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        joints, lbl = poses[fi], labels[fi]
        for jname, jidx in JOINT_INDICES.items():
            x, y, c = joints[jidx]
            if c <= 0 or not (np.isfinite(x) and np.isfinite(y)):
                continue
            li = LABEL_NAMES.index(jname)
            color = (0, 220, 0) if lbl[li] == 1 else (0, 0, 200)
            cv2.circle(frame, (int(x), int(y)), 12, color, -1)
            cv2.circle(frame, (int(x), int(y)), 12, (255, 255, 255), 2)

        ph = 28 * 4 + 50
        cv2.rectangle(frame, (PANEL_X0, PANEL_Y0), (PANEL_X0 + PANEL_W, PANEL_Y0 + ph), (25, 25, 25), -1)
        cv2.rectangle(frame, (PANEL_X0, PANEL_Y0), (PANEL_X0 + PANEL_W, PANEL_Y0 + ph), (80, 80, 80), 1)
        cv2.putText(frame, f'Frame {fi}  {fi/fps:.2f}s',
                    (PANEL_X0 + 10, PANEL_Y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        for i, name in enumerate(LABEL_NAMES):
            y = PANEL_Y0 + 50 + i * 28
            val = int(lbl[i])
            color = (0, 220, 0) if val else (0, 0, 200)
            cv2.circle(frame, (PANEL_X0 + 20, y + 5), 6, color, -1)
            cv2.putText(frame, f'{name}: {val}', (PANEL_X0 + 35, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f'  -> {out_path}')


def gen_timeline(labels_npy, fps, out_path):
    print(f'  [5/5] Generating timeline video...')
    labels = np.load(labels_npy).astype(int)
    total = len(labels)
    W, H = 1280, 720
    ML, MR, MT, MB = 160, 60, 60, 80
    CW, CH = W - ML - MR, H - MT - MB
    RH = CH // 4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    for fi in range(total):
        img = np.full((H, W, 3), (30, 30, 30), dtype=np.uint8)
        t = fi / fps
        cv2.putText(img, f'Contact Timeline  ({t:.2f}s / {total/fps:.2f}s)',
                    (ML, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
        # Time axis
        ay = H - MB + 40
        cv2.line(img, (ML, ay), (ML + CW, ay), (60, 60, 60), 2)
        nt = max(2, min(10, total // int(max(fps, 1))))
        for i in range(nt + 1):
            tf = int(i * total / nt)
            tx = ML + int(tf / total * CW)
            cv2.line(img, (tx, ay - 4), (tx, ay + 4), (220, 220, 220), 1)
            cv2.putText(img, f'{tf/fps:.1f}s', (tx - 20, ay + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        # Progress line
        px = ML + int(fi / total * CW) if total else ML
        cv2.line(img, (px, MT - 10), (px, ay), (0, 150, 255), 2)
        # Rows
        colors = [(0, 200, 0), (0, 200, 200), (200, 100, 0), (0, 100, 200)]
        vs = max(0, fi - int(fps * 2))
        ve = min(total, fi + int(fps * 2))
        ppf = CW / max(total, 1)
        for row in range(4):
            y0, y1 = MT + row * RH, MT + (row + 1) * RH - 4
            cv2.putText(img, LABEL_NAMES[row], (15, (y0 + y1) // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[row], 2)
            cv2.rectangle(img, (ML, y0), (ML + CW, y1), (45, 45, 45), -1)
            for f in range(vs, min(ve, total)):
                x = ML + int(f * ppf)
                bw = max(2, int(ppf * 1.5))
                c = (0, 200, 0) if labels[f, row] == 1 else (80, 80, 80)
                cv2.rectangle(img, (x, y0 + 2), (x + bw, y1 - 2), c, -1)
            if row > 0:
                cv2.line(img, (ML, y0), (ML + CW, y0), (60, 60, 60), 1)
        if fi < total:
            vals = labels[fi].astype(int)
            info = '  |  '.join(f'{n}: {v}' for n, v in zip(LABEL_NAMES, vals))
            cv2.putText(img, f'[{info}]', (ML, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        writer.write(img)
    writer.release()
    print(f'  -> {out_path}')


def cleanup(out_dir, keep):
    for name, should_keep in keep.items():
        if should_keep:
            continue
        target = out_dir / {
            'openpose_json': 'openpose_json',
            'raw_npy':       f'{out_dir.stem}_raw.npy',
            'processed_npy': f'{out_dir.stem}_processed.npy',
            'pred_npy':      f'{out_dir.stem}_pred_labels.npy',
            'results_txt':   f'{out_dir.stem}_results.txt',
            'timeline':      f'{out_dir.stem}_timeline.mp4',
            'overlay':       f'{out_dir.stem}_overlay.mp4',
        }[name]
        if target.is_dir():
            import shutil
            shutil.rmtree(target)
            print(f'  删除目录: {target.name}/')
        elif target.is_file():
            target.unlink()
            print(f'  删除文件: {target.name}')


def process_video(video_path):
    name = video_path.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir     = out_dir / 'openpose_json'
    raw_npy      = out_dir / f'{name}_raw.npy'
    processed_npy = out_dir / f'{name}_processed.npy'
    pred_npy     = out_dir / f'{name}_pred_labels.npy'
    result_txt   = out_dir / f'{name}_results.txt'
    timeline_mp4 = out_dir / f'{name}_timeline.mp4'
    overlay_mp4  = out_dir / f'{name}_overlay.mp4'

    print(f'\n====== Processing: {name} ======')

    run_openpose(video_path, json_dir)
    run_opjson2npy(json_dir, raw_npy)
    run_preprocess(json_dir, processed_npy)
    run_inference(processed_npy, pred_npy)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    n_vis = sum([KEEP.get('results_txt', False), KEEP.get('overlay', False), KEEP.get('timeline', False)])
    vi = 1
    if KEEP.get('results_txt'):
        write_text_result(video_path, pred_npy, result_txt)
        vi += 1
    if KEEP.get('overlay'):
        gen_overlay(video_path, raw_npy, pred_npy, overlay_mp4)
        vi += 1
    if KEEP.get('timeline'):
        gen_timeline(pred_npy, fps, timeline_mp4)

    cleanup(out_dir, KEEP)
    print(f'====== Done: {name} ======')


def main():
    if not INPUT_DIR.is_dir():
        print(f'错误: 输入目录不存在: {INPUT_DIR}')
        print(f'请先创建 {INPUT_DIR} 并将视频文件放入其中')
        sys.exit(1)

    videos = sorted(glob.glob(str(INPUT_DIR / '*.mp4')))
    if not videos:
        print(f'在 {INPUT_DIR}/ 下未找到 .mp4 文件')
        sys.exit(1)

    print(f'找到 {len(videos)} 个视频: {[Path(v).stem for v in videos]}')
    print(f'保留配置:')
    for k, v in KEEP.items():
        print(f'  {k}: {"保留" if v else "删除"}')

    for v in videos:
        process_video(Path(v))

    print(f'\n全部完成！结果保存在 {OUTPUT_DIR}/ 下')


if __name__ == '__main__':
    main()
