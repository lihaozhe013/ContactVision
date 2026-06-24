#!/usr/bin/env python3
"""
一键跑通整个流程：遍历 ./input/ 下的所有视频，
依次执行 OpenPose → opjson2npy → preprocess → inference → 文本结果，
所有输出放到 ./output/<视频名>/ 下。
"""

import subprocess, sys, os, glob, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / 'input'
OUTPUT_DIR = ROOT / 'output'
OPENPOSE_BIN = '/home/lihaozhe/dev/openpose/build/examples/openpose/openpose.bin'
OPENPOSE_MODEL = '/home/lihaozhe/dev/openpose/models'
CHECKPOINT = ROOT / 'checkpoints' / 'best_model.pth'
VENV_PYTHON = ROOT / '.venv' / 'bin' / 'python'

LABEL_NAMES = ['left_toe', 'right_toe', 'left_heel', 'right_heel']


def run_openpose(video_path, json_dir):
    print(f'  [1/4] OpenPose extracting keypoints...')
    subprocess.run([
        OPENPOSE_BIN,
        '--video', str(video_path),
        '--write_json', str(json_dir),
        '--display', '0',
        '--render_pose', '0',
        '--model_folder', OPENPOSE_MODEL,
    ], check=True)


def run_opjson2npy(json_dir, out_path):
    print(f'  [2/4] Converting JSON -> raw .npy...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.opjson2npy',
        '--input_dir', str(json_dir),
        '--output_path', str(out_path),
    ], check=True, cwd=str(ROOT))


def run_preprocess(json_dir, out_path):
    print(f'  [2b/4] Preprocessing JSON -> processed .npy...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.preprocess',
        '--input_dir', str(json_dir),
        '--output_path', str(out_path),
    ], check=True, cwd=str(ROOT))


def run_inference(input_npy, out_path):
    print(f'  [3/4] Running inference...')
    subprocess.run([
        str(VENV_PYTHON), '-m', 'contact_vision.inference',
        '--input_path', str(input_npy),
        '--output_path', str(out_path),
        '--model_path', str(CHECKPOINT),
    ], check=True, cwd=str(ROOT))


def write_text_result(video_path, labels_npy, out_path):
    print(f'  [4/4] Generating text result...')
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


def process_video(video_path):
    name = video_path.stem
    out_dir = OUTPUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    json_dir = out_dir / 'openpose_json'
    raw_npy = out_dir / f'{name}_raw.npy'
    processed_npy = out_dir / f'{name}_processed.npy'
    pred_npy = out_dir / f'{name}_pred_labels.npy'
    result_txt = out_dir / f'{name}_results.txt'

    print(f'\n====== Processing: {name} ======')

    run_openpose(video_path, json_dir)
    run_opjson2npy(json_dir, raw_npy)
    run_preprocess(json_dir, processed_npy)
    run_inference(processed_npy, pred_npy)
    write_text_result(video_path, pred_npy, result_txt)

    print(f'====== Done: {name} ======')
    return result_txt


def main():
    if not INPUT_DIR.is_dir():
        print(f'错误: 输入目录不存在: {INPUT_DIR}')
        print(f'请创建 {INPUT_DIR} 并将视频文件放入其中')
        sys.exit(1)

    videos = sorted(glob.glob(str(INPUT_DIR / '*.mp4')))
    if not videos:
        print(f'在 {INPUT_DIR}/ 下未找到 .mp4 文件')
        sys.exit(1)

    print(f'找到 {len(videos)} 个视频: {[Path(v).stem for v in videos]}')

    for v in videos:
        process_video(Path(v))

    print(f'\n全部完成！结果保存在 {OUTPUT_DIR}/ 下')
    print('每个视频的目录结构:')
    print('  output/<视频名>/')
    print('    ├── openpose_json/    (OpenPose 原始 JSON)')
    print('    ├── <名>_raw.npy       (原始关键点)')
    print('    ├── <名>_processed.npy (预处理后关键点)')
    print('    ├── <名>_pred_labels.npy (预测标签)')
    print('    └── <名>_results.txt   (可读的结果文本)')


if __name__ == '__main__':
    main()
