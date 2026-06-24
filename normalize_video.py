#!/usr/bin/env python3
"""
视频标准化工具：检查手机视频的旋转元数据，将竖屏视频转正为横屏。
处理后的视频放到 input/ 目录下供 run_pipeline.py 使用。

用法:
  .venv/bin/python normalize_video.py video.mp4
  .venv/bin/python normalize_video.py video1.mp4 video2.mp4
  .venv/bin/python normalize_video.py ./raw_videos/*.mp4
"""

import sys, cv2, os
from pathlib import Path

INPUT_DIR = Path(__file__).resolve().parent / 'input'

def get_rotation(video_path):
    cap = cv2.VideoCapture(str(video_path))
    rot = cap.get(cv2.CAP_PROP_ORIENTATION_META)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return rot, w, h, fps, total

def rotation_to_code(rot):
    if rot == 90:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    elif rot == 270 or rot == -90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rot == 180:
        return cv2.ROTATE_180
    return None

def normalize_video(video_path):
    video_path = Path(video_path)
    if not video_path.is_file():
        print(f'文件不存在: {video_path}')
        return False

    rot, w, h, fps, total = get_rotation(video_path)
    name = video_path.stem

    print(f'{video_path.name}: {w}x{h}, {fps:.2f}fps, {total}帧, 旋转元数据={rot}°')

    code = rotation_to_code(rot)
    if code is None:
        # No rotation needed, just copy/link to input/
        dest = INPUT_DIR / video_path.name
        os.system(f'cp "{video_path}" "{dest}"')
        print(f'  无需旋转，已复制到 input/')
        return True

    # Determine output dimensions after rotation
    out_w, out_h = h, w  # 90° flips w/h
    print(f'  检测到 {rot}° 旋转，转正为 {out_w}x{out_h}...')

    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dest = INPUT_DIR / f'{name}.mp4'
    writer = cv2.VideoWriter(str(dest), fourcc, fps, (out_w, out_h))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rotated = cv2.rotate(frame, code)
        writer.write(rotated)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f'  处理中... {frame_idx}/{total} 帧')

    cap.release()
    writer.release()
    print(f'  完成! 已保存到 {dest} ({frame_idx} 帧)')
    return True


def main():
    if not INPUT_DIR.is_dir():
        INPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = sys.argv[1:]
    # Expand glob patterns if any
    expanded = []
    for p in paths:
        if '*' in p or '?' in p:
            expanded.extend(str(x) for x in Path().resolve().glob(p))
        else:
            expanded.append(p)
    paths = expanded

    if not paths:
        print('未找到匹配的文件')
        sys.exit(1)

    for p in paths:
        normalize_video(p)
        print()

    print(f'全部处理完成! 视频已放到 {INPUT_DIR}/，现在可以运行 run_pipeline.py')


if __name__ == '__main__':
    main()
