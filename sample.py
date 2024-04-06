# standard library
from pathlib import Path
import argparse
import os

from pixclassify import sample_from_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixClassify")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--sample_fps", type=float, default=30)
    args = parser.parse_args()

    video_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    fps = args.sample_fps

    for video in video_dir.glob('*.[mp4|avi|mov|MOV|MP4|AVI]'):
        if video.is_file():
            output = out_dir / video.stem
            os.makedirs(output, exist_ok=True)
            print(f'sample "{video.stem}" from video:', video, '->', output)
            sample_from_video(video, output, fps)