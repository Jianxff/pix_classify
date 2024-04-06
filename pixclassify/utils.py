# standard library
from pathlib import Path
from typing import *
# thirdparty
import numpy as np
import cv2
import matplotlib.pyplot as plt
# lightglue
from lightglue import viz2d

def sample_from_video(
    video_path: Union[Path, str],
    output_dir: Union[Path, str],
    sample_fps: float = 30
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f'Could not open video file: {video_path}')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_rate = int(fps / sample_fps)
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i % sample_rate == 0:
            cv2.imwrite(str(output_dir / f'frame{i:06d}.jpg'), frame)
        i += 1
    cap.release()

def draw_matches(
    imgs: List[np.ndarray], # 0 for query and 1 for reference
    feats: List[Dict],
    matches: Dict,
    output: Union[str, Path] = None,
    viz: bool = False
) -> None:
    # data    
    kpts0, kpts1 = feats[0]['keypoints'], feats[1]['keypoints']
    matches01 = matches['matches']
    m_kpts0, m_kpts1 = kpts0[matches01[..., 0]], kpts1[matches01[..., 1]]
    inliers = matches01.shape[0] / kpts1.shape[0]
    
    # plot images and matches
    viz2d.plot_images(imgs)
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'{(inliers * 100):.1f}% inliers, {matches["stop"]} layers', fs=20)
    
    # output
    if viz: plt.show()
    if output: plt.savefig(output)
    plt.close()
    

    