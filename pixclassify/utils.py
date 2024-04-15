# standard library
from pathlib import Path
from typing import *
# thirdparty
import numpy as np
import cv2
import matplotlib.pyplot as plt
# lightglue
from lightglue import viz2d

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

""" Sample frames from video
params:
    video_path: Path to video
    output_dir: Output directory
    sample_fps: Sample rate
"""
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



""" Draw matches between two images
params:
    imgs: List of two images
    feats: List of two features
    matches: Dictionary of matches
    output: Output path
"""
def draw_matches(
    imgs: List[np.ndarray], # 0 for query and 1 for reference
    feats: List[Dict],
    matches: Dict,
    output: Union[str, Path] = None,
    ref_label: str = None,
    color: str = "w"
) -> None:
    # data    
    kpts0, kpts1 = feats[0]['keypoints'], feats[1]['keypoints']
    matches01 = matches['matches']
    m_kpts0, m_kpts1 = kpts0[matches01[..., 0]], kpts1[matches01[..., 1]]
    inliers = matches01.shape[0] / kpts1.shape[0]
    
    # plot images and matches
    viz2d.plot_images(imgs)
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'{(inliers * 100):.1f}% inliers, {matches["stop"]} layers', fs=10)
    if ref_label: viz2d.add_text(1, ref_label, fs=10, color=color)
    
    # output
    if output: plt.savefig(output)


""" Convert matplotlib plot to numpy array """
def plt_to_numpy_rgb(resize: Tuple[int, int] = None) -> np.ndarray:
    fig = plt.gcf()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    if resize is not None:
        (x, y) = resize
        bg = np.ones((y, x, 3), dtype=np.uint8) * 255
        f = np.min([x / data.shape[1], y / data.shape[0]])
        data = cv2.resize(data, None, fx=f, fy=f)
        bg[:data.shape[0], :data.shape[1]] = data
        data = bg
    return data


""" Get system fonts """
def get_system_fonts():
    from matplotlib.font_manager import FontManager

    mpl_fonts = set(f.name for f in FontManager().ttflist)

    print('all font list get from matplotlib.font_manager:')
    for f in sorted(mpl_fonts):
        print('\t' + f)
    

    