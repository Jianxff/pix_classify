# standard library
from pathlib import Path
import os
from typing import *
# third party
import matplotlib.pyplot as plt
import argparse
# Pix Classify
import pixclassify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixClassify")
    parser.add_argument("query", type=str)
    parser.add_argument("--database", type=str, default="./data/features.h5")
    parser.add_argument("--savedir", type=str, default=None, help="Save directory for each matching label(image only)")
    parser.add_argument("--saveall", action="store_true", help="Save all results for each reference image(image only)")
    parser.add_argument("--viz", action="store_true", help="Visualize results")
    parser.add_argument("--top-k", type=int, default=2, help="Top k results")
    parser.add_argument("--gt", type=str, default=None, help="Ground truth label(video only)")
    parser.add_argument("--make-video", type=str, default=None, help="Make video from matching frames(video only)")
    args = parser.parse_args()

    db_file = Path(args.database)
    classifier = pixclassify.Classifier(database=db_file)

    savedir = Path(args.savedir) if args.savedir else None
    if savedir:
        os.makedirs(savedir, exist_ok=True)

    query = Path(args.query)

    # check if video or image
    if query.suffix in [".mp4", ".avi", ".mov"]:
        res = classifier.identify_video(
            query=query,
            top_k=args.top_k,
            viz=args.viz,
            gt_label=args.gt,
            make_video=args.make_video,
        )

    else:
        res = classifier.identify(
            query=query, 
            top_k=args.top_k,
            draw_match=args.viz,
            savedir=savedir,
            saveall=args.saveall
        )
        if args.viz:
            plt.show()

    print(res)


