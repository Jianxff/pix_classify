# standard library
from pathlib import Path
import os
from typing import *
# third party
import argparse
import matplotlib.pyplot as plt
# Pix Classify
import pixclassify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixClassify")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--database", type=str, default="./data/features.h5")
    parser.add_argument("--savefig", action="store_true", default=False, help="Make video and figure from matching frames")
    args = parser.parse_args()

    db_file = Path(args.database)
    classifier = pixclassify.Classifier(database=db_file, read_only=True)

    EVAL_TOTAL_FRAMES = 0
    EVAL_TOTAL_WRONG_FRAMES = 0
    EVAL_TOTAL_TIME = 0
    EVAL_TOTAL_CONF = 0

    ###### Folder Structure #####
    # - $root                   #
    #   - label1                #
    #       - video1.mp4        #
    #       - video2.mp4        #
    #       - image1.png        #
    #   - label2                #
    #       - video3.mp4        #
    #       - image2.jpg        #
    #############################
    
    data_dir = Path(args.data)
    os.makedirs(args.out_dir, exist_ok=True)
    out_dir = Path(args.out_dir)

    # summary files
    file_detail_out = open(str(out_dir / "eval_detail.csv"), "w")
    file_detail_out.write("label,file,count,accuracy,inliers,time,top_others\n")
    file_total_out = open(str(out_dir / "eval_summary.csv"), "w")
    file_total_out.write("label,count,accuracy\n")
    
    sub_dirs = [x for x in data_dir.iterdir() if x.is_dir()]
    # labeled_data: [(label1, [videos1], [images1]), ...]
    labeled_data = [] 
    # grab files
    for sub in sub_dirs:
        label = sub.name
        video_files = sorted(list(sub.glob('*.[m|a][p|o|v][4|v|i]')))
        image_files = sorted(list(sub.glob('*.[j|p][p|n][g|e]')))
        # add to labeled_data
        labeled_data.append((label, video_files, image_files))
    

    ### evaluate
    for l_index, (label, videos, images) in enumerate(labeled_data):
        print("#" * 50)
        print(f"Evaluating {l_index+1}/{len(labeled_data)}: {label}")

        EVAL_LABEL_FRAMES = 0
        EVAL_LABEL_WRONG_FRAMES = 0

        ### videos
        for v_index, video in enumerate(videos):
            out_video_path = None
            if args.savefig:
                out_video_path = str(out_dir / label / f"{video.stem}.mp4")
                os.makedirs(out_dir / label, exist_ok=True)
                
            # res: [(label, proportion aka. acc, conf), ...]
            res, EVAL_VIDEO_FRAMES, t = classifier.identify_video(query=video, gt_label=label, top_k=3, make_video=out_video_path)
            # to list
            res = [list(x) for x in res]
            # proportion and average confidence            
            for r in res:
                r[1], r[2] = 100 * r[1], 100 * r[2]

            acc, conf = res[0][1], res[0][2]

            ##########################################################################################
            print(f"  - video {v_index+1}/{len(videos)}: acc: {acc:.1f}, conf: {conf:.2f}, time: {t:.2f} s, top others: [{res[1][0]}/{res[1][1]:.2f}, {res[2][0]}/{res[2][1]:.2f}]")
            # csv: label, file, count, accuracy, inliers, time, top_others
            file_detail_out.write(f"{label},{video},{EVAL_VIDEO_FRAMES},{acc},{conf},{t},[{res[1][0]}:{res[1][1]:.3f}][{res[2][0]}:{res[2][1]:.3f}]\n")
            ##########################################################################################

            EVAL_LABEL_FRAMES += EVAL_VIDEO_FRAMES
            EVAL_LABEL_WRONG_FRAMES += (EVAL_VIDEO_FRAMES * (1 - acc/100))
            EVAL_TOTAL_TIME += t
            EVAL_TOTAL_CONF += (conf / 100 * EVAL_VIDEO_FRAMES)

        ### images
        for i_index, image in enumerate(images):
            res, t = classifier.identify(query=image, top_k=2, draw_match=args.savefig)
            # assert label result
            if res[0][0] != label: EVAL_LABEL_WRONG_FRAMES += 1
            other = res[1]
            conf = 100 * res[0][1]

            if args.savefig:
                os.makedirs(out_dir / label, exist_ok=True)
                out_figure_path = str(out_dir / label / f"{image.stem}.png")
                plt.savefig(out_figure_path)
                plt.close()

            ##########################################################################################
            print(f"  - image {i_index + 1}/{len(images)}: success: {res[0][0] == label}, conf:{conf:.2f}, time: {t:.2f} s, prob other: {other[0]}/{other[1]*100:.1f}")
            # csv: label, file, count, accuracy, inliers, time, top_others
            file_detail_out.write(f"{label},{image},1,{'T' if res[0][0] == label else 'F'},{conf},{t},[{other[0]}:{other[1]:.3f}]\n")
            ##########################################################################################
            
            EVAL_LABEL_FRAMES += 1
            EVAL_TOTAL_TIME += t
            EVAL_TOTAL_CONF += res[0][1]
        
        acc = 100 * (1 - EVAL_LABEL_WRONG_FRAMES / EVAL_LABEL_FRAMES)

        ##########################################################################################
        print(f"Result: {label}, acc: {acc:.1f}")
        # csv: label, count, accuracy
        file_total_out.write(f"{label},{EVAL_LABEL_FRAMES},{acc}\n")
        ##########################################################################################

        EVAL_TOTAL_FRAMES += EVAL_LABEL_FRAMES
        EVAL_TOTAL_WRONG_FRAMES += EVAL_LABEL_WRONG_FRAMES

    # total accuracy
    acc = 100 * (1 - EVAL_TOTAL_WRONG_FRAMES / EVAL_TOTAL_FRAMES)
    conf = 100 * (EVAL_TOTAL_CONF / EVAL_TOTAL_FRAMES)

    ##########################################################################################
    print(f"Total accuracy: {acc:.1f}, Averate confidence: {conf:.2f}, Total time: {EVAL_TOTAL_TIME:.2f} s")
    file_detail_out.write(f"total,total,{EVAL_TOTAL_FRAMES},{acc},{conf},{EVAL_TOTAL_TIME},-\n")
    file_total_out.write(f"total,{EVAL_TOTAL_FRAMES},{acc}\n")
    ##########################################################################################

    file_detail_out.close()
    file_total_out.close()


    




