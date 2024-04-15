# standard library
from pathlib import Path
from typing import *
import os
import time
# third party
from PIL import Image
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
# lightglue
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch

from .featuredb import FeatureDB
from . import utils

BASIC_CONFIG = {
    "superpoint": {
        "max_num_keypoints": 2048
    },
    "lightglue": {}
}


class Classifier:
    extractor_: SuperPoint
    matcher_: LightGlue
    database_: FeatureDB


    """ Initialize Classifier
    params:
        database: Union[str, Path]
            - path to feature database
        superpoint_conf: Dict
            - configuration for superpoint extractor
        lightglue_conf: Dict
            - configuration for lightglue matcher
        read_only: bool
            - read only mode for database
    """
    def __init__(
        self,
        database: Union[str, Path] = "./features.h5",
        superpoint_conf: Dict = BASIC_CONFIG["superpoint"],
        lightglue_conf: Dict = BASIC_CONFIG["lightglue"],
        read_only: bool = False
    ) -> None:
        # superpoint extractor
        self.extractor_ = SuperPoint(
            **superpoint_conf
        ).eval().cuda()
        # lightglue matcher
        self.matcher_ = LightGlue(
            features="superpoint", 
            **lightglue_conf
        ).eval().cuda()
        # feature database
        if read_only: assert os.path.exists(database), f"file for database not found: {database}"
        self.database_ = FeatureDB(path=database)



    """ Get all labels """
    def get_classes(self) -> List[str]:
        return self.database_.get_labels()



    """ Add feature data for single label
    params:
        label: str
            - label for current class
        images: List[Union[str, Path, np.ndarray, Image.Image]]
            - list of images for current class
                - str, Path: path to image
                - np.ndarray: numpy image data
                - Image.Image: PIL Image
    """
    def add_class(
        self,
        label: str,
        images: List[Union[str, Path, np.ndarray, Image.Image]],
    ) -> None:
        if not isinstance(images, list):
            images = [images]
        for image in tqdm(images):
            # read images
            if isinstance(image, (str, Path)):
                image = cv2.imread(str(image))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if isinstance(image, Image.Image):
                image = np.array(image)
            # convert to torch Tensor
            image = numpy_image_to_torch(image).cuda()
            # extract features
            feats = self.extractor_.extract(image)
            feats = {k: v[0].cpu().numpy() for k, v in feats.items()}
            # add to database
            self.database_.add_feature(
                label=label,
                data=feats,
                image=image.cpu().numpy()
            )



    """ Match features from query image to labeled features
    params:
        label: str
            - label to match
        feats: torch.Tensor
            - features from query image
        image: torch.Tensor
            - query image
        stop_threshold: float
            - stop threshold for matching
        savedir: Path
            - save directory for each matching label
        saveall: bool
            - save all results for each reference image
        get_extra: bool
            - get extra data for visualization
    
    return: Tuple[float, Dict]
        - (score, {extra data})
    """
    def _match_label(
        self,
        label: str,
        feats: torch.Tensor,
        image: torch.Tensor = None,
        stop_threshold: float = 0.8,
        savedir: Path = None,
        saveall: bool = False,
        get_extra: bool = False
    ) -> Tuple[float, Dict]: # (score, {extra data})
        # check dirs
        if saveall:
            if savedir is None: savedir = Path('./result')
            os.makedirs(savedir / f'match_{label}', exist_ok=True)
        score, final_matches, final_feature, idx = -1.0, None, None, 0
        query_image = image

        # query labeled features from database
        features = self.database_.get_features(label, image=(savedir is not None))

        # iter all features from current label
        for feature in features:
            # fit data structure
            for k in feature:
                feature[k] = torch.unsqueeze(feature[k], 0).cuda()
            match = rbd(self.matcher_({'image0':feats, 'image1':feature}))

            ref_feature = rbd(feature)
            query_feature = rbd(feats)

            # calculate max match rate
            score_i = match["matches"].shape[0] / ref_feature["keypoints"].shape[0]
            if score_i > score:
                score, final_matches, final_feature = score_i, match, ref_feature
            
            # visualize
            if (savedir is not None) and saveall:
                utils.draw_matches(
                    imgs=[query_image, ref_feature["image"]],
                    feats=[query_feature, ref_feature],
                    matches=match,
                    output=savedir / f'match_{label}' / f'{idx}_{(score_i * 100):.1f}.png',
                )
                idx += 1
            if score > stop_threshold:
                break

        # save visualize for current match
        if savedir:
            utils.draw_matches(
                imgs=[query_image, final_feature["image"]],
                feats=[query_feature, final_feature],
                matches=final_matches,
                output=savedir / f'match_{label}_{(score * 100):.1f}.png',
            )
        
        extra_data = {
            'query_image' : query_image,
            'ref_image' : final_feature["image"],
            'query_feature' : query_feature,
            'ref_feature' : final_feature,
            'matches' : final_matches
        } if get_extra else {}

        return score, extra_data



    """ Identify labels from query image
    params:
        query: Union[str, Path, np.ndarray, Image.Image]
            - str, Path: path to image
            - np.ndarray: numpy image data
            - Image.Image: PIL Image
        top_k: int
            - top k prob results
        stop_threshold: float
            - stop threshold for matching
        savedir: Union[str, Path]
            - save directory for each matching label
        saveall: bool
            - save all results for each reference image
        draw_match: bool
            - draw match results, cached on plt current fig

    return: Tuple[List[Tuple[str, float]], float] 
        - ([(label1, score1), (label2, score2), ...], time)
    """
    def identify(
        self,
        query: Union[str, Path, np.ndarray, Image.Image],
        top_k:          Optional[int] = 1,
        stop_threshold: Optional[float] = 0.8,
        gt_label:       Optional[str] = None,
        savedir:        Optional[Union[str, Path]] = None,
        saveall:        Optional[bool] = False,
        draw_match:     Optional[bool] = False,
    ) -> Tuple[List[Tuple[str, float]], float]: # ([(label1, score1), (label2, score2), ...], time)
        st = time.time()
        
        # read images
        if isinstance(query, (str, Path)):
            query = cv2.imread(str(query))
            query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        if isinstance(query, Image.Image):
            query = np.array(query)
        # convert to torch Tensor
        query = numpy_image_to_torch(query).cuda()

        # extract features
        feats = self.extractor_.extract(query)
        # print(f"Feature extraction time: {1000 * (time.time() - st):.2f}ms")

        # match
        labels = self.database_.get_labels()
        match_results = {}

        # match form all labels
        for label in labels:
            match_score, extra_data = self._match_label(
                label=label,
                feats=feats,
                image=query,
                stop_threshold=stop_threshold,
                savedir=savedir,
                saveall=saveall,
                get_extra=draw_match
            )
            match_results[label] = (match_score, extra_data)

            # if it must be current label
            if match_score > stop_threshold:
                break
                    
        # convert {label: (score, extra_data)} to (label, score, extra_data)
        match_results_list: List[Tuple[str, float, Dict]] \
            = [(k, v[0], v[1]) for k, v in match_results.items()]
        # sort by match score 
        match_results_list = sorted(match_results_list, key=lambda x: x[1], reverse=True)

        ##########################################################################################
        # Visualization
        ##########################################################################################
        if draw_match:
            flabel, fdata = match_results_list[0][0], match_results_list[0][2]
            utils.draw_matches(
                imgs=[fdata['query_image'], fdata['ref_image']],
                feats=[fdata['query_feature'], fdata['ref_feature']],
                matches=fdata['matches'],
                ref_label=flabel,
                color='r' if (gt_label and flabel != gt_label) else 'w'
            )
        ##########################################################################################
        
        # get top k:
        data = [] # [(label1, score1), (label2, score2), ...]
        for i in range(np.min([top_k, len(match_results_list)])):
            data.append(
                (match_results_list[i][0], match_results_list[i][1])
            )
        # fill with empty
        if len(data) < top_k:
            for _ in range(top_k - len(data)):
                data.append(('-', 0))
        
        # print(f"Matching time: {1000 * (time.time() - st):.2f}ms")
        return data, time.time() - st
    


    """ Identify labels from query video
    params:
        query: Union[str, Path, cv2.VideoCapture]
            - str, Path: path to video
            - cv2.VideoCapture: video capture object
        top_k: int
            - top k prob results
        stop_threshold: float
            - stop threshold for matching
        gt_label: str
            - ground truth label
        viz: bool
            - visualize results for each frame
        make_video: str
            - make video from matching frames
    
    return: Tuple[List[Tuple[str, float, float]], int, float]
        - ([(label1, proportion1, average inliers ratio1), ...], total frames, time)
    """
    def identify_video(
        self,
        query:          Union[str, Path, cv2.VideoCapture],
        top_k:          Optional[int] = 1,
        stop_threshold: Optional[float] = 0.8,
        gt_label:       Optional[str] = None,
        viz:            Optional[bool] = False,
        make_video:     Optional[str] = None
    ) -> Tuple[List[Tuple[str, float, float]], int, float]:
        st = time.time()

        # read video
        if isinstance(query, (str, Path)):
            query = cv2.VideoCapture(str(query))
        
        total = int(query.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(query.get(cv2.CAP_PROP_FPS))
        predicts = {} # label: [inliers ratio, count]

        # step frames
        out_video, resize = None, None
        pbar = tqdm(total=total)
        for _ in range(total):
            _, frame = query.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # res: ([(label1, score1), ...], time)
            res, _  = self.identify(
                query=frame, 
                top_k=top_k, 
                gt_label=gt_label,
                stop_threshold=stop_threshold,
                draw_match=(viz or make_video)
            )
            
            ##########################################################################################
            # Visualization
            ##########################################################################################
            pbar.set_description(f"Predict/Conf: '{res[0][0]}'/{100*res[0][1]:.2f}%, '{res[1][0]}'/{100*res[1][1]:.2f}%")
            pbar.update()
            if True:
                if viz or make_video:
                    image = utils.plt_to_numpy_rgb(resize)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if resize is None:
                        h, w, _ = image.shape
                        resize = (w, h)

                    if make_video:
                        if out_video is None:
                            out_video = cv2.VideoWriter(make_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    plt.close()
                    
                if viz:
                    cv2.imshow("match", image)
                    cv2.waitKey(5)
                if make_video:
                    out_video.write(image)
                if gt_label and res[0][0] != gt_label:
                    if make_video:
                        for _ in range(fps // 2): out_video.write(image)
                    if viz:
                        input("wrong prediction, press any key to continue")
            ##########################################################################################

            if gt_label is not None:
                res = [res[0]] # only check for wrong identify
            # add to predicts
            for (label, score) in res:
                if label not in predicts:
                    predicts[label] = [score, 1]
                else:
                    predicts[label][0] += score
                    predicts[label][1] += 1
        
        pbar.close()
        if out_video: out_video.release()

        # calculate average
        results = [] # [(label, proportion, inliers), ...]
        for label in predicts:
            score, cnt = predicts[label]
            results.append((label, cnt / total, score / cnt))

        # sort by proportion
        results = sorted(results, key=lambda x: x[1], reverse=True)
        results = results[:np.min([top_k, len(results)])]
        # fill with empty
        if len(results) < top_k:
            for _ in range(top_k - len(results)):
                results.append(('-', 0, 0))

        # move gt_label to head
        if gt_label is not None:
            for i in range(len(results)):
                if results[i][0] == gt_label:
                    results[0], results[i] = results[i], results[0]
                    break

        return results, total, time.time() - st



    """ Clear database """
    def clear(self, label: str = None):
        if label is None:
            self.database_.clear_all()
        else:
            self.database_.del_label(label)