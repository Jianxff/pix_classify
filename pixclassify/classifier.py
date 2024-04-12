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
# lightglue
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch

from .featuredb import FeatureDB
from .utils import draw_matches

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

    ## initialize
    def __init__(
        self,
        database: Union[str, Path] = "./features.h5",
        superpoint_conf: Dict = BASIC_CONFIG["superpoint"],
        lightglue_conf: Dict = BASIC_CONFIG["lightglue"]
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
        self.database_ = FeatureDB(path=database)


    ## get all labels
    def get_classes(self) -> List[str]:
        return self.database_.get_labels()


    ## add feature data for single label
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


    ## match form single label
    def _match_label(
        self,
        label: str,
        feats: torch.Tensor,
        image: torch.Tensor = None,
        stop_threshold: float = 0.8,
        tempdir: Path = None,
        saveall: bool = False
    ) -> float:
        # check dirs
        if saveall:
            if tempdir is None: tempdir = Path('./result')
            os.makedirs(tempdir / f'match_{label}', exist_ok=True)
        score, final_matches, final_feature, idx = -1.0, None, None, 0
        query_image = image

        # query labeled features from database
        features = self.database_.get_features(label, image=(tempdir is not None))

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
            if (tempdir is not None) and saveall:
                draw_matches(
                    imgs=[query_image, ref_feature["image"]],
                    feats=[query_feature, ref_feature],
                    matches=match,
                    output=tempdir / f'match_{label}' / f'{idx}_{(score_i * 100):.1f}.png',
                )
                idx += 1
            if score > stop_threshold:
                break

        # visualize for current match
        if tempdir:
            draw_matches(
                imgs=[query_image, final_feature["image"]],
                feats=[query_feature, final_feature],
                matches=final_matches,
                output=tempdir / f'match_{label}_{(score * 100):.1f}.png',
            )
        return score


    ## identify labels
    def identify(
        self,
        query: Union[str, Path, np.ndarray, Image.Image],
        top_k: int = 1,
        stop_threshold: float = 0.8,
        tempdir: Union[str, Path] = None,
        saveall: bool = False
    ) -> List[str]:
        # read images
        if isinstance(query, (str, Path)):
            query = cv2.imread(str(query))
            query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
        if isinstance(query, Image.Image):
            query = np.array(query)
        # convert to torch Tensor
        query = numpy_image_to_torch(query).cuda()

        # extract features
        st = time.time()
        feats = self.extractor_.extract(query)
        nd = time.time()
        print(f"Feature extraction time: {1000 * (nd - st):.2f}ms")

        # match
        labels = self.database_.get_labels()
        match_scores = np.array([0.0] * len(labels))

        st = time.time()
        # match form all labels
        for i in range(len(labels)):
            label = labels[i]
            match_scores[i] = self._match_label(
                label=label,
                feats=feats,
                image=query,
                stop_threshold=stop_threshold,
                tempdir=tempdir,
                saveall=saveall
            )
            if(match_scores[i] > stop_threshold):
                break
                    
        # sort by match rate
        labels = sorted(labels, key=lambda x: match_scores[labels.index(x)], reverse=True)
        match_scores = sorted(match_scores, reverse=True)
        
        # get top k:
        data = []
        for i in range(np.min([top_k, len(labels)])):
            data.append((labels[i], match_scores[i]))
        
        print(f"Matching time: {1000 * (time.time() - st):.2f}ms")

        return data
    
    ## clear labeled data
    def clear(self, label: str = None):
        if label is None:
            self.database_.clear_all()
        else:
            self.database_.del_label(label)