# standard library
from pathlib import Path
from typing import *
import os
# thirdparty
import torch
import numpy as np
# h5py
import h5py


class FeatureDB:
    filename_: Path

    """ initialize database with file path """
    def __init__(self, path: Union[str, Path]) -> None:
        self.filename_ = str(path)
        if not os.path.exists(self.filename_):
            self.clear_all()


    """ clear all data in database """
    def clear_all(self) -> None:
        with h5py.File(self.filename_, 'w') as f:
            f.close()


    """ get all labels from database """
    def get_labels(self) -> List[str]:
        with h5py.File(self.filename_, 'r') as f:
            labels = list(f.keys())
            f.close()
        return labels
    

    """ delete label from database """
    def del_label(self, label: str) -> None:
        with h5py.File(self.filename_, 'a') as f:
            if label in f:
                del f[label]
            f.close()

    
    """ add feature to database for single label
    params:
        label: label name
        data: feature data
        image: raw image data, optional
    """
    def add_feature(
        self,
        label: str,
        data: Dict,
        image: Optional[np.ndarray] = None
    ) -> None:
        # data structure:
        # index: [value1, value2, value3 ...]
        with h5py.File(self.filename_, 'a') as f:
            if not label in f:
                f.create_group(label)
            id_ = str(len(f[label]))
            f[label].create_group(id_)
            for k, v in data.items():
                f[label][id_].create_dataset(k, data=v)
            if image is None:
                image = np.zeros((1, 1))
            f[label][id_].create_dataset('image', data=image)
            f.close()


    """ get features from single label
    params:
        label: label name
        image: whether include raw image data
    
    return: list of dict
    """
    def get_features(
        self,
        label: str,
        image: bool = False
    ) -> Dict[str, Dict]:
        datalist = []
        with h5py.File(self.filename_, 'r') as f:
            grp = f[label]
            for k, v in grp.items():
                data = {}
                for kk, vv in v.items():
                    if kk == 'image' and not image:
                        continue
                    data[kk] = torch.from_numpy(vv.__array__()).float()
                if 'keypoints' in data:
                    datalist.append(data)
            f.close()    
        return datalist
    

    """ check all labels """
    def check(self) -> None:
        with h5py.File(self.filename_, 'r') as f:
            for key in f.keys():
                print(f"Label: {key}, References: {len(list(f[key].keys()))}")
            f.close()