# PixClassify
This repository contains the **robust classifier** for similar objects and difficult view angles, based on [superpoint](https://github.com/rpautrat/SuperPoint) and [lightglue](https://github.com/cvg/LightGlue).

### Installation
```bash
# clone the repository
cd pix_classify
conda env create -f environment.yml
conda activate pixclassify
```

### Usage
- Generate the database
```bash
## prepare the dataset as the following structure
# - /path/to/dataset
#   - class_label_1
#     - xx.jpg/png
#     - xx.jpg/png
#       ...
#  - class_label_2
#     - xx.jpg/png
#       ...
##

## run command to extract and store features
#  the database files will be stored in /path/to/dataset/features.h5
#  if you want to overwrite all database, use --overwrite as a parameter
python train.py --data /path/to/dataset
```

- Identify for image or video
```bash
## run command to identify the object
#  the result will be shown in the terminal
#  type "--top-k ${num}" to specify the number of top-k matches
python infer.py --database /path/to/feature.h5 --query /path/to/query(jpg/png/mp4/mov/avi) 

## if you want to visualize the result, use "--viz" as parameter

## add "--gt $class_label" to specify the ground truth of the query image

## (image only) if you want to save the result of matches, use "--savedir ${/path/to/savedir}" as parameter
#  the final match of each class will be stored in the dir

## (image only) if you want to save all result of each reference of each class, use "--saveall" as parameter 

## (video only) if you want to save video for matching, use "--make-video=$/path/to/video/mp4" as parameter
```

- Evaluate for labeled images and videos
```bash
## run command to evaluate the performance
#  the result will be saved in '/path/to/outdir/eval_deatil.csv' and '/path/to/outdir/eval_summary.csv'
python evaluate.py --database /path/to/feature.h5 --data /path/to/test/data --out-dir /path/to/out/dir

## The test data should be structured as the following
# - $/path/to/test/data  
#   - label1       
#       - video1.mp4 
#       - video2.mp4 
#       - image1.png  
#   - label2        
#       - video3.mp4  
#       - image2.jpg 

## if you want to save video and image from matching, use "--savefig" as parameter, files will be saved in /path/to/outdir/label
```
