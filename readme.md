# PixClassify
This repository contains the **robust classifier** for similar objects and difficult view angles, based on [superpoint](https://github.com/rpautrat/SuperPoint) and [lightglue](https://github.com/cvg/LightGlue).

### Installation
```bash
# clone the repository
cd PixClassify
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
python extract.py --data /path/to/dataset
```

- Identify the object
```bash
## run command to identify the object
#  the result will be shown in the terminal
python identify.py --database /path/to/feature.h5 --query /path/to/query(jpg/png) 

## if you want to save the result of matches, use "--tempdir /path/to/tempdir" as parameter
#  the final match of each class will be stored in the dir

## if you want to save all result of each reference of each class, use "--saveall" as parameter 
```