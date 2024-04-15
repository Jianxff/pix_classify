# standard library
from pathlib import Path
import argparse
# pixclassify
import pixclassify


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixClassify")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--overwrite", action="store_true", default=False)
    args = parser.parse_args()

    datadir = Path(args.data)

    db_file = datadir / 'features.h5'
    classifier = pixclassify.Classifier(database=db_file)
    if args.overwrite:
        classifier.clear()

    for label_dir in datadir.glob('*'):
        if label_dir.is_dir():
            raw_images = sorted(list(label_dir.glob('*.[j|p][p|n]g')))
            label = label_dir.name
            print(f'extract feature for {label}')
            classifier.add_class(label, raw_images)
    

    

    