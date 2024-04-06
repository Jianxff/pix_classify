# standard library
from pathlib import Path
import os
from typing import *
# third party
import argparse
# Pix Classify
from pixclassify import PixClassify

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PixClassify")
    parser.add_argument("--database", type=str, default="./data/features.h5")
    parser.add_argument("--query", type=str)
    parser.add_argument("--tempdir", type=str, default=None)
    parser.add_argument("--saveall", action="store_true")
    args = parser.parse_args()

    db_file = Path(args.database)
    classifier = PixClassify(database=db_file)

    tempdir = Path(args.tempdir) if args.tempdir else None
    if tempdir:
        os.makedirs(tempdir, exist_ok=True)

    query = Path(args.query)
    res = classifier.identify(
        query=query, 
        top_k=2,
        tempdir=tempdir,
        saveall=args.saveall
    )

    print(res)


