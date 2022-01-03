import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

data_root_dir = Path("../_DATASETS_/occMapDataset")
train_val_root_dirs = [data_root_dir / "train", data_root_dir / "val"]

new_radar_img_prefix = "r_20_t1"

for data_dir in train_val_root_dirs:
    # create a new directory to store the results
    r20_t1_img_path = data_dir / new_radar_img_prefix
    os.makedirs(str(r20_t1_img_path), exist_ok=True)

    # list of all r1 images in the data directory
    r1_img_paths = (data_dir / "r_1").glob("**/*.png")
    r20_img_paths = (data_dir / "r_20").glob("**/*.png")

    for r1_img_path, r20_img_path in tqdm(zip(r1_img_paths, r20_img_paths)):
        # get current img's postfix
        img_postfix = r1_img_path.name.split(("__"))[-1]

        # load the r1 and r20 image
        r1_img = np.asarray(Image.open(str(r1_img_path)))
        r20_img = np.asarray(Image.open(str(r20_img_path)))

        # create image from r20 static and r1 dynamic detections
        r20_t1_img = np.zeros_like(r20_img)
        r20_t1_img[r20_img == 255] = 255
        r20_t1_img[r1_img == 128] = 128            

        # viz
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.imshow(r1_img)
        # plt.subplot(1, 3, 2)
        # plt.imshow(r20_img)
        # plt.subplot(1, 3, 3)
        # plt.imshow(r20_t1_img)
        # plt.show()

        # store the r20_t1 image
        Image.fromarray(r20_t1_img).save(str(r20_t1_img_path / str(new_radar_img_prefix + "__" + img_postfix)))