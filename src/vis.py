import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocotools.coco import COCO


def read_image(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def show_image_coco(
    image_id: int, image_path: str, coco_labels: COCO, with_mask: bool = True
) -> np.ndarray:
    im_info = coco_labels.loadImgs(image_id)[0]

    if not with_mask:
        plt.imshow(read_image(image_path + im_info["file_name"]))
    else:
        objs_ann = coco_labels.imgToAnns[image_id]

        res_image = read_image(image_path + im_info["file_name"])

        plt.imshow(res_image)
        coco_labels.showAnns(objs_ann)
