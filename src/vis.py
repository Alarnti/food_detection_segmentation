from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO


def read_image(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def show_image_coco(
    image_id: int,
    image_path: str,
    coco_labels: COCO,
    with_mask: bool = True,
    with_bb: bool = False,
) -> np.ndarray:
    im_info = coco_labels.loadImgs(image_id)[0]

    if not with_mask:
        plt.imshow(read_image(image_path + im_info["file_name"]))
    else:
        objs_ann = coco_labels.imgToAnns[image_id]

        res_image = read_image(image_path + im_info["file_name"])

        if not with_bb:
            for obj_ann in objs_ann:
                (x, y, w, h) = obj_ann["bbox"]
                p1 = np.array([y, x]).astype(int)
                p2 = np.array([y + h, x + w]).astype(int)
                res_image = cv2.rectangle(res_image, p1, p2, (255, 0, 0), 5)

        plt.imshow(res_image)
        coco_labels.showAnns(objs_ann)


def draw_bb(im: np.ndarray, bb: List[float]) -> np.ndarray:
    (x1, y1, x2, y2) = bb
    p1 = np.array([y1, x1]).astype(int)
    p2 = np.array([y2, x2]).astype(int)

    return cv2.rectangle(im, p1, p2, (255, 0, 0), 5)


def get_random_color() -> List[int]:
    return list(np.random.choice(range(256), size=3))


def show_mask_bb(torch_ds: torch.utils.data.Dataset, ind: int) -> None:
    im = torch_ds[ind][0].cpu().detach().numpy()
    im = np.moveaxis(im, 0, -1).copy()
    im = (255 * im).astype(np.uint8)

    for mask_el in torch_ds[ind][1]["masks"]:
        mask = mask_el.numpy()

        color = get_random_color()
        mask_cpy = np.zeros((mask.shape[0], mask.shape[1], 3), np.uint8)
        mask_cpy[:, :] = color
        mask_cpy = cv2.bitwise_and(mask_cpy, mask_cpy, mask=mask)

        im = cv2.addWeighted(mask_cpy, 0.4, im, 1, 0)

    for bb_el in torch_ds[ind][1]["boxes"]:
        im = draw_bb(im, bb_el)

    plt.imshow(im)
