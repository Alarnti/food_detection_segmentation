import os

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO


class FoodDataset(torch.utils.data.Dataset):
    def __init__(
        self, img_path: str, coco_ds_path: str, trans: torchvision.transforms = None
    ):
        self.img_path = img_path
        self.trans = trans

        self.coco_ds = COCO(coco_ds_path)
        self.img_ids = sorted(self.coco_ds.getImgIds())

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx: index of sample
        return:
            dict containing:
            - np.ndarray image of shape (H, W)
            - target (dict) containing:
                - boxes:    FloatTensor[N, 4], N being the n° of instances and it's bounding
                boxe coordinates in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H;
                - labels:   Int64Tensor[N], class label (0 is background);
                - image_id: Int64Tensor[1], unique id for each image;
                - area:     Tensor[N], area of bbox;
                - iscrowd:  UInt8Tensor[N], True or False;
                - masks:    UInt8Tensor[N, H, W], segmantation maps;
        """
        img_id = self.img_ids[idx]
        img_obj = self.coco_ds.loadImgs(img_id)[0]
        anns_obj = self.coco_ds.loadAnns(self.coco_ds.getAnnIds(img_id))

        img = Image.open(os.path.join(self.img_path, img_obj["file_name"]))

        bboxes = np.array([ann["bbox"] for ann in anns_obj])
        masks = np.array([self.coco_ds.annToMask(ann) for ann in anns_obj])
        areas = np.array([ann["area"] for ann in anns_obj])

        num_bbs = len(bboxes)
        for i in range(num_bbs):
            bboxes[i][2] += bboxes[i][0]
            bboxes[i][3] += bboxes[i][1]

        boxes = torch.as_tensor(bboxes, dtype=torch.float32)
        labels = torch.ones(len(anns_obj), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = torch.as_tensor(areas)
        iscrowd = torch.zeros(len(anns_obj), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.trans is not None and False:
            img, target = self.trans(img, target)

        return torchvision.transforms.ToTensor()(img), target

    def __len__(self) -> int:
        return len(self.img_ids)
