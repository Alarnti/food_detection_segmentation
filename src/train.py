import time
from typing import Dict

import numpy as np
import torch
import torchvision
from torch.optim import Optimizer
from torch.utils.data.dataloader import DataLoader
from torchmetrics.detection.map import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

import wandb
from dataset import FoodDataset


def create_maskrcnn(num_classes: int, hidden_layer_num: int = 256) -> torch.nn.Module:
    model_food = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model_food.roi_heads.box_predictor.cls_score.in_features
    model_food.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model_food.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model_food.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    for param in model_food.parameters():
        param.requires_grad = False
    for param in model_food.roi_heads.parameters():
        param.requires_grad = True

    return model_food


def log_wandb(losses: Dict[str, float], all_losses: float) -> None:
    loss_mask = losses["loss_mask"]
    loss_objectness = losses["loss_objectness"]
    loss_rpn_box_reg = losses["loss_rpn_box_reg"]
    loss_classifier = losses["loss_classifier"]
    loss_box_reg = losses["loss_box_reg"]

    wandb.log(
        {
            "loss_mask": loss_mask,
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
            "loss_classifier": loss_classifier,
            "loss_box_reg": loss_box_reg,
            "all_losses": all_losses.cpu().detach().numpy(),
        }
    )


def train_epoch(
    model: torch.nn.Module, tr_loader: DataLoader, optimizer: Optimizer, device: str
) -> None:
    for i_iter, (images, targets) in enumerate(tqdm(tr_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        losses_detached = {
            key: l.cpu().detach().numpy() for key, l in loss_dict.items()
        }

        log_wandb(losses_detached, losses)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    lr_scheduler: torch.optim.lr_scheduler,
) -> float:
    metr = MeanAveragePrecision(
        box_format="xyxy",
        iou_thresholds=None,
        rec_thresholds=[1, 10, 100],
        class_metrics=False,
    )

    mean_val_loss = 0
    with torch.no_grad():
        for i_iter, (images, targets) in enumerate(tqdm(val_loader)):
            model.train()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            mean_val_loss += losses.cpu().detach().numpy()

            losses_detached = {
                key: l.cpu().detach().numpy() for key, l in loss_dict.items()
            }

            log_wandb(losses_detached, losses)

            model.eval()
            res = model(images)
            metr.update(res, targets)

    mean_val_loss /= len(val_loader)

    wandb.log({"mean_val_loss": mean_val_loss})
    lr_scheduler.step(mean_val_loss)

    wandb.log({"mean_val_loss": lr_scheduler.optimizer.param_groups[0]["lr"]})

    wandb.log({"val_map": metr.compute()["map"]})

    return mean_val_loss


def save_model(model: torch.nn.Module, path: str, descr: str) -> None:
    torch.save(model.state_dict(), path + "maskrcnn_" + descr)


def train(
    num_epochs: int,
    model: torch.nn.Module,
    tr_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    optimizer: Optimizer,
    lr_scheduler: torch.optim.lr_scheduler,
    save_path: str,
) -> None:

    val_loss_min = 1e10
    model.cuda()
    for epoch in range(num_epochs):

        model.train()
        train_epoch(model, tr_loader, optimizer, device)

        mean_val_loss = validate(model, val_loader, optimizer, device, lr_scheduler)

        if mean_val_loss < val_loss_min:
            save_model(model, save_path, str(epoch) + "_" + str(mean_val_loss))
            val_loss_min = mean_val_loss


def main() -> None:
    RANDOM_SEED = 42

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    PROJECT_PATH = "/home/albert/Documents/projects/food_detection_segmentation/"
    TRAIN_IMAGES_PATH = PROJECT_PATH + "data/public_training_set_release_2.0/images/"
    TRAIN_LABELS = (
        PROJECT_PATH + "data/public_training_set_release_2.0/annotations.json"
    )

    VAL_IMAGES_PATH = PROJECT_PATH + "data/public_validation_set_2.0/images/"
    VAL_LABELS = PROJECT_PATH + "data/public_validation_set_2.0/annotations.json"

    DEVICE = "cuda"
    MODEL_SAVE_PATH = PROJECT_PATH + "runs/"

    NUM_CLASSES = 498

    train_ds = FoodDataset(TRAIN_IMAGES_PATH, TRAIN_LABELS)
    val_ds = FoodDataset(VAL_IMAGES_PATH, VAL_LABELS)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=6,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        shuffle=True,
        num_workers=6,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model_maskrcnn = create_maskrcnn(NUM_CLASSES)

    params = [p for p in model_maskrcnn.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    wandb.init(project="food", entity="alarnti")
    wandb.config = {"learning_rate": 0.001, "epochs": 100, "batch_size": 2}

    save_path = MODEL_SAVE_PATH + "run_" + str((time.time())) + "mask_rcnn" + "/"

    train(
        10,
        model_maskrcnn,
        train_loader,
        val_loader,
        DEVICE,
        optimizer,
        lr_scheduler,
        save_path,
    )


if __name__ == "__main__":
    main()
