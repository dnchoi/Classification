import argparse
import os
import time
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from tqdm import tqdm

from libs.dataloader import dataset
from libs.logger import Logger
from models import resnet

sns.set(style="darkgrid", font_scale=1.2)
logger = Logger(name="log/", save=False)


def hyp_parser(parser):
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=1e-3,
        type=float,
        help="initial learning rate",
    )
    # parser.add_argument("--momentum", default=0.9, type=float, help="Momentum value for optim")
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="Maximum epoch stop for training",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
    )
    return parser


def argparses():
    parser = argparse.ArgumentParser()
    parser = hyp_parser(parser)

    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--save",
        type=str,
        default="weights",
        help="Dataset path for training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/{your data class}",
        help="Dataset path for training",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="datasets/classes.txt",
        help="Dataset class text file path",
    )
    args = parser.parse_args()

    return args


def data_distribution(logger, data, classes):
    pbar = tqdm(data)

    name_cls = []
    num = []
    n = 0
    for i in pbar:
        for j in range(len(classes)):
            if classes[j] in i:
                name_cls.append(classes[j])
        num.append(n)
        n += 1
        # num_cls[j].append(i)
        pbar.set_description_str(f"{i}")

    frame = zip(num, data, name_cls)
    df = pd.DataFrame(frame, columns=["number", "path", "class"])

    logger.info(f"\n{df}")
    # sns.countplot(x="class", hue="class", data=df)
    # plt.show()


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group["lr"]


# function to calculate metric per mini-batch
def metric_batch(output, target):
    pred = output.argmax(1, keepdim=True)
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects


# function to calculate loss per mini-batch
def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    metric_b = metric_batch(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), metric_b


# function to calculate loss and metric per epoch
def loss_epoch(model, loss_func, dataset_dl, device, sanity_check=False, opt=None):
    running_loss = 0.0
    running_metric = 0.0
    len_data = len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)

        loss_b, metric_b = loss_batch(loss_func, output, yb, opt)

        running_loss += loss_b

        if metric_b is not None:
            running_metric += metric_b

        if sanity_check is True:
            break

    loss = running_loss / len_data
    metric = running_metric / len_data

    return loss, metric


def train_val(net, args, train_loader, val_loader, test_loader, device, model_name):
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    opt = optim.Adam(net.parameters(), lr=0.001)

    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=10)

    loss_list = {"train": [], "val": []}
    matric_list = {"train": [], "val": []}

    pbar = tqdm(range(args.epochs))
    best_loss = None
    net = nn.DataParallel(net, output_device=0)
    for epoch in pbar:
        start = time.time()
        lr = get_lr(opt)

        net.train()

        train_loss, train_matric = loss_epoch(net, loss_func, train_loader, device)
        loss_list["train"].append(train_loss)
        matric_list["train"].append(train_matric)

        net.eval()

        with torch.no_grad():
            val_loss, val_matric = loss_epoch(net, loss_func, val_loader, device)

        loss_list["val"].append(val_loss)
        matric_list["val"].append(val_matric)

        if epoch == 0:
            best_loss = val_loss
        elif val_loss < best_loss:
            logger.info(f"Best loss update : {best_loss} -> {val_loss}")
            best_loss = val_loss

            os.makedirs(os.path.join(args.save, model_name), exist_ok=True)
            infer_name = os.path.join(args.save, model_name, "infer_last.pth")
            total_name = os.path.join(args.save, model_name, "total_last.pth")
            resume_name = os.path.join(args.save, model_name, "resume_last.pth")
            # inference model save
            torch.save(net.state_dict(), infer_name)
            # all model layout save
            torch.save(net, total_name)
            # resume checkpoint save
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": train_loss,
                },
                resume_name,
            )
        else:
            pass

        lr_scheduler.step(val_loss)
        end = time.time()
        pbar.set_description_str(
            f"Epoch : {epoch}/{args.epochs-1}, \
            lr = {lr}, t_loss = {train_loss}, \
            v_loss = {val_loss}, \
            acc = {100 * val_matric}, \
            time = {end - start}"
        )

    return net, loss_list, matric_list


def training(args):
    logger.info(f"{args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)

    train_list = sorted(glob(os.path.join(args.dataset, "train", "*.jpg")))
    test_list = sorted(glob(os.path.join(args.dataset, "test", "*.jpg")))

    # logger.debug(train_list)
    # logger.debug(test_list)

    classes = []
    with open(args.classes) as f:
        classes = f.read().splitlines()
    logger.info(classes)

    data_distribution(logger, train_list, classes)
    train_list, val_list = train_test_split(train_list, test_size=0.2)

    # data Augumentation
    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_data = dataset(
        file_list=train_list,
        classes=classes,
        transform=train_transforms,
    )
    val_data = dataset(
        file_list=val_list,
        classes=classes,
        transform=val_transforms,
    )
    test_data = dataset(
        file_list=test_list,
        classes=classes,
        transform=test_transforms,
    )

    do_train_list = [
        ["resnet152", resnet.resnet152().to(device=device), 6],
        ["resnet101", resnet.resnet101().to(device=device), 10],
        ["resnet50", resnet.resnet50().to(device=device), 16],
        ["resnet34", resnet.resnet34().to(device=device), 48],
        ["resnet18", resnet.resnet18().to(device=device), 80],
    ]
    for models in do_train_list:
        train_loader = DataLoader(
            dataset=train_data,
            batch_size=models[2],
            shuffle=True,
            num_workers=args.workers,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            dataset=test_data,
            batch_size=models[2],
            shuffle=True,
            num_workers=args.workers,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=models[2],
            shuffle=True,
            num_workers=args.workers,
            persistent_workers=True,
        )

        net = models[1]
        summary(net, (3, args.input_size, args.input_size))
        logger.info(f"Trained model name : {models[0]}")
        model, loss_hist, metric_hist = train_val(
            net, args, train_loader, val_loader, test_loader, device, models[0]
        )
        # Train-Validation Progress
        num_epochs = args.epochs
        save_dir = os.path.join(args.save, models[0])
        os.makedirs(os.path.join(args.save, models[0]), exist_ok=True)
        # plot loss progress
        plt.clf()
        plt.title("Train-Val Loss")
        plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
        plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
        plt.ylabel("Loss")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.savefig(f'{os.path.join(save_dir, "train_val.png")}', dpi=300)

        # plot accuracy progress
        plt.clf()
        plt.title("Train-Val Accuracy")
        plt.plot(range(1, num_epochs + 1), metric_hist["train"], label="train")
        plt.plot(range(1, num_epochs + 1), metric_hist["val"], label="val")
        plt.ylabel("Accuracy")
        plt.xlabel("Training Epochs")
        plt.legend()
        plt.savefig(f'{os.path.join(save_dir, "accuracy.png")}', dpi=300)

    # pbar = tqdm(range(args.epochs))
    # for epoch in pbar:
    #     epoch_loss = 0
    #     epoch_accuracy = 0
    #     for data, label in train_loader:
    #         data = data.to(device)
    #         label = label.to(device)

    #         output = model(data)
    #         loss = criterion(output, label)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         acc = (output.argmax(dim=1) == label).float().mean()
    #         epoch_accuracy += acc / len(train_loader)
    #         epoch_loss += loss / len(train_loader)

    #     pbar.set_description_str(
    #         f"Epoch : {epoch + 1}, train accuracy : {epoch_accuracy}, train loss : {epoch_loss}"
    #     )

    #     with torch.no_grad():
    #         epoch_val_accuracy = 0
    #         epoch_val_loss = 0
    #         for data, label in val_loader:
    #             data = data.to(device)
    #             label = label.to(device)

    #             val_output = model(data)
    #             val_loss = criterion(val_output, label)

    #             acc = (val_output.argmax(dim=1) == label).float().mean()
    #             epoch_val_accuracy += acc / len(val_loader)
    #             epoch_val_loss += val_loss / len(val_loader)

    #         pbar.set_description_str(
    #             f"Epoch : {epoch + 1}, val accuracy : {epoch_val_accuracy}, val loss : {epoch_val_loss}"
    #         )


def main():
    opt = argparses()
    training(opt)


if __name__ == "__main__":
    main()
