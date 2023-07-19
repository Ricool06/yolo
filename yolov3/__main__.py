import sys
from typing import TypedDict
from torch import Tensor
import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, CenterCrop, Lambda
from torchvision.transforms.functional import to_tensor, center_crop, to_pil_image
from PIL.JpegImagePlugin import JpegImageFile
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torch import nn


def draw_test_image(data: datasets.VOCDetection):
    image_tensor: Tensor
    target: Tensor

    for image_tensor, target in data:
        img = to_pil_image(image_tensor)

        draw = ImageDraw.Draw(img)
        x, y, w, h = target.flatten().tolist()
        rect = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        draw.rectangle(rect, outline=(40, 180, 80, 40), width=4)

        img.save("test.jpeg")
        break


def equal_size_transforms(input, target):
    desired_size = [512, 512]
    original_size = [
        int(target["annotation"]["size"]["width"]),
        int(target["annotation"]["size"]["height"]),
    ]
    left_pad = (desired_size[0] - original_size[0]) / 2
    top_pad = (desired_size[1] - original_size[1]) / 2

    tensor_input = to_tensor(input)
    cropped_input = center_crop(tensor_input, desired_size)
    # print(f"{cropped_input.size()=}")

    bounding_boxes = []
    for obj in target["annotation"]["object"]:
        bbox = obj["bndbox"]
        xmin, ymin, xmax, ymax = (
            int(bbox["xmin"]) + left_pad,
            int(bbox["ymin"]) + top_pad,
            int(bbox["xmax"]) + left_pad,
            int(bbox["ymax"]) + top_pad,
        )
        width = xmax - xmin
        height = ymax - ymin
        yolo_bbox = [xmin + width / 2, ymin + height / 2, width, height]
        bounding_boxes.append(yolo_bbox)
        break

    tensor_bbox_target = Tensor(bounding_boxes)
    # print(f"{tensor_bbox_target.size()=}")
    return cropped_input, tensor_bbox_target


def load_data():
    train_data = datasets.VOCDetection(
        "data",
        year="2012",
        image_set="train",
        download=False,
        transforms=equal_size_transforms,
    )
    val_data = datasets.VOCDetection(
        "data",
        year="2012",
        image_set="trainval",
        download=False,
        transforms=equal_size_transforms,
    )
    test_data = datasets.VOCDetection(
        "data",
        year="2012",
        image_set="val",
        download=False,
        transforms=equal_size_transforms,
    )

    if "--draw-test" in sys.argv:
        draw_test_image(val_data)

    batch_size = 4

    return (
        DataLoader(train_data, batch_size=batch_size),
        DataLoader(val_data, batch_size=batch_size),
        test_data,
    )


class Yolo(nn.Module):
    """Yolo defines a basic pytorch YOLOv3 neural network."""

    def __init__(self, device: str):
        super(Yolo, self).__init__()

        kernel_size = 3

        # self.layer = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=64,
        #     kernel_size=kernel_size,
        #     stride=(1, 1),
        #     padding=(kernel_size / 2, kernel_size / 2),
        # )
        self.layer = nn.Sequential(
            # nn.Conv2d(
            #     in_channels=3,
            #     out_channels=3,
            #     kernel_size=3,
            # ),
            # nn.LeakyReLU(),

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                bias=False
            ),
            nn.BatchNorm2d(32, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                bias=False
            ),
            nn.BatchNorm2d(64, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=32,
                kernel_size=1,
                stride=1,
                padding=1 // 2,
                bias=False
            ),
            nn.BatchNorm2d(32, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=3 // 2,
                bias=False
            ),
            nn.BatchNorm2d(64, momentum=0.03, eps=1E-4),
            nn.LeakyReLU(0.1, inplace=True),

            # YOLOv3 from config^^^^ in progress, up to: https://github.com/ultralytics/yolov3/blob/06138062869c41d3df130e07c5aa92fa5a01dad5/cfg/yolov3.cfg#L59
        )

        self.device = device

    def forward(self, x):
        return self.layer(x)


def make_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(device)

    model = Yolo(device).to(device)
    print(model)
    return model


def train(
    dataloader: DataLoader,
    model: Yolo,
    loss_fn: nn.CrossEntropyLoss,
    optimiser: torch.optim.SGD,
):
    dataset_size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(model.device), y.to(model.device)

        prediction = model(X)
        loss = loss_fn(prediction, y)

        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"{loss=} [{current:>5d}/{dataset_size}]")


def test(
    dataloader: DataLoader,
    model: Yolo,
    loss_fn: nn.CrossEntropyLoss,
):
    dataset_size = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    model.eval()

    loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(model.device), y.to(model.device)

            prediction = model(X)
            loss = loss_fn(prediction, y).item()

            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()

    loss /= number_of_batches
    correct /= dataset_size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")


def create_loss_fn_and_optimiser(model: Yolo):
    return (nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=1e-3))


if __name__ == "__main__":
    train_loader, val_loader, test_data = load_data()
    model = make_model()
    loss_fn, optimiser = create_loss_fn_and_optimiser(model)

    if "--train" in sys.argv:
        for epoch in range(20):
            print(f"---- {epoch=} ----\n")
            train(train_loader, model, loss_fn, optimiser)

            test(val_loader, model, loss_fn)

        torch.save(model.state_dict(), "model.pth")

    with torch.no_grad():
        for image_tensor_batch, target_batch in val_loader:
            print(f"{image_tensor_batch.size()=}")
            print(f"{target_batch.size()=}")
            
            out = model(image_tensor_batch.to(model.device))

            for each_out in out:
                print(f"{each_out.size()=}")
                break

            # for i in range(5):
            #     in_img = to_pil_image(image_tensor_batch[i])
            #     in_img.save(f"test_images/in{i}.jpeg")
                
            #     out_img = to_pil_image(out[i])
            #     out_img.save(f"test_images/out{i}.jpeg")

            
            break
