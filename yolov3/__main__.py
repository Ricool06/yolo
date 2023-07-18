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
    desired_size = [600, 600]
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

    batch_size = 64

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
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
            ),
            nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=3,
            #     out_channels=9,
            #     kernel_size=9,
            #     stride=3
            # ),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=9,
            #     out_channels=81,
            #     kernel_size=81,
            #     stride=9
            # ),
            # nn.LeakyReLU(),
            # nn.Conv2d(
            #     in_channels=81,
            #     out_channels=3,
            #     kernel_size=3
            # ),
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


if __name__ == "__main__":
    train_loader, val_loader, test_data = load_data()
    model = make_model()

    with torch.no_grad():
        for image_tensor_batch, target in val_loader:
            print(f"{image_tensor_batch.size()=}")
            
            out = model(image_tensor_batch.to(model.device))

            for i in range(5):
                in_img = to_pil_image(image_tensor_batch[i])
                in_img.save(f"test_images/in{i}.jpeg")
                
                out_img = to_pil_image(out[i])
                out_img.save(f"test_images/out{i}.jpeg")

            
            break
