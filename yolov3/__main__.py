import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import sys


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_data():
    training_data = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    batch_size = 64

    training_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return (training_dataloader, test_dataloader, training_data, test_data)


class MyNeuralNetwork(nn.Module):
    """NeuralNetwork defines a basic pytorch neural network."""

    def __init__(self, device: str):
        super(MyNeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.device = device

    def forward(self, x):
        flattened = self.flatten(x)
        logits = self.linear_relu_stack(flattened)
        return logits


def make_model():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(device)

    model = MyNeuralNetwork(device).to(device)
    print(model)
    return model


def create_loss_fn_and_optimiser(model: MyNeuralNetwork):
    return (nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=1e-3))


def train(
    dataloader: DataLoader,
    model: MyNeuralNetwork,
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
    model: MyNeuralNetwork,
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


def eval(
    test_data: datasets.FashionMNIST,
    model: MyNeuralNetwork,
    i: int,
):
    model.eval()

    X, y = test_data[i][0], test_data[i][1]

    with torch.no_grad():
        X = X.to(model.device)
        prediction = model(X)
        predicted, actual = classes[prediction[0].argmax(0)], classes[y]

        return predicted, actual


if __name__ == "__main__":
    training_dataloader, test_dataloader, training_data, test_data = load_fashion_data()
    model = make_model()
    loss_fn, optimiser = create_loss_fn_and_optimiser(model)

    if "--train" in sys.argv:
        for epoch in range(20):
            print(f"---- {epoch=} ----\n")
            train(training_dataloader, model, loss_fn, optimiser)

            test(test_dataloader, model, loss_fn)

        torch.save(model.state_dict(), "model.pth")
    else:
        model.load_state_dict(torch.load("model.pth"))

        plt.style.use("grayscale")
        plt.axis("off")
        figure = plt.figure(figsize=(25, 25))

        for i in range(25):
            predicted, actual = eval(test_data, model, i)
            figure.add_subplot(5, 5, i + 1, frameon=False)
            plt.xlabel(f"{predicted=} {actual=}")
            plt.imshow(numpy.reshape(test_data[i][0], (28, 28)))

        figure.tight_layout()
        plt.savefig(f"graph.png")

    print("Done.")
