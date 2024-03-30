import os
import argparse
import wandb
from tqdm import trange

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils

from util import MetricsCalculator, RealNVPLoss

parser = argparse.ArgumentParser(description="Glow trainer")

# Train settings
parser.add_argument("--epochs", default=60, type=int)
parser.add_argument("--batch", default=16, type=int, help="batch size")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate") # Results in NaN weigths if set to higher value
parser.add_argument("--loss_scale", type=float, default=1.)
parser.add_argument("--n_samples", default=20, type=int, help="number of samples")
parser.add_argument("--resume", type=str, default=None, help="resume file name")

# Model Architechture
parser.add_argument("--model", help="Name of the architechture", 
                    choices=["r_base", "r_vit", "r_revvit"], default="r_base")
parser.add_argument("--model-args", nargs="+", default=[])

# Misc
parser.add_argument("--debug", action="store_true")


img_size = 32
temp = 0.7

def main():
    transform_train = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ]
    )

    trainset = datasets.CIFAR10(root='data', download=True, transform=transform_train)
    if args.debug:
        trainset = Subset(trainset, list(range(10)))
    trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True)
    testset = datasets.CIFAR10(root='data', train=False, transform=transform_test)
    testset = Subset(testset, list(range(10 if args.debug else 1000)))
    testloader = DataLoader(testset, batch_size=args.batch, shuffle=False)

    model_module = __import__("models." + args.model, fromlist=["get_model"])

    model = model_module.get_model(args.model_args)

    model = model.to(device)
    if device == 'cuda':
        model = nn.DataParallel(model)

    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print(f'Resuming from checkpoint at checkpoints/{args.resume}...')
        assert os.path.isdir('checkpoints'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'checkpoints/{args.resume}')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs * 0.5, args.epochs * 0.75], verbose=True)
    global criterion
    criterion = RealNVPLoss()

    print('Initializing KID and FID..')
    global metrics_calculator
    metrics_calculator = MetricsCalculator(device)
    metrics_calculator.initialize(testloader)

    print("Training...")
    for epoch in trange(start_epoch, args.epochs):
        train(epoch, model, optimizer, trainloader)
        test(epoch, model, testloader)
        lr_scheduler.step(epoch)

def train(epoch, model: nn.Module, optimizer, dataloader: DataLoader):
    model.train()
    
    total_loss = 0
    for i, (images, _) in enumerate(dataloader):
        optimizer.zero_grad()
        images = images.to(device)

        z, sldj = model(images)

        loss = criterion(z, sldj)
        loss *= args.loss_scale
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    wandb.log({"train_loss": avg_loss}, epoch)
    print("Train Loss", epoch, avg_loss)


def test(epoch, model: nn.Module, dataloader: DataLoader):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for images, _ in dataloader:
            images = images.to(device)

            z, sldj = model(images)

            loss = criterion(z, sldj)
            loss *= args.loss_scale
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    state = {
        "model": model.state_dict(),
        "test_loss": avg_loss,
        "epoch": epoch
    }
    if not args.debug:
        os.makedirs("checkpoints", exist_ok=True)
        last_path = os.path.join("checkpoints", f"{args.model}_{epoch - 3}.pt")
        if os.path.exists(last_path):
            os.remove(last_path)
        path = os.path.join("checkpoints", f"{args.model}_{epoch}.pt")
        torch.save(state, path)

        arti = wandb.Artifact(args.model, "model")
        arti.add_file(path)
        wandb.log_artifact(arti, epoch)

    sample_images = sample(model)

    wandb.log({"test_loss": avg_loss}, epoch)
    print("Test Loss", epoch, avg_loss)
    wandb.log({"image_metrics": metrics_calculator.compute(sample_images)}, epoch)

    images_concat = utils.make_grid(sample_images, nrow=args.n_samples // 2, padding=2, pad_value=255, normalize=True, range=(-0.5, 0.5))
    wandb.log({"samples": wandb.Image(transforms.ToPILImage()(images_concat))}, epoch)


def sample(model):
    z = torch.randn((args.n_samples, 3, img_size, img_size), dtype=torch.float32, device=device)
    with torch.inference_mode():
        x, _ = model(z, reverse=True)
        samples = torch.sigmoid(x)
    return samples


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="All Architectures", 
            config={
                "architechture": args.model,
                "dataset": "CIFAR-10",
                "epochs": args.epochs,
                "learning_rate": args.lr,
                },
            dir="logs",
            save_code=True,
            mode="disabled" if args.debug else "online")

    main()