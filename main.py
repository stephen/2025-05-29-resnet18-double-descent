import argparse
from datetime import datetime
import multiprocessing as mp
import sys
from jaxtyping import Float
from dataclasses import asdict, dataclass, field
import torch as t
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from typing import Callable, Optional

from resnet18 import ResNet

default_device = t.device('mps') if t.backends.mps.is_available() else t.device('cuda') if t.cuda.is_available() else t.device('cpu')

TRAIN_TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ]
)

TEST_TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ]
)

def get_cifar() -> tuple[torchvision.datasets.cifar.CIFAR10, torchvision.datasets.cifar.CIFAR10]:
    training = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=True, transform=TRAIN_TRANSFORM)
    testing = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=False, transform=TEST_TRANSFORM)

    return training, testing

train_set, test_set = get_cifar()

@dataclass
class ModelArgs:
    num_classes: int = 10
    k: int = 64

@dataclass
class TrainingArgs:
    model_args: ModelArgs = field(default_factory=ModelArgs)
    batch_size: int = 128
    epochs: int = 4_000
    rank: int = 0

    wandb_group_name: Optional[str] = None
    wandb_run_name: Optional[str] = None

    device: t.device = default_device

def make_resnet18(args: ModelArgs) -> torchvision.models.ResNet:
    # Note: the paper mentions using bn -> relu -> conv (preactivation) resnet, but we use postactivation resnet.
    # https://gitlab.com/harvard-machine-learning/double-descent/-/blob/master/models/resnet18k.py?ref_type=heads#L8
    # I stuck with postactivation resnet18 because it's the default in pytorch.
    model = ResNet(**asdict(args))
    # model = t.compile(model) # XXX: I tried to get this to work but ran into errors.
    return model # type: ignore

class Trainer:
    def __init__(self, args: TrainingArgs):
        self.args = args

        self.tqdm_args: Callable[[str], dict] = lambda desc: {
            'position': self.args.model_args.k,
            'desc': f"{desc} k={self.args.model_args.k}",
            'leave': False,
        }
        pass

    def setup(self):
        self.model = make_resnet18(self.args.model_args).to(self.args.device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=1e-4)
        self.train_set = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        self.test_set = DataLoader(test_set, batch_size=self.args.batch_size * 2, shuffle=False)
        self.samples_trained = 0

        wandb.init(
            project="2025-05-29-resnet18-double-descent",
            config=asdict(self.args),
            group=self.args.wandb_group_name,
            name=self.args.wandb_run_name,
        )
        wandb.watch(self.model, log="gradients")


    def teardown(self):
        del self.train_set, self.test_set, self.optimizer, self.model
        if self.args.device.type == "cuda":
            t.cuda.empty_cache()

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
        return False

    def train_epoch(self, imgs: Float[t.Tensor, "b c w h"], labels: Float[t.Tensor, "b"]): # type: ignore
        imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)

        logits = self.model(imgs)

        loss = t.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()


        self.samples_trained += len(imgs)

        return loss

    @t.inference_mode()
    def evaluate(self):
        self.model.eval()

        correct, total, losses = 0.0, 0.0, []

        for imgs, labels in self.test_set:
            imgs, labels = imgs.to(self.args.device), labels.to(self.args.device)
            logits = self.model(imgs)

            losses.append(t.nn.functional.cross_entropy(logits, labels).item())
            correct += (logits.argmax(-1) == labels).sum().item()
            total += len(imgs)

        accuracy = correct / total
        loss = sum(losses) / len(losses)
        return loss, accuracy

    def train(self):
        self.setup()

        test_accuracy = self.evaluate()

        for epoch in tqdm(range(self.args.epochs), **self.tqdm_args("epochs")):
            self.model.train()
            train_loss = 0.0
            for imgs, labels in self.train_set:
                train_loss = self.train_epoch(imgs, labels)

                wandb.log({"train_loss": train_loss, "lr": self.optimizer.param_groups[0]['lr']}, step=self.samples_trained)

            test_loss, test_accuracy = self.evaluate()

            wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss, "epoch": epoch}, step=self.samples_trained)

        wandb.finish()

def train(args: TrainingArgs):
    # hack: figure out what mp.pool index we are.
    print(f"running job k={args.model_args.k}, rank={args.rank}")

    if t.cuda.is_available():
        args.device = t.device(f"cuda:{args.rank}")

    with Trainer(args) as trainer:
        trainer.train()

        path = f"data/run-k-{args.model_args.k}.pth"
        t.save(trainer.model.state_dict(), path)
        print(f"saved to {path=}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs-per-gpu', type=int, default=1, help='How many jobs to run per gpu (default=1)')
    return parser.parse_args()

def main():
    args = parse_args()
    gpu_count = t.cuda.device_count() if t.cuda.is_available() else 4 # if mps, we fake it.

    print("starting on", default_device)
    print("gpu count:", gpu_count, "jobs per gpu:", args.jobs_per_gpu, "total parallelism:", gpu_count * args.jobs_per_gpu)
    print("checking that tensors work", t.ones((1, 2, 3)).to(default_device).bool().all())

    run_group_name = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    jobs = [
        TrainingArgs(
            model_args=ModelArgs(k=k),
            wandb_group_name=run_group_name,
            wandb_run_name=f"{k=}",
            rank=k%gpu_count,
        )
        for k in range(1, 65)
    ]

    if t.cuda.is_available():
        mp.set_start_method("spawn") # cuda gets unhappy with fork.

    with mp.Pool(processes=gpu_count * args.jobs_per_gpu) as pool:
        pool.map(train, jobs)

    print("group:", run_group_name)
    print("ended at:", datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    sys.exit(0)

if __name__ == "__main__":
    main()
