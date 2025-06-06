import argparse
from datetime import datetime
import multiprocessing as mp
import random
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
import numpy as np

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

def get_cifar(noise_rate: float = 0.0) -> tuple[torchvision.datasets.cifar.CIFAR10, torchvision.datasets.cifar.CIFAR10]:
    training = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=True, transform=TRAIN_TRANSFORM)

    if noise_rate != 0.0:
        labels = np.array(training.targets)
        label_count = len(labels)
        picks = np.random.choice(label_count, int(label_count * noise_rate), replace=False)
        print(f"introducing {noise_rate:.2%} noise, {len(picks)} labels")
        for i in picks:
            labels[i] = np.random.choice([c for c in range(10) if c != labels[i]])
        training.targets = labels.tolist()

    testing = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=False, transform=TEST_TRANSFORM)

    return training, testing

# get_cifar once at the start to prevent multiple workers from downloading separately.
get_cifar()

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
    label_noise: float = 0.0

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
        train_set, test_set = get_cifar(self.args.label_noise)

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
        test_loss, test_accuracy = self.evaluate()
        wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss, "epoch": 0}, step=self.samples_trained)

        for epoch in tqdm(range(self.args.epochs), **self.tqdm_args("epochs")):
            self.model.train()
            train_loss = 0.0
            for imgs, labels in self.train_set:
                train_loss = self.train_epoch(imgs, labels)

                wandb.log({"train_loss": train_loss, "lr": self.optimizer.param_groups[0]['lr']}, step=self.samples_trained)

            test_loss, test_accuracy = self.evaluate()
            wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss, "epoch": epoch}, step=self.samples_trained)

        file = f"run-k-{self.args.model_args.k}-noise-{self.args.label_noise}"
        path = f"data/{file}.pth"
        t.save(self.model.state_dict(), path)
        print(f"saved to {path=}")

        artifact = wandb.Artifact(file, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)

        wandb.finish()



def train(args: TrainingArgs):
    # hack: figure out what mp.pool index we are.
    print(f"running job k={args.model_args.k}, rank={args.rank}")

    if t.cuda.is_available():
        args.device = t.device(f"cuda:{args.rank}")

    with Trainer(args) as trainer:
        trainer.train()
def main():
    if t.cuda.is_available():
        mp.set_start_method("spawn") # cuda gets unhappy with fork.

    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=0.0, help='How much noise to apply')
    parser.add_argument('--jobs-per-gpu', type=int, default=1, help='How many jobs to run per gpu (default=1)')
    parser.add_argument('--full', action='store_true', default=False, help='Run all k=1...64 or run k=1, 2, 4, ...64')
    args = parser.parse_args()

    gpu_count = t.cuda.device_count() if t.cuda.is_available() else 1

    print("starting on", default_device)
    print("gpu count:", gpu_count, "jobs per gpu:", args.jobs_per_gpu, "total parallelism:", gpu_count * args.jobs_per_gpu)
    print("checking that tensors work", t.ones((1, 2, 3)).to(default_device).bool().all())

    run_group_name = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # instead of loading jobs starting from gpu 0, randomly pick start offset.
    random_start = random.randint(0, gpu_count - 1)
    k_set = range(1, 65) if args.full else [2**x for x in range(0, 7)]
    jobs = [
        TrainingArgs(
            model_args=ModelArgs(k=k),
            wandb_group_name=run_group_name,
            wandb_run_name=f"{k=}",
            label_noise=args.noise,
            rank=(i + random_start)%gpu_count,
        )
        for i, k in enumerate(k_set)
    ]
    print(f"{k_set=}")

    with mp.Pool(processes=gpu_count * args.jobs_per_gpu) as pool:
        pool.map(train, jobs)

    print("group:", run_group_name)
    print("ended at:", datetime.now().strftime('%Y-%m-%dT%H-%M-%S'))
    sys.exit(0)

if __name__ == "__main__":
    main()
