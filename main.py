from jaxtyping import Float
from dataclasses import asdict, dataclass, field
import torch as t
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from resnet18 import ResNet

device = t.device('mps') if t.backends.mps.is_available() else t.device('cpu')

TRANSFORM = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        # https://github.com/kuangliu/pytorch-cifar/issues/19
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261]),
    ]
)

def get_cifar() -> tuple[torchvision.datasets.cifar.CIFAR10, torchvision.datasets.cifar.CIFAR10]:
    training = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=True, transform=TRANSFORM)
    testing = torchvision.datasets.CIFAR10(Path.cwd() / "data", download=True, train=False, transform=TRANSFORM)

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
        pass

    def setup(self):
        self.model = make_resnet18(self.args.model_args).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=1e-4)
        self.train_set = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True, num_workers=8, persistent_workers=True)
        self.test_set = DataLoader(test_set, batch_size=self.args.batch_size * 2, shuffle=False, num_workers=2, persistent_workers=True)
        self.samples_trained = 0

        wandb.init(project="2025-05-29-resnet18-double-descent", config=asdict(self.args))
        wandb.watch(self.model, log="gradients")

        pass

    def train_epoch(self, imgs: Float[t.Tensor, "b c w h"], labels: Float[t.Tensor, "b"]): # type: ignore
        imgs, labels = imgs.to(device), labels.to(device)

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

        for imgs, labels in tqdm(self.test_set, desc="evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
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

        for epoch in tqdm(range(self.args.epochs), desc="epochs"):
            pbar = tqdm(self.train_set, desc="training")
            self.model.train()
            train_loss = 0.0
            for imgs, labels in pbar:
                train_loss = self.train_epoch(imgs, labels)
                pbar.set_postfix(loss=f"{train_loss:.3f}", ex_seen=f"{self.samples_trained:06}")

                wandb.log({"train_loss": train_loss, "lr": self.optimizer.param_groups[0]['lr']}, step=self.samples_trained)

            test_loss, test_accuracy = self.evaluate()

            pbar.set_postfix(loss=f"{train_loss:.3f}", accuracy=f"{test_accuracy:.2f}", ex_seen=f"{self.samples_trained:06}")
            wandb.log({"test_accuracy": test_accuracy, "test_loss": test_loss, "epoch": epoch}, step=self.samples_trained)

        wandb.finish()

def main():
    print("running on", device)
    print("checking that tensors work", t.ones((1, 2, 3)).to(device).bool().all())
    trainer = Trainer(TrainingArgs(model_args=ModelArgs(
        k=33,
    )))
    trainer.train()

if __name__ == "__main__":
    main()
