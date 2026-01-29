import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset

SEED = 2026
BATCH_SIZE = 64
EPOCHS = 40
MAX_LR = 0.01

BASE_DIR = "/scratch/m25csa032/assignment1"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        return x, y, idx

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

train_ds_aug = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
train_ds_plain = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_test)
test_ds_plain = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)

# Deterministic train/val split with a fixed seed; validation uses NO augmentation.
g = torch.Generator().manual_seed(SEED)
perm = torch.randperm(len(train_ds_aug), generator=g).tolist()
train_size = int(0.9 * len(perm))
train_idx = perm[:train_size]
val_idx = perm[train_size:]

trainset = IndexedDataset(Subset(train_ds_aug, train_idx))
valset = IndexedDataset(Subset(train_ds_plain, val_idx))
testset = IndexedDataset(test_ds_plain)

trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=1)
valloader = DataLoader(valset, BATCH_SIZE, shuffle=False, num_workers=1)
testloader = DataLoader(testset, BATCH_SIZE, shuffle=False, num_workers=1)

classes = ('Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck')

class Block(nn.Module):
    def __init__(self, in_c, out_c, stride=1, residual=False):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out += self.shortcut(x)
        return torch.relu(out)

class FlexibleCNN(nn.Module):
    def __init__(self, residual=False):
        super().__init__()
        self.in_c = 64
        self.conv = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 5, 1, residual)
        self.layer2 = self._make_layer(128, 5, 2, residual)
        self.layer3 = self._make_layer(256, 5, 2, residual)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def _make_layer(self, out_c, blocks, stride, residual):
        layers = []
        for s in [stride] + [1]*(blocks-1):
            layers.append(Block(self.in_c, out_c, s, residual))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn(self.conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x,1))

def train_model(model, name):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.OneCycleLR(opt, MAX_LR, steps_per_epoch=len(trainloader), epochs=EPOCHS)

    tr_loss, tr_acc, va_loss, va_acc = [], [], [], []
    best = 0

    for e in range(EPOCHS):
        model.train()
        t, correct, train_tot = 0, 0, 0
        for x, y, _idx in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            l = loss_fn(out, y)
            l.backward()
            opt.step()
            sched.step()
            t += l.item()
            correct += (out.argmax(1) == y).sum().item()
            train_tot += y.size(0)

        model.eval()
        v, c, tot = 0, 0, 0
        with torch.no_grad():
            for x, y, _idx in valloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                v += loss_fn(out,y).item()
                c += (out.argmax(1)==y).sum().item()
                tot += y.size(0)

        acc = 100*c/tot
        tr_loss.append(t/len(trainloader))
        tr_acc.append(100*correct/train_tot)
        va_loss.append(v/len(valloader))
        va_acc.append(acc)

        if acc > best:
            best = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR,f"{name}.pth"))

        print(f"{name} Epoch {e+1}: Train Acc {tr_acc[-1]:.2f}% | Val Acc {acc:.2f}%")

    # Loss curves
    plt.figure()
    plt.plot(tr_loss, label="Train Loss", color="tab:blue")
    plt.plot(va_loss, label="Val Loss", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR,f"{name.lower()}_dynamics.png"))
    plt.close()

    # Accuracy curves
    plt.figure()
    plt.plot(tr_acc, label="Train Acc", color="tab:blue")
    plt.plot(va_acc, label="Val Acc", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR,f"{name.lower()}_accuracy.png"))
    plt.close()

    with open(os.path.join(RESULTS_DIR, f"{name.lower()}_metrics.json"), "w") as f:
        json.dump(
            {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "max_lr": MAX_LR,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": va_loss,
                "val_acc": va_acc,
            },
            f,
            indent=2,
        )

    model.load_state_dict(torch.load(os.path.join(MODEL_DIR,f"{name}.pth")))
    model.eval()

    preds, labels = [], []
    failures = []

    with torch.no_grad():
        for x, y, idx in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = torch.softmax(out,1)
            conf, pred = prob.max(1)

            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())

            mask = pred != y
            for i in mask.nonzero(as_tuple=False).flatten():
                failures.append((int(idx[i].item()), x[i].cpu(), y[i].item(), pred[i].item(), float(conf[i].item())))

    cm = confusion_matrix(labels, preds)

    errors = cm.sum() - np.trace(cm)
    error_rate = 100 * errors / cm.sum()
    print(f"{name} Errors: {errors} / {cm.sum()} ({error_rate:.2f}%)")
    print(f"{name} Test Accuracy: {100-error_rate:.2f}%")

    plt.figure(figsize=(9,7))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10),classes,rotation=45)
    plt.yticks(range(10),classes)
    thresh = cm.max()/2
    for i in range(10):
        for j in range(10):
            plt.text(j,i,cm[i,j],
                     ha="center",va="center",
                     color="white" if cm[i,j]>thresh else "black",
                     fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{name.lower()}_confusion_matrix.png"), dpi=300)
    plt.close()

    failures.sort(key=lambda z: z[4], reverse=True)
    failures = failures[:12]

    with open(os.path.join(RESULTS_DIR, f"{name.lower()}_failures.json"), "w") as f:
        json.dump(
            [
                {
                    "test_index": int(ix),
                    "true": int(t),
                    "pred": int(p),
                    "confidence": float(c),
                }
                for (ix, _img, t, p, c) in failures
            ],
            f,
            indent=2,
        )

    fig, ax = plt.subplots(2,6, figsize=(16,6))
    ax = ax.flatten()

    for i,(ix,img,t,p,c) in enumerate(failures):
        img = img.numpy()
        img = img*np.array([0.2023,0.1994,0.2010]).reshape(3,1,1)
        img += np.array([0.4914,0.4822,0.4465]).reshape(3,1,1)
        img = np.transpose(img,(1,2,0))
        img = np.clip(img,0,1)
        ax[i].imshow(img)
        ax[i].set_title(f"idx:{ix} T:{classes[t]} P:{classes[p]}\n{c:.2f}", fontsize=8)
        ax[i].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,f"{name.lower()}_failures.png"), dpi=300)
    plt.close()

    return tr_acc, va_acc, cm

if __name__ == "__main__":
    base = FlexibleCNN(False)
    mod = FlexibleCNN(True)

    base_tr_acc, base_va_acc, base_cm = train_model(base,"Baseline")
    mod_tr_acc, mod_va_acc, mod_cm = train_model(mod,"Modified")

    plt.figure()
    plt.plot(base_va_acc, label="Baseline Val", color="tab:blue")
    plt.plot(mod_va_acc, label="Modified Val", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR,"comparison_curves.png"))
    plt.close()

    # Canonical 3 failure cases (highest-confidence baseline failures) to use in the report + Grad-CAM.
    with open(os.path.join(RESULTS_DIR, "baseline_failures.json"), "r") as f:
        baseline_failures = json.load(f)
    with open(os.path.join(RESULTS_DIR, "chosen_failure_cases.json"), "w") as f:
        json.dump(
            {
                "seed": SEED,
                "chosen_test_indices": [int(x["test_index"]) for x in baseline_failures[:3]],
            },
            f,
            indent=2,
        )

    base_class = base_cm.diagonal()/base_cm.sum(axis=1)*100
    mod_class = mod_cm.diagonal()/mod_cm.sum(axis=1)*100

    x = np.arange(10)
    plt.figure()
    plt.bar(x-0.2, base_class, 0.4, label="Baseline", color="tab:blue")
    plt.bar(x+0.2, mod_class, 0.4, label="Modified", color="tab:orange")
    plt.xticks(x, classes, rotation=45)
    plt.ylabel("Per-class Accuracy (%)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR,"per_class_accuracy.png"))
    plt.close()
