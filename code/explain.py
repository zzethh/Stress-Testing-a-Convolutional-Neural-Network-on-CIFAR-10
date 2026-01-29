import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from train import FlexibleCNN

BASE_DIR = "/scratch/m25csa032/assignment1"
RESULTS_DIR = os.path.join(BASE_DIR,"results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_gradcam_model(residual: bool, weights_path: str):
    model = FlexibleCNN(residual).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    state = {"acts": None, "grads": None}

    def f_hook(_m, _i, o):
        state["acts"] = o

    def b_hook(_m, _gi, go):
        state["grads"] = go[0]

    layer = model.layer3[-1].conv2
    layer.register_forward_hook(f_hook)
    layer.register_full_backward_hook(b_hook)
    return model, state

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

testset = torchvision.datasets.CIFAR10(BASE_DIR+"/data", train=False, download=True, transform=transform)

with open(os.path.join(RESULTS_DIR, "chosen_failure_cases.json"), "r") as f:
    chosen = json.load(f)["chosen_test_indices"]

base_model, base_state = build_gradcam_model(
    residual=False, weights_path=os.path.join(BASE_DIR, "models", "Baseline.pth")
)
mod_model, mod_state = build_gradcam_model(
    residual=True, weights_path=os.path.join(BASE_DIR, "models", "Modified.pth")
)

classes = ('Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck')

def gradcam_for_index(model, state, test_index: int):
    x, y = testset[test_index]
    x = x.unsqueeze(0).to(device)
    y = torch.tensor([y], device=device)

    out = model(x)
    prob = torch.softmax(out, 1)
    conf, pred = prob.max(1)

    model.zero_grad()
    out[0, pred.item()].backward()

    acts = state["acts"]
    grads = state["grads"]
    w = grads.mean(dim=(2, 3))[0]
    cam = (acts[0] * w[:, None, None]).sum(0)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam = cam.detach().cpu().numpy()

    img = x.detach().cpu().squeeze().numpy().transpose(1, 2, 0)
    img = img * np.array([0.2023, 0.1994, 0.2010]) + np.array([0.4914, 0.4822, 0.4465])
    img = np.clip(img, 0, 1)
    return img, cam, int(y.item()), int(pred.item()), float(conf.item())

fig, ax = plt.subplots(len(chosen), 6, figsize=(18, 3 * len(chosen)))
if len(chosen) == 1:
    ax = np.expand_dims(ax, 0)

for r, test_index in enumerate(chosen):
    b_img, b_cam, b_t, b_p, b_c = gradcam_for_index(base_model, base_state, test_index)
    m_img, m_cam, m_t, m_p, m_c = gradcam_for_index(mod_model, mod_state, test_index)

    # Baseline: image / cam / overlay
    ax[r, 0].imshow(b_img)
    ax[r, 0].set_title(f"Baseline img\nidx:{test_index}", fontsize=9)
    ax[r, 1].imshow(b_cam, cmap="jet")
    ax[r, 1].set_title(f"Baseline Grad-CAM\nT:{classes[b_t]} P:{classes[b_p]} {b_c:.2f}", fontsize=9)
    ax[r, 2].imshow(b_img)
    ax[r, 2].imshow(b_cam, alpha=0.5, cmap="jet")
    ax[r, 2].set_title("Baseline overlay", fontsize=9)

    # Modified: image / cam / overlay
    ax[r, 3].imshow(m_img)
    ax[r, 3].set_title("Modified img", fontsize=9)
    ax[r, 4].imshow(m_cam, cmap="jet")
    ax[r, 4].set_title(f"Modified Grad-CAM\nT:{classes[m_t]} P:{classes[m_p]} {m_c:.2f}", fontsize=9)
    ax[r, 5].imshow(m_img)
    ax[r, 5].imshow(m_cam, alpha=0.5, cmap="jet")
    ax[r, 5].set_title("Modified overlay", fontsize=9)

    for c in range(6):
        ax[r, c].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gradcam_comparison.png"), dpi=300)
