import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import time
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image.fid import FrechetInceptionDistance

from models.vae32 import VAE32, vae_loss
from models.diffusion32 import load_ddpm_cifar32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("metrics", exist_ok=True)
os.makedirs("results/vae", exist_ok=True)
os.makedirs("results/diffusion", exist_ok=True)

def get_cifar10(train: bool):
    tfm = transforms.ToTensor()
    ds = datasets.CIFAR10(root="data", train=train, download=True, transform=tfm)
    return ds

def append_csv_row(path: str, row: str):
    header = "Model,FID,SSIM,PSNR,Sec_per_img\n"
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(header)
    with open(path, "a") as f:
        f.write(row + "\n")

def train_vae(epochs: int = 5, batch_size: int = 128, max_train_samples: int = 20000):
    print("\n=== Training VAE (32x32 CIFAR-10) ===")
    ds = get_cifar10(train=True)
    if max_train_samples and max_train_samples < len(ds):
        ds = Subset(ds, list(range(max_train_samples)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VAE32(latent_dim=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        for x, _ in loader:
            x = x.to(DEVICE)
            x_hat, mu, logvar = model(x)
            loss = vae_loss(x_hat, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        avg = total_loss / len(loader.dataset)
        print(f"[VAE] Epoch {ep+1}/{epochs} - Loss: {avg:.4f}")
    torch.save(model.state_dict(), "results/vae/vae32_cifar10.pth")
    return model

def evaluate_vae(model: VAE32, num_test_samples: int = 1000):
    print("\n=== Evaluating VAE ===")
    ds_test = get_cifar10(train=False)
    if num_test_samples and num_test_samples < len(ds_test):
        ds_test = Subset(ds_test, list(range(num_test_samples)))
    loader = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=2)

    model.eval()
    all_ssim, all_psnr = [], []
    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)

    t0 = time.time()
    first_x, first_xhat = None, None

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)

            if first_x is None:
                first_x = x.cpu()
                first_xhat = x_hat.cpu()

            # For SSIM/PSNR: convert to uint8 HWC
            x_np = (x.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype("uint8")
            xh_np = (x_hat.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype("uint8")
            for i in range(x_np.shape[0]):
                s_val = ssim(x_np[i], xh_np[i], data_range=255, channel_axis=2)
                p_val = psnr(x_np[i], xh_np[i], data_range=255)
                all_ssim.append(float(s_val))
                all_psnr.append(float(p_val))

            # For FID: input must be [N,C,H,W] in [0,1]
            fid.update(x, real=True)
            fid.update(x_hat, real=False)

    total_time = time.time() - t0
    num_images = len(ds_test)
    sec_per_img = total_time / float(num_images)

    mean_ssim = sum(all_ssim) / len(all_ssim)
    mean_psnr = sum(all_psnr) / len(all_psnr)
    fid_score = float(fid.compute().cpu().item())

    print(f"[VAE] FID={fid_score:.4f}, SSIM={mean_ssim:.4f}, PSNR={mean_psnr:.2f}, Sec/img={sec_per_img:.4f}")

    # Save grids
    save_image(make_grid(first_x[:32], nrow=8), "results/vae/vae_originals_grid.png")
    save_image(make_grid(first_xhat[:32], nrow=8), "results/vae/vae_reconstructions_grid.png")

    metrics = {
        "FID": fid_score,
        "SSIM": mean_ssim,
        "PSNR": mean_psnr,
        "Sec_per_img": sec_per_img,
    }
    return metrics

from torchvision.utils import save_image, make_grid

def run_ddpm(num_samples: int = 1000, batch_size: int = 64):
    print("\n=== Running DDPM (google/ddpm-cifar10-32) ===")
    pipe, model_id = load_ddpm_cifar32(DEVICE)

    # Load real CIFAR-10 test subset for FID
    ds_test = get_cifar10(train=False)
    if num_samples and num_samples < len(ds_test):
        ds_test = Subset(ds_test, list(range(num_samples)))
    loader_real = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)

    # Update FID with real images
    with torch.no_grad():
        for x, _ in loader_real:
            x = x.to(DEVICE)
            fid.update(x, real=True)

    # Generate fake images
    from torchvision.transforms.functional import to_tensor
    fake_tensors = []
    remaining = num_samples
    t0 = time.time()
    with torch.no_grad():
        while remaining > 0:
            cur = min(batch_size, remaining)
            out = pipe(batch_size=cur)
            imgs = out.images
            for im in imgs:
                fake_tensors.append(to_tensor(im))
            remaining -= cur
    total_time = time.time() - t0
    sec_per_img = total_time / float(num_samples)

    fake_stack = torch.stack(fake_tensors, dim=0).to(DEVICE)
    for i in range(0, fake_stack.size(0), batch_size):
        fid.update(fake_stack[i:i+batch_size], real=False)

    fid_score = float(fid.compute().cpu().item())
    print(f"[DDPM] FID={fid_score:.4f}, Sec/img={sec_per_img:.4f}")

    # Save a grid of first 32 images
    save_image(make_grid(fake_stack[:32].cpu(), nrow=8), "results/diffusion/ddpm32_samples_grid.png")

    metrics = {
        "FID": fid_score,
        "Sec_per_img": sec_per_img,
        "ModelID": model_id,
    }
    return metrics

def main():
    csv_path = os.path.join("metrics", "model_metrics.csv")

    # Train + evaluate VAE
    vae_model = train_vae(epochs=3, batch_size=128, max_train_samples=20000)
    vae_metrics = evaluate_vae(vae_model, num_test_samples=1000)
    append_csv_row(
        csv_path,
        "VAE,{FID:.4f},{SSIM:.4f},{PSNR:.4f},{Sec:.4f}".format(
            FID=vae_metrics["FID"],
            SSIM=vae_metrics["SSIM"],
            PSNR=vae_metrics["PSNR"],
            Sec=vae_metrics["Sec_per_img"],
        )
    )

    # Run DDPM
    ddpm_metrics = run_ddpm(num_samples=1000, batch_size=64)
    append_csv_row(
        csv_path,
        "DDPM,{FID:.4f},,,{Sec:.4f}".format(
            FID=ddpm_metrics["FID"],
            Sec=ddpm_metrics["Sec_per_img"],
        )
    )

    print("\nAll done. Metrics saved to", csv_path)
    print("VAE images in results/vae/, DDPM images in results/diffusion/.")

if __name__ == "__main__":
    main()
