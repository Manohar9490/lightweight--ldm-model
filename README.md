# 32x32 CIFAR-10 VAE vs Diffusion (DDPM) Project


This project contains:

- A **Variational Autoencoder (VAE)** trained on CIFAR-10 (32x32)
- A **pretrained DDPM diffusion model** `google/ddpm-cifar10-32`

Both are evaluated with:

- **FID** (Fréchet Inception Distance)
- **SSIM** (Structural Similarity)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Sec_per_img** (seconds per image)


Things which this project will perform:

- Download CIFAR-10
- Train the VAE for a few epochs
- Evaluate the VAE on a subset of test images
  - Compute FID, SSIM, PSNR, Sec_per_img
  - Save original and reconstruction grids in `results/vae/`
- Run DDPM (`google/ddpm-cifar10-32`)
  - Generate 1000 fake images
  - Compute FID vs CIFAR-10 test set
  - Measure Sec_per_img
  - Save a grid of samples in `results/diffusion/`

```text
Model,FID,SSIM,PSNR,Sec_per_img
VAE,xx.xxxx,yy.yyyy,zz.zzzz,ss.ssss
DDPM,aa.aaaa,,,bb.bbbb
```

