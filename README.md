# SAAAE: Self-Attention Anchored VAE for Noise-Controlled Image Reconstruction

## 🔍 Overview

**SAAAE (Self-Attention Anchored Autoencoder)** is a novel variant of the Variational Autoencoder (VAE) designed for enhanced image reconstruction. While traditional VAEs apply noise uniformly across latent dimensions—often distorting important features—SAAAE introduces self-attention-guided noise suppression, selectively anchoring critical regions in the latent space.

---

## 🧠 Key Idea

We introduce a self-attention mechanism to compute an importance map over input features, which is then used to control the magnitude of stochastic noise during reparameterization. Latent dimensions deemed “important” by attention are injected with less noise.

```python
def reparameterize(self, mu, logvar, attention_map):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    positive_mask = 1 - attention_map  # high attention → less noise
    return mu + std * (eps * positive_mask)
```

This mechanism enables SAAAE to retain semantic integrity, especially under masking or noise-corruption.

---

## 📦 Features

- ✅ Self-supervised attention-based noise modulation  
- ✅ KL divergence is selectively applied (decoupled from anchors)  
- ✅ Compatible with any VAE-style architecture  
- ✅ Outperforms vanilla VAE in image masking and recovery

---

## 🖼 Dataset

- **STL-10**
- Image size: `64 × 64`
- Transform: `Resize → ToTensor → Normalize([-1, 1])`

---

## 🏃‍♂️ Usage

### 1. Install dependencies

```bash
pip install torch torchvision einops torchmetrics matplotlib
```

### 2. Train the model

```bash
jupyter notebook SAAAE_image_mask-linear.ipynb
```

---

## 📁 Project Structure

```
SAAAE_image_mask-linear.ipynb      # Main notebook for training and visualization
models/
├── saaae.py                       # Model definition
data/
├── STL10/                         # Dataset (auto-downloaded)
results/
├── recon/                         # Reconstruction outputs
```

---

## 📊 Quantitative Results

| Metric       | VAE    | SAAAE  |
|--------------|--------|--------|
| MSE ↓        | 58.98  | **34.11** |
| KL + MSE ↓   | 26.82  | **20.01** |
| Convergence  | Slow   | **Faster** |

---

## 🖼️ Qualitative Comparison

### Original  
<img src="https://github.com/user-attachments/assets/2e6be6b4-f9cd-44e5-a415-3f4e6f86d1c8" width="400"/>

---

### Epoch 1

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/141632c3-7755-4a3f-b79b-4b5745177a3a" width="400"/> | <img src="https://github.com/user-attachments/assets/be664478-9823-4bf7-8c75-7ac54d215483" width="400"/> |

### Epoch 101

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/516f65a0-f2ed-4aee-82f5-33a632d6dd9e" width="400"/> | <img src="https://github.com/user-attachments/assets/0bfe3fba-9b1b-4c6a-9e3f-9d34ed40cc9e" width="400"/> |

### Epoch 201

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/fcf165cd-79d7-4889-803f-afa769810fae" width="400"/> | <img src="https://github.com/user-attachments/assets/7686cb11-5674-4c91-9471-ffdd517bcb76" width="400"/> |

### Epoch 301

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/adadaa71-f4a5-4ab3-ac2f-a0ecded0215c" width="400"/> | <img src="https://github.com/user-attachments/assets/5f10b52f-7379-488d-8e76-9e8da950eb8f" width="400"/> |

### Epoch 1401

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/766a56ea-11e7-4312-849b-82f3e430a01b" width="400"/> | <img src="https://github.com/user-attachments/assets/3bd8ad0b-6406-4e05-91fd-d34ae68606b4" width="400"/> |

### Epoch 2401

| VAE | SAAAE |
|-----|-------|
| <img src="https://github.com/user-attachments/assets/c65e6147-3a46-4e7d-96d2-ee7242522f43" width="400"/> | <img src="https://github.com/user-attachments/assets/07493bc9-533f-4156-91ff-6ab224b040cd" width="400"/> |

---

## 🔬 Applications

- Masked image inpainting  
- Feature-preserving latent compression  
- Robust denoising  
- Flexible backbone for conditional generative tasks

---

## 📘 Citation



---

## 🧩 License

MIT License
