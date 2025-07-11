SAAAE: Self-Attention Anchored VAE for Noise-Controlled Image Reconstruction
🔍 Overview
SAAAE (Self-Attention Anchored Autoencoder) is a novel variant of the Variational Autoencoder (VAE) designed for enhanced image reconstruction. While traditional VAEs apply noise uniformly across latent dimensions—often distorting important features—SAAAE introduces self-attention-guided noise suppression, selectively anchoring critical regions in the latent space.

🧠 Key Idea
We introduce a self-attention mechanism to compute an importance map over input features, which is then used to control the magnitude of stochastic noise during reparameterization. Latent dimensions deemed “important” by attention are injected with less noise.

python
복사
편집
def reparameterize(self, mu, logvar, attention_map):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    positive_mask = 1 - attention_map  # high attention → less noise
    return mu + std * (eps * positive_mask)
This mechanism allows SAAAE to retain semantic integrity during reconstruction, especially under partial masking or corruption.

📦 Features
✅ Self-supervised attention-based noise modulation

✅ Selective KL divergence regularization

✅ Seamless drop-in replacement for standard VAEs

✅ Tested on image masking & reconstruction tasks

🖼 Dataset
Dataset: STL-10

Image resolution: 64×64

Preprocessing: Resize → ToTensor → Normalize([-1, 1])

🏃‍♂️ Usage
1. Install dependencies
bash
복사
편집
pip install torch torchvision einops torchmetrics matplotlib
2. Run training notebook
bash
복사
편집
jupyter notebook SAAAE_image_mask-linear.ipynb
📁 Project Structure
bash
복사
편집
SAAAE_image_mask-linear.ipynb      # Main training & evaluation notebook
models/
├── saaae.py                       # SAAAE model definition
data/
├── STL10/                         # STL10 dataset (auto-downloaded)
results/
├── recon/                         # Output image reconstructions
📊 Results
Metric	VAE	SAAAE
MSE ↓	58.98	34.11
KL+MSE ↓	26.82	20.01
Convergence	Slow	Faster

SAAAE shows superior fidelity and convergence speed compared to the baseline VAE.

🔬 Applications
Masked image inpainting

Feature-preserving compression

Denoising & corruption recovery

General-purpose VAE backbone

📘 Citation
bibtex
복사
편집
@misc{hong2025saaae,
  title={Self-Attention Anchored VAE for Noise-Controlled Image Reconstruction},
  author={Younggi Hong et al.},
  year={2025},
  note={arXiv preprint in preparation}
}
🧩 License
MIT License



VAE VS SAAAE
<img width="1423" height="707" alt="image" src="https://github.com/user-attachments/assets/4ef5d3b9-6d14-4ae2-b8b0-1c911a8f2780" />

Original

<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/2e6be6b4-f9cd-44e5-a415-3f4e6f86d1c8" />

Epoch_1

VAE
<img width="551" height="343" alt="image" src="https://github.com/user-attachments/assets/141632c3-7755-4a3f-b79b-4b5745177a3a" />

SAAAE

<img width="538" height="334" alt="image" src="https://github.com/user-attachments/assets/be664478-9823-4bf7-8c75-7ac54d215483" />

Epoch_101

VAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/516f65a0-f2ed-4aee-82f5-33a632d6dd9e" />

SAAAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/0bfe3fba-9b1b-4c6a-9e3f-9d34ed40cc9e" />

Epoch_201

VAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/fcf165cd-79d7-4889-803f-afa769810fae" />

SAAAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/7686cb11-5674-4c91-9471-ffdd517bcb76" />

Epoch_301

VAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/adadaa71-f4a5-4ab3-ac2f-a0ecded0215c" />

SAAAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/5f10b52f-7379-488d-8e76-9e8da950eb8f" />

Epoch_1401

VAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/766a56ea-11e7-4312-849b-82f3e430a01b" />

SAAAE
<img width="540" height="332" alt="image" src="https://github.com/user-attachments/assets/3bd8ad0b-6406-4e05-91fd-d34ae68606b4" />

Epoch_2401

VAE
<img width="540" height="336" alt="image" src="https://github.com/user-attachments/assets/c65e6147-3a46-4e7d-96d2-ee7242522f43" />

SAAAE
<img width="540" height="336" alt="image" src="https://github.com/user-attachments/assets/07493bc9-533f-4156-91ff-6ab224b040cd" />

