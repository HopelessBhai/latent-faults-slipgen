# latent-faults-slipgen

<p align="center">
  <b>latent-faults-slipgen</b><br/>
  latent-space surrogate model for stochastic earthquake slip generation
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Repo-latent--faults--slipgen-111827?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Python-3.10%2B-111827?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Interface-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Domain-Seismology-0ea5e9?style=for-the-badge" />
</p>

---

## QUICK NAV

- [Project Snapshot](#project-snapshot)
- [System Architecture](#system-architecture)
- [Paper Figures](#paper-figures)
- [Fast Start](#fast-start)
- [Pipeline Runbook](#pipeline-runbook)
- [Artifacts Map](#artifacts-map)
- [Repository Layout](#repository-layout)
- [Reproducibility Notes](#reproducibility-notes)
- [Failure Debug Tree](#failure-debug-tree)
- [Citation](#citation)

---

## Project Snapshot

This repository implements a two-stage generative pipeline:

1. **Representation learning** with VQ-VAE on slip images.
2. **Conditional generation** via parameter-to-latent mapping + decoder reconstruction.

The code accompanies the manuscript:
`Latent Faults: A Latent-Space Surrogate Model for Stochastic Earthquake Slip Generation from Sparse Source Parameters`.

### Reported Manuscript Results (headline)

| Metric block | Reported result |
|---|---|
| Dataset size | 200 SRCMOD events |
| Grid standardization | `50 x 50` |
| 2D PSD correlation | `~0.93` |
| Radial PSD correlation | `~0.96` |

---

## System Architecture

<p align="center">
  <img src="report_pics/full_pipeline.png" alt="Full latent-fault generation pipeline" width="96%">
</p>

<p align="center">
  <img src="report_pics/VQVAE.png" alt="VQ-VAE architecture used for latent representation learning" width="96%">
</p>

---

## Paper Figures

<details>
<summary><b>Contrastive alignment concept (CLIP-style latent loss)</b></summary>
<br/>
<p align="center">
  <img src="report_pics/clip_loss.png" alt="Contrastive alignment loss visualization" width="78%">
</p>
</details>

<details>
<summary><b>Representative generated slip maps (qualitative examples)</b></summary>
<br/>
<p align="center">
  <img src="report_pics/sim_combined.png" alt="Combined qualitative slip map results" width="96%">
</p>
</details>

---

## Fast Start

```bash
# 1) environment
python -m venv .venv
source .venv/bin/activate

# 2) dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) training + inference
python train_vqvae.py
python train_mapper_decoder.py
python run_inference.py
```

Interactive app:

```bash
streamlit run interactive_slip_app.py
```

---

## Pipeline Runbook

<details>
<summary><b>STAGE 0 — Data Contracts (required inputs)</b></summary>

Required by training/inference scripts:

- `Dataset/text_vec.npy`
  - dict-style mapping: `event_key -> feature vector`
- `Dataset/filtered_images_train/`
  - training rupture images (PNG/JPG)
- `Dataset/filtered_images_test/`
  - test rupture images
- `assets/dz.json`
  - event key -> `Dz` value

Preprocessing and exploratory notebooks are intentionally not tracked in this clean release branch.

</details>

<details>
<summary><b>STAGE 1 — VQ-VAE training + latent extraction</b></summary>

Run:

```bash
python train_vqvae.py
```

Expected artifacts:
- `models/vqvae_finetuned.pth`
- `embeddings/image_latents.pkl`

</details>

<details>
<summary><b>STAGE 2 — Optional hyperparameter search</b></summary>

Run:

```bash
python tune_mapper.py
```

Expected artifact:
- `models/best_hyperparams.json`

</details>

<details>
<summary><b>STAGE 3 — Parameter -> latent mapper + decoder training</b></summary>

Run:

```bash
python train_mapper_decoder.py
```

Expected artifacts:
- `models/latent_model.pth`
- `models/decoder_model.pth`
- `scaler_x.pkl`
- `plots/` outputs (if plotting enabled)

</details>

<details>
<summary><b>STAGE 4 — Batch inference + metrics</b></summary>

Run:

```bash
python run_inference.py
```

Expected artifacts:
- `Dataset/predicted_images_LAT_LON/`
- `Dataset/slip_arrays_inference/`
- `error_metrics/`
- `test_metrics.json`

</details>

---

## Artifacts Map

| Category | Path(s) | Produced by |
|---|---|---|
| VQ-VAE weights | `models/vqvae_finetuned.pth` | `train_vqvae.py` |
| Image latent dictionary | `embeddings/image_latents.pkl` | `train_vqvae.py` |
| Mapper/decoder weights | `models/latent_model.pth`, `models/decoder_model.pth` | `train_mapper_decoder.py` |
| Input scaler | `scaler_x.pkl` | `train_mapper_decoder.py` / `latent_mapper.py` |
| Inference arrays | `Dataset/slip_arrays_inference/` | `run_inference.py` |
| Evaluation dumps | `error_metrics/`, `test_metrics.json` | `decoder.py` / `run_inference.py` |

---

## Repository Layout

```text
.
├── Dataset/ (local only, ignored from git)
├── assets/
│   ├── utils.py
│   └── dz.json
├── models/
│   ├── best_hyperparams.json
│   ├── fixed_hyperparams.json
│   └── fixed_hyperparams_manual.json
├── train_vqvae.py
├── latent_mapper.py
├── train_mapper_decoder.py
├── run_inference.py
├── interactive_slip_app.py
├── decoder.py
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Reproducibility Notes

- Input dimension is inferred from `Dataset/text_vec.npy`; keep feature format stable.
- `scaler_x.pkl` must match the exact model weights used during inference.
- Event-key naming consistency is mandatory across `text_vec.npy`, image filenames, and `assets/dz.json`.
- Slip conversion expects `assets/normalizing_slip_range.npy` (referenced in `assets/utils.py`).
- For strict paper reproduction, align script constants with manuscript hyperparameters (epochs, codebook size, split policy).

---

## Failure Debug Tree

- **Missing file error**
  - verify `Dataset/text_vec.npy`, `models/vqvae_finetuned.pth`, `scaler_x.pkl`, `assets/dz.json`
- **Very low matched samples**
  - check key consistency across `text_vec.npy`, image filenames, and `dz.json`
- **Slip values look wrong**
  - verify `Dz` lookup and `assets/normalizing_slip_range.npy`
- **Unstable train/val behavior**
  - check split policy + hyperparameter configuration alignment between runs

---

## Data Source

- SRCMOD finite-fault database: [https://www.seismo.ethz.ch/static/srcmod/Homepage.html](https://www.seismo.ethz.ch/static/srcmod/Homepage.html)

---

## Citation

Use the repository link as the citation target for now:

- [https://github.com/HopelessBhai/latent-faults-slipgen](https://github.com/HopelessBhai/latent-faults-slipgen)

---

## License

This project is licensed under the **MIT License**.

See `LICENSE` for full terms.
