# SR3-NAFNet-SSDA for 8x Rice Seed Super-Resolution

This repository contains the implementation of the proposed conditional diffusion model for 8x rice-seed microscopy super-resolution. The restoration backbone combines SR3-style diffusion, NAFNet blocks, and simple spatial decomposition attention (SSDA).

## Data

The paired dataset and its fixed train/validation/test split are available from Zenodo: [Rice Seed 8x Super-Resolution Dataset with Calibrated Smartphone Degradation](https://doi.org/10.5281/zenodo.23085080).

Download every `seed_dataset.tar.gz.part.*` file and `REASSEMBLE.txt`. Reconstruct and extract the archive:

```bash
cat seed_dataset.tar.gz.part.* > seed_dataset.tar.gz
tar -xzf seed_dataset.tar.gz
```

On Windows, use `copy /b seed_dataset.tar.gz.part.* seed_dataset.tar.gz` before extraction. The extracted dataset contains `train`, `val`, and `test` directories. For this code, place or link `train` and `val` below `data/`:

```text
data/
  train/
    hr_512/
    lr_64/
    sr_64_512/
  val/
    hr_512/
    lr_64/
    sr_64_512/
```

`hr_512` is the reference image, `lr_64` is the low-resolution image, and `sr_64_512` is the 8x bicubic-upsampled conditional input. The same structure is used for testing.

The calibrated smartphone-inspired degradation implementation used to create the LR images is included in the Zenodo data package as `generate_dataset.py`. It is not necessary to download model weights, intermediate outputs, or third-party baseline implementations to reproduce this repository.

## Installation

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

```bash
git clone https://github.com/win-star-yu/sr3_nafnet_ssda.git
cd sr3_nafnet_ssda
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA driver from the official PyTorch installation page if needed.

## Training

The reproducible example configuration is `config/rice_seed_x8.json`. Update only `datasets.train.dataroot`, `datasets.val.dataroot`, and `path.experiments_root` if your folders are elsewhere.

```bash
python sr.py -c config/rice_seed_x8.json -p train -gpu 0
```

The reference setting uses 64x64 to 512x512 8x reconstruction, Adam with learning rate `1e-5`, batch size `8`, 300,000 iterations, and a 2,000-step linear diffusion schedule (`1e-6` to `1e-2`). Checkpoints are written every 10,000 iterations.

## Evaluation

Set `path.resume_state` in the configuration to the checkpoint prefix without `_gen.pth`, set `phase` to `val`, then run:

```bash
python sr.py -c config/rice_seed_x8.json -p val -gpu 0
```

The evaluation path uses DDIM with 25 sampling steps and `eta=0`. Results and logs are written under the directory specified by `path.experiments_root`.

## License

Code is released under the [MIT License](LICENSE). The dataset has its own license and citation information on Zenodo.
