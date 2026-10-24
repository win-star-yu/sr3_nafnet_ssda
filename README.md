RiceSeed-Mic: Data, Code and Study Protocol

1. Public records and licences

Dataset: https://doi.org/10.5281/zenodo.23085080
Development repository: https://github.com/win-star-yu/sr3_nafnet_ssda

The dataset is licensed under CC BY 4.0. The code is licensed under MIT.
The Zenodo code record associated with the current GitHub release is the
archived code record to cite for this implementation.

Pretrained checkpoint: [Baidu Netdisk](https://pan.baidu.com/s/1EgG0UCHYpkNAleYd2gT2iQ?pwd=tq3j) (code: `tq3j`).

2. Dataset files and reconstruction

The dataset archive is distributed as 79 parts named
seed_dataset.tar.gz.part.000 through seed_dataset.tar.gz.part.078, together
with SHA256SUMS.txt, REASSEMBLE.txt, LICENSE.txt and this documentation.
Download all parts to one directory. On Linux/macOS:

    sha256sum -c SHA256SUMS.txt
    cat seed_dataset.tar.gz.part.[0-9][0-9][0-9] > seed_dataset.tar.gz
    tar -xzf seed_dataset.tar.gz

On Windows Command Prompt:

    copy /b seed_dataset.tar.gz.part.* seed_dataset.tar.gz
    tar -xzf seed_dataset.tar.gz

Dataset layout:

    seed_dataset/
        train/hr_512/    train/lr_64/    train/sr_64_512/
        val/hr_512/      val/lr_64/      val/sr_64_512/
        test/hr_512/     test/lr_64/     test/sr_64_512/

Matching filenames identify an HR/LR pair. HR patches are 512 x 512 pixels
and LR patches are 64 x 64 pixels. The train, validation and test subsets
contain 16,808, 2,129 and 2,145 pairs, respectively. Source images were
assigned to a subset before patch extraction.

The controlled degradation includes Gaussian low-pass filtering, continuous
colour-response mapping, constrained luminance sharpening, JPEG
encoding/decoding, eightfold area downsampling and residual intensity
perturbation. The main parameters are HR Gaussian sigma = 6 pixels;
sharpening Gaussian sigma = 3.2 pixels and strength = 0.5; JPEG quality =
95; and final-LR read-noise SD = 0.20 with a signal-dependent coefficient
of 0.25 on the 8-bit intensity scale.

3. Hair-length measurement tables

Hair_Length_Measurements_Per_Hair.csv contains 60 per-hair averages from
three observers for 20 physical seeds, each with three hairs. Lengths are
in micrometres. Hair_Length_Measurement_Summary.csv reports method-level
MAE, MRE, RMSE, bias, limits of agreement and confidence intervals.

4. Code files and commands

The single entry point is sr.py. The portable configuration is
config/rice_phone_x8.json. Before use, set datasets.train.dataroot and
datasets.val.dataroot to the selected dataset split directories. Each split
requires hr_512/, lr_64/ and sr_64_512/; the latter contains bicubic
upsampling of the matching LR images.

Install and train:

    python -m pip install -r requirements.txt
    python sr.py -c config/rice_phone_x8.json -p train -gpu 0

For evaluation, set path.resume_state to the checkpoint prefix without the
_gen.pth suffix, set datasets.val.dataroot to the desired validation or test
split, and run:

    python sr.py -c config/rice_phone_x8.json -p val -gpu 0

For the final reported test, set path.resume_state, keep
datasets.test.dataroot at the test split, and run:

    python sr.py -c config/rice_phone_x8.json -p test -gpu 0

The supplied test configuration requires exactly 2,145 image pairs and stops
with an error if the test split is incomplete. Metrics are computed in RGB
with evaluation.crop_border = 0, i.e. on the full image without an x8 border
crop. SSIM uses an 11 x 11 Gaussian window with sigma = 1.5; its internal
valid-window operation excludes only the five-pixel convolution margin.

SSDA is implemented in model/sr3_modules/nafnet.py and uses depthwise 1x7
and 7x1 directional convolutions. Diffusion and DDIM are implemented in
model/sr3_modules/diffusion.py. Image loading is implemented in
data/LRHR_dataset.py.

The supplied configuration reproduces the manuscript settings: a linear
2,000-step beta schedule from 0.0001 to 0.02, 340,000 training iterations,
batch size 2, and Adam learning rate 0.00002. Evaluation uses DDIM with 25
network evaluations and eta = 0. The initial state is formed by perturbing
the bicubic LR condition at round(110 x 0.28) = 31; for image index i, the
initial Gaussian noise uses the local seed 4407 + i. Consequently, the
default 25-step trajectory is 31, 30, 29, 27, 26, 25, 24, 22, 21, 20, 19,
17, 16, 15, 14, 12, 11, 10, 9, 7, 6, 5, 4, 2, 1, 0.

5. Evaluation conventions

For the synthetic test set, PSNR and SSIM use the reported image-region
conventions. Perceptual metrics and foreground-region measurements follow
the corresponding manuscript and supplementary-information definitions.
Use the supplied LR images for evaluation rather than regenerating a
different degradation.
