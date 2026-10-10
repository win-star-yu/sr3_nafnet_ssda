# SR3-NAFNet-SSDA 8x

This directory contains the runnable SR3-NAFNet-SSDA code and the rice-seed configuration. The single entry point for both training and inference is `sr.py`.

Install the listed dependencies in the 4090 environment:

```bash
conda activate sr3
pip install -r requirements.txt
```

The configuration expects `train`, `val`, and `test` under `/data/mycontrast/rice_dataset_phone_wide10x_color_aligned_x8_20260908_830recalibrated`, each with `hr_512`, `lr_64`, and `sr_64_512`.

Train:

```bash
CUDA_VISIBLE_DEVICES=0 python sr.py -c config/rice_phone_x8_830recal_condbicubic_scratch_stage1.json -p train -gpu 0
```

For inference, copy the configuration to a separate file, set `path.resume_state` to the checkpoint prefix without `_gen.pth`, and run:

```bash
CUDA_VISIBLE_DEVICES=0 python sr.py -c config/infer.json -p val -gpu 0
```

The SR3 configuration retains the supplied training settings and adds the required UNet block. SSDA uses spatial kernel size `k=7` in `model/sr3_modules/nafnet.py`.
