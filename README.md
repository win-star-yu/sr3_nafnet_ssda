# SR3-NAFNet-SSDA 8x

This directory contains  SR3-NAFNet-SSDA code and the rice-seed configuration. The single entry point for both training and inference is `sr.py`.

Place the dataset under `data/train` and `data/val`; each split must contain `hr_512`, `lr_64`, and `sr_64_512`.

Install the listed dependencies :

```bash
conda activate sr3
pip install -r requirements.txt
```

Train:

```bash
CUDA_VISIBLE_DEVICES=0 python sr.py -c config/rice_phone_x8.json -p train -gpu 0
```

For inference, copy the configuration to a separate file, set `path.resume_state` to the checkpoint prefix without `_gen.pth`, and run:

```bash
CUDA_VISIBLE_DEVICES=0 python sr.py -c config/infer.json -p val -gpu 0
```


