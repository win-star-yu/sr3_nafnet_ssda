# SR3-NAFNet-SSDA for Rice Seed 8x Super-Resolution

This repository contains the runnable implementation of SR3-NAFNet-SSDA and a portable rice-seed configuration.

The accompanying dataset and measurement tables are described in [README.txt](README.txt). Install dependencies and run:

```bash
python -m pip install -r requirements.txt
python sr.py -c config/rice_phone_x8.json -p train -gpu 0
```

For evaluation, set `path.resume_state` to a checkpoint prefix without the `_gen.pth` suffix, set `datasets.val.dataroot` to the desired split, then run:

```bash
python sr.py -c config/rice_phone_x8.json -p val -gpu 0
```

The configuration uses relative dataset locations: `data/train` and `data/val`. Each split must contain `hr_512`, `lr_64`, and `sr_64_512` with matching filenames. SSDA uses depthwise directional kernels of size 1x7 and 7x1.

The code is released under the MIT License.
