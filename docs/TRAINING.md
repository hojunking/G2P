# Training and evaluation

All commands run from `segmentation/`. Prerequisites:
the installation section of the top-level README and
[DATA_PREPARATION.md](DATA_PREPARATION.md).

## Overview

```
preparation stage (offline, Gaussians required)
    features.npy, boundary.npy
        |
        +--> 1. appearance encoder pre-training      (Sonata, 400 ep, bs 1)
        |         pre_trained/appearance_encoder_scannet.pth
        |
        +--> 2. G2P segmentation training            (PT v3 + B-S, 800 ep, bs 4)
                  |
                  +--> 3. evaluation with TTA        (Gaussian-free)
```

## 1. Appearance encoder (teacher)

Trains a Sonata encoder from scratch on `(mu^p, c, alpha') in R^7`, i.e. the
usual geometric normal replaced by the aggregated Gaussian opacity.

```bash
sh scripts/train.sh -g 1 -d sonata -n appearance-encoder \
   -c pretrain-appearance-encoder
```

400 epochs, batch size 1, lr 0.002, on an NVIDIA A6000. Copy the result where
the segmentation config expects it:

```bash
cp exp/sonata/appearance-encoder/model/model_last.pth \
   pre_trained/appearance_encoder_scannet.pth
```

The checkpoint stores weights under `student.backbone.*`; the loader strips that
prefix automatically.

> The segmentation config raises if the teacher checkpoint does not load, so
> this step cannot be skipped. To train without distillation, use
> `ablation-boundary-only.py`.

## 2. G2P segmentation

```bash
sh scripts/train.sh -g 1 -d scannet -n g2p -c g2p-scannet-v2
```

800 epochs, batch size 4, AdamW with `max_lr=[0.002, 0.0002]` under OneCycleLR,
on a single RTX 3090.

Confirm in `exp/scannet/g2p/train.log` that the teacher loaded:

```
Loaded 452/452 parameters under ('embedding.', 'enc.') (100.0%) from ... after stripping prefix 'student.backbone.'
```

Override any config field without editing the file:

```bash
EXTRA_OPTIONS="data.train.boundary_root=data/boundary/eta05 \
               data.val.boundary_root=data/boundary/eta05" \
sh scripts/train.sh -g 1 -d scannet -n g2p-eta05 -c g2p-scannet-v2
```

Resume an interrupted run with `-r true`.

## 3. Evaluation

Validation runs every `eval_epoch` during training. For the reported numbers use
the tester, which applies 13-view test-time augmentation (rotation x scale, plus
a flip):

```bash
sh scripts/test.sh -g 1 -d scannet -n g2p -w model_best
```

Results land in `exp/scannet/g2p/test.log` and per-scene predictions in
`exp/scannet/g2p/result/`.

Inference needs no Gaussian data: the backbone consumes `(coord, color, normal)`.
`features.npy` is still read by the dataset because the teacher branch shares the
pipeline, but the teacher is not evaluated.

## Ablations

| Config | Setting | Tab. 7 |
|---|---|---|
| `baseline-ptv3.py` | neither branch | 77.0 |
| `ablation-boundary-only.py` | boundary guidance only | 77.8 |
| `ablation-distill-only.py` | distillation only | 77.8 |
| `g2p-scannet-v2.py` | both | 78.4 |

Preparation-stage ablations do not need retraining of the teacher; regenerate
the assets with different flags and point the config at them:

| Paper | Command |
|---|---|
| Tab. 10, Euclidean matching | `extract_features.py --metric euclidean` |
| Tab. 10, `k` sweep | `extract_features.py --k {10,20,30}` |
| Fig. 11, `eta` sweep | `build_boundary.py --eta {0.1..0.9}` |
| Tab. 8, `B_sem` only | `build_boundary.py --method semantic` |
| Tab. 8, `B_scale` only | `build_boundary.py --method scale` |

Loss-weight sweeps (Tab. 15) are pure config overrides:

```bash
EXTRA_OPTIONS="model.criteria.0.boundary_loss_weight=0.8 hooks.7.distill_loss_weight=0.2" \
sh scripts/train.sh -g 1 -d scannet -n g2p-lb08-ld02 -c g2p-scannet-v2
```
