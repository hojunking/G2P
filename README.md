# G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Segmentation

Official PyTorch implementation of the paper **"G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Segmentation"**, accepted to **European Conference on Computer Vision (ECCV), 2026**.

[Hojun Song](https://hojunking.github.io/webpages/hojunsong)<sup>1,\*</sup>, [Chae-yeong Song](https://www.linkedin.com/in/chae-yeong-song-171816259)<sup>1,2,\*</sup>, [Jeong-hun Hong](https://invhun.github.io/en/about/)<sup>1</sup>, [Chaewon Moon](https://github.com/mchaewon)<sup>1</sup>, [Soo Ye Kim](https://sites.google.com/view/sooyekim)<sup>3</sup>, [Yiyi Liao](https://yiyiliao.github.io/)<sup>4</sup>, [Jaehyup Lee](https://sites.google.com/view/knuairlab/team/professor?authuser=0)<sup>1</sup>, and [Sang-hyo Park](https://sites.google.com/view/knuvi/s-park?authuser=0)<sup>1,†</sup>

<sup>1</sup> Kyungpook National University, South Korea
<sup>2</sup> Korea Electronics Technology Institute, South Korea
<sup>3</sup> Adobe Research, USA
<sup>4</sup> Zhejiang University, China

<sup>\*</sup> Equal contribution &nbsp;&nbsp; <sup>†</sup> Corresponding author

[![arXiv](https://img.shields.io/badge/arXiv-2601.03510-red)](https://arxiv.org/abs/2601.03510) [![Project Page](https://img.shields.io/badge/Project%20Page-G2P-green)](https://hojunking.github.io/webpages/G2P/)

---

## 📰 News

- **2026-06** Paper accepted to ECCV 2026.
- **2026-08** Code released.

## TODO

- [x] Release the main code (preparation and segmentation stages)
- [x] Release the ScanNet v2 configurations
- [ ] Release pre-trained checkpoints
- [ ] Release the pre-computed Stage 1 assets

## Installation

As in the paper: the segmentation model is trained on a single NVIDIA
RTX 3090, and the appearance encoder on an NVIDIA A6000.

```bash
git clone https://github.com/hojunking/G2P.git && cd G2P
```

**Docker** (recommended):

```bash
cd segmentation
docker build -t g2p:latest .
docker run -it --gpus '"device=0"' --shm-size 32G \
  -v $(pwd):/workdir -v /path/to/pointcept/data:/data:ro -w /workdir \
  g2p:latest /bin/bash
```

`--shm-size` matters -- the dataloader uses shared memory and the 64 MB default
is not enough. `TORCH_CUDA_ARCH_LIST` in the Dockerfile is pinned to `8.6`
(RTX 3090 / A6000); change it for a different GPU.

**Conda** -- a standard Pointcept environment; see
[Pointcept's installation guide](https://github.com/Pointcept/Pointcept#installation)
if `spconv` or `flash-attn` give trouble:

```bash
cd segmentation
conda env create -f environment.yml && conda activate g2p
cd libs/pointops && python setup.py install && cd ../..
```

The preparation stage is independent and needs only NumPy/SciPy:

```bash
cd preparation && pip install -r requirements.txt
```

## Data preparation

[ScanNet v2](http://www.scan-net.org) in Pointcept format (1201 train / 312 val
scenes, 20 classes) plus a 3D Gaussian Splatting reconstruction of the same
scenes -- we use [SceneSplat-7K][scenesplat], roughly 1.5M Gaussians per scene.
Setup: [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md).

```yaml
# preparation/configs/scannet.yaml
scene_root: /path/to/pointcept/data/scannet
gs_root:    /path/to/scenesplat7k/scannet
gs_ply_template: "{scene}/ckpts/point_cloud_30000.ply"
```

## Stage 1: Gaussian-to-Point preparation

Runs once, offline. Produces the two assets Stage 2 consumes. Gaussians are
needed **only here** -- inference is Gaussian-free.

```bash
cd preparation

# Gaussian-to-Point feature augmentation  (Sec. 3.3, Eq. 1-3)
python tools/extract_features.py \
    --config configs/scannet.yaml \
    --output-root ../segmentation/data/features/scannet_k20 \
    --k 20 --radius 0.06 --num-workers 24

# Scale-based boundary pseudo-labels      (Sec. 3.4, Eq. 4-5)
python tools/build_boundary.py \
    --config configs/scannet.yaml \
    --features-root ../segmentation/data/features/scannet_k20 \
    --output-root ../segmentation/data/boundary/scannet_eta07_rs004 \
    --method both --eta 0.7 --semantic-radius 0.04 --num-workers 8
```

Both steps skip scenes whose output already exists, so an interrupted run can
simply be restarted. Output format and hyperparameters:
[preparation/README.md](preparation/README.md).

## Stage 2: Training

```bash
cd segmentation
mkdir -p data pre_trained
ln -sfn /path/to/pointcept/data/scannet data/scannet
```

**Appearance encoder.** A Sonata encoder trained from scratch on
`(mu^p, c, alpha') in R^7`, i.e. the geometric normal replaced by the aggregated
Gaussian opacity. 400 epochs, batch size 1, on an A6000.

```bash
sh scripts/train.sh -g 1 -d sonata -n appearance-encoder \
   -c pretrain-appearance-encoder

cp exp/sonata/appearance-encoder/model/model_last.pth \
   pre_trained/appearance_encoder_scannet.pth
```

**Segmentation.** PT v3 + boundary-semantic block + appearance distillation.
800 epochs, batch size 4, on a single RTX 3090.

```bash
sh scripts/train.sh -g 1 -d scannet -n g2p -c g2p-scannet-v2
```

## Inference

13-view test-time augmentation. The backbone consumes `(coord, color, normal)`
only -- no Gaussian data is required.

```bash
sh scripts/test.sh -g 1 -d scannet -n g2p -w model_best
```

## Configurations

| Config | Paper |
|---|---|
| `configs/sonata/pretrain-appearance-encoder.py` | Sec. 3.5, appearance encoder |
| `configs/scannet/g2p-scannet-v2.py` | Full method, Tab. 1 |
| `configs/scannet/ablation-boundary-only.py` | Tab. 7, boundary guidance only |
| `configs/scannet/ablation-distill-only.py` | Tab. 7, distillation only |
| `configs/scannet/baseline-ptv3.py` | Tab. 1/7, PT v3 baseline |

Key hyperparameters (Sec. 4.1): `r_g = 0.06 m`, `k = 20`, `eta = 0.7`,
`r_s = 0.04 m`, `lambda_b = 0.9`, `lambda_d = 0.4`. Ablations and further
options: [docs/TRAINING.md](docs/TRAINING.md).

## Repository structure

```
preparation/    Stage 1 -- Gaussian-to-Point augmentation, boundary extraction
segmentation/   Stage 2 -- Pointcept fork: training and evaluation
docs/           Data setup and training details
```

## Citation

```bibtex
@article{song2025g2p,
  title   = {G2P: Gaussian-to-Point Attribute Alignment for Boundary-Aware 3D Segmentation},
  author  = {Song, Hojun and Song, Chae-yeong and Hong, Jeong-hun and Moon, Chaewon and
            Kim, Soo Ye and Liao, Yiyi and Lee, Jaehyup and Park, Sang-hyo},
  journal = {arXiv preprint arXiv:2601.03510},
  year    = {2025}
}
```

## Acknowledgements

`segmentation/` is a fork of [Pointcept](https://github.com/Pointcept/Pointcept),
trimmed to the ScanNet v2 path. The appearance encoder uses Pointcept's
implementation of [Sonata](https://github.com/facebookresearch/sonata). The
boundary-semantic block is our re-implementation of the head described in
[BFANet](https://github.com/weiguangzhao/BFANet). Gaussian reconstructions come
from [SceneSplat-7K][scenesplat].

## License

MIT, see [LICENSE](LICENSE). `segmentation/` is a fork of
[Pointcept](https://github.com/Pointcept/Pointcept) and keeps its own MIT
license at [segmentation/LICENSE](segmentation/LICENSE).

ScanNet v2 and SceneSplat-7K are not redistributed here; obtain them from their
sources and follow the terms there.

[scenesplat]: https://github.com/unique1i/SceneSplat
