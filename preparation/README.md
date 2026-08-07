# G2P preparation stage

Offline preprocessing that turns a point-cloud scene and its 3D Gaussian
Splatting reconstruction into the two assets the segmentation stage consumes:

| Asset | Paper | Written by |
|---|---|---|
| `features.npy` | Gaussian-to-Point feature augmentation, Sec. 3.3 / Eq. 1-3 / Alg. 1 | `tools/extract_features.py` |
| `boundary.npy` | Gaussian scale-based boundary extraction, Sec. 3.4 / Eq. 4-5 | `tools/build_boundary.py` |

Gaussians are needed **only here**. Inference is Gaussian-free.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Point the config at your data:

```yaml
# configs/scannet.yaml
scene_root: /path/to/pointcept/data/scannet     # {train,val}/<scene>/coord.npy ...
gs_root:    /path/to/scenesplat7k/scannet
gs_ply_template: "{scene}/ckpts/point_cloud_30000.ply"
```

`scene_root` expects Pointcept-format scenes; follow the Pointcept ScanNet
preprocessing script to produce them. `gs_root`
expects per-scene 3DGS PLY files -- we use [SceneSplat-7K][scenesplat], roughly
1.5M Gaussians per scene. Adjust `gs_ply_template` to match your layout.

Every path can also be overridden on the command line
(`--scene-root`, `--gs-root`, `--gs-ply-template`).

## Run

Stage 1 -- Gaussian-to-Point feature augmentation:

```bash
python tools/extract_features.py \
    --dataset scannet --config configs/scannet.yaml \
    --output-root data/features/scannet_k20 \
    --k 20 --radius 0.06 --num-workers 24
```

Stage 2 -- boundary pseudo-labels:

```bash
python tools/build_boundary.py \
    --dataset scannet --config configs/scannet.yaml \
    --features-root data/features/scannet_k20 \
    --output-root data/boundary/scannet_eta07_rs004 \
    --method both --eta 0.7 --semantic-radius 0.04 --num-workers 8
```

Both stages skip scenes whose output already exists, so an interrupted run can
simply be restarted (`--overwrite` to force recomputation). Per-scene failures
are collected and reported at the end, and the process exits non-zero unless
`--continue-on-error` is given.

## Paper hyperparameters

| Flag | Symbol | Default | Where |
|---|---|---|---|
| `--radius` | `r_g` | 0.06 m | Gaussian candidate search radius, Sec. 3.3 |
| `--k` | `k` | 20 | Mahalanobis neighbours, Eq. 1 (Tab. 10) |
| `--eta` | `eta` | 0.7 | Scale pruning ratio, Eq. 5 (Fig. 11) |
| `--semantic-radius` | `r_s` | 0.04 m | Semantic boundary radius, Sec. 3.4 (Tab. 8) |
| `--metric` | -- | `mahalanobis` | `euclidean` reproduces the Tab. 10 baseline |
| `--method` | -- | `both` | `B = B_scale u B_sem`; `scale` / `semantic` isolate each cue |
| `--fallback` | -- | `nearest` | Fallback when a point has too few Gaussians inside `r_g`, per Alg. 1 |

## Output format

```
<features_root>/{train,val}/<scene>/features.npy   float32 (N, 7)
<boundary_root>/{train,val}/<scene>/boundary.npy   int64   (N,) in {0, 1}
```

`N` always equals the scene's `coord.npy`. The 7 feature columns are

| Columns | Content |
|---|---|
| `0:3` | aggregated Gaussian scale `S'`, in metres (no normalisation) |
| `3:4` | aggregated opacity `alpha'`, in `[0, 1]` |
| `4:7` | aggregated rotation direction (unit vector) |

Only `0:3` (boundary extraction) and `3:4` (appearance-encoder input and
distillation) are consumed downstream; the rotation block is stored for
analysis only -- Fig. 8(d) shows rotation is stochastic and uncorrelated with
object boundaries, which is why it is not used as a cue. `--include-sh` extends
the array to `(N, 55)` with a 48-dim spherical-harmonics block, used only for
the Tab. 14 `SH'` ablation.

The paper writes the augmented point as `p_i = (mu^p, c, n, S', alpha') in R^13`.
That is the conceptual point representation; on disk we store only the
Gaussian-derived part, since `mu^p`, `c` and `n` already live in the Pointcept
scene directory.


[scenesplat]: https://github.com/unique1i/SceneSplat
