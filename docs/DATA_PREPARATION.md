# Data

G2P needs two things per scene: the point cloud, and a 3D Gaussian Splatting
reconstruction of the same scene.

## ScanNet v2

[ScanNet v2](http://www.scan-net.org) is an RGB-D dataset of indoor scans with
dense semantic annotations: 1201 training and 312 validation scenes over 20
classes. Request access from the official site, then run Pointcept's
preprocessing:

```bash
cd segmentation
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py \
    --dataset_root /path/to/ScanNet \
    --output_root  /path/to/pointcept/data/scannet
```

```
data/scannet/{train,val}/<scene>/
    coord.npy  color.npy  normal.npy  segment20.npy  segment200.npy  instance.npy
```

Only `coord`, `color`, `normal` and `segment20` are used here.

## Gaussians

Per-scene 3DGS reconstructions. We use [SceneSplat-7K][scenesplat] (~1.5M
Gaussians per scene), which provides ScanNet reconstructions directly; any 3DGS
implementation works as long as the PLY carries the standard vertex properties
(`x y z`, `nx ny nz`, `f_dc_*`, `f_rest_*`, `opacity`, `scale_0..2`,
`rot_0..3`). ScanNet ships depth, so COLMAP is not needed.

```
scenesplat7k/scannet/<scene>/ckpts/point_cloud_30000.ply
```

Set `gs_ply_template` in `preparation/configs/scannet.yaml` if your layout
differs.

## Configure and run Stage 1

```yaml
# preparation/configs/scannet.yaml
scene_root: /path/to/pointcept/data/scannet
gs_root:    /path/to/scenesplat7k/scannet
gs_ply_template: "{scene}/ckpts/point_cloud_30000.ply"
```

See [../preparation/README.md](../preparation/README.md) for the commands and
the output format. The result:

```
segmentation/data/
├── scannet/{train,val}/<scene>/*.npy
├── features/scannet_k20/{train,val}/<scene>/features.npy      float32 (N, 7)
└── boundary/scannet_eta07_rs004/{train,val}/<scene>/boundary.npy  int64 (N,)
```

[scenesplat]: https://github.com/unique1i/SceneSplat
