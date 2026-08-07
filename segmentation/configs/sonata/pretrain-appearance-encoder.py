"""Appearance-encoder (teacher) pre-training on ScanNet v2 (paper Sec. 3.5).

Repurposes the Sonata self-supervised framework to learn appearance features
from the G2P-augmented point representation. Each point is encoded as

    p_i = (mu^p, c, alpha') in R^7

i.e. the usual geometric normal ``n`` is replaced by the aggregated Gaussian
opacity ``alpha'``, injecting view-consistent visibility cues in place of
geometry. Trained from scratch on the ScanNet v2 *train* split only.

Schedule follows Sonata: 400 epochs, batch size 1, lr 0.002 (Supplementary
Sec. A).

The resulting checkpoint is consumed by configs/scannet/g2p-scannet-v2.py as
``teacher_checkpoint``. Its weights live under ``student.backbone.*``;
``load_checkpoint`` strips that prefix and verifies the encoder loaded in full.

Run:
    sh scripts/train.sh -g 1 -d sonata -n appearance-encoder -c pretrain-appearance-encoder
"""

_base_ = ["../_base_/default_runtime.py"]

# ----------------------------------------------------------------------------
# Data locations
# ----------------------------------------------------------------------------
data_root = "data/scannet"
features_root = "data/features/scannet_k20"

dataset_type = "ScanNetDistillDataset"
feature_flag = ["opacity"]
sh_degree = 0

batch_size = 1
num_worker = 12
mix_prob = 0
clip_grad = 3.0
empty_cache = False
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = False
enable_wandb = False

# ----------------------------------------------------------------------------
# Model -- coord(3) + color(3) + opacity(1) = 7 input channels.
# ----------------------------------------------------------------------------
model = dict(
    type="Sonata-v1m1",
    backbone=dict(
        type="PT-v3m2",
        in_channels=7,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        enc_mode=True,
        mask_token=True,
    ),
    teacher_custom=dict(attn_drop=0.0, proj_drop=0.0, drop_path=0.0),
    head_in_channels=1088,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=2 / 8,
    roll_mask_loss_weight=2 / 8,
    unmask_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
)

# ----------------------------------------------------------------------------
# Schedule (layer-wise lr decay, as in Sonata)
# ----------------------------------------------------------------------------
epoch = 400
base_lr = 0.002
lr_decay = 0.9
base_wd = 0.04
final_wd = 0.2

_enc_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(_enc_depths) - sum(_enc_depths[:e]) - b - 1),
    )
    for e in range(len(_enc_depths))
    for b in range(_enc_depths[e])
]
del _enc_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ----------------------------------------------------------------------------
# Multi-view self-supervised pipeline; "features" carries the opacity channel.
# ----------------------------------------------------------------------------
view_keys = ("coord", "origin_coord", "color", "features")
global_feat_keys = ("global_coord", "global_color", "global_features")
local_feat_keys = ("local_coord", "local_color", "local_features")

_photometric = [
    dict(
        type="RandomColorJitter",
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.02,
        p=0.8,
    ),
    dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
    dict(type="NormalizeColor"),
]
_geometric = [
    dict(type="CenterShift", apply_z=True),
    dict(type="RandomScale", scale=[0.9, 1.1]),
    dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
    dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
    dict(type="RandomFlip", p=0.5),
    dict(type="RandomJitter", sigma=0.005, clip=0.02),
    dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
]

transform = [
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        view_keys=view_keys,
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=_photometric,
        global_transform=_geometric,
        local_transform=_geometric + _photometric,
        max_size=65536,
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.02}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_offset",
            "grid_size",
            "name",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=global_feat_keys,
        local_feat_keys=local_feat_keys,
    ),
]

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        test_mode=False,
        features_root=features_root,
        features_flag=feature_flag,
        sh_degree=sh_degree,
        transform=transform,
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=5),
]
