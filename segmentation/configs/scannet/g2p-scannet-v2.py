"""G2P on ScanNet v2 -- the full method (paper Tab. 1).

PT v3 backbone + BFANet boundary-semantic block, trained with
    L_total = L_sem + lambda_b * L_bou + lambda_d * L_distill
where the boundary head is supervised by the scale-derived pseudo-labels and
the appearance encoder distils GS-derived opacity cues into the backbone.

Inference is Gaussian-free: the backbone consumes (coord, color, normal) only.

Prerequisites, produced by ../../preparation:
    features_root/{train,val}/<scene>/features.npy
    boundary_root/{train,val}/<scene>/boundary.npy
and an appearance encoder trained with configs/sonata/pretrain-appearance-encoder.py.

Run:
    sh scripts/train.sh -g 1 -d scannet -n g2p-scannet-v2 -c g2p-scannet-v2
"""

_base_ = ["../_base_/default_runtime.py"]

# ----------------------------------------------------------------------------
# Data locations -- point these at your own preparation-stage output.
# ----------------------------------------------------------------------------
data_root = "data/scannet"
features_root = "data/features/scannet_k20"
boundary_root = "data/boundary/scannet_eta07_rs004"
teacher_checkpoint = "pre_trained/appearance_encoder_scannet.pth"

# Gaussian attributes exposed to the appearance-encoder teacher. The paper uses
# opacity only: the teacher is pre-trained on (mu^p, c, alpha') in R^7.
feature_flag = ["opacity"]
sh_degree = 0
# coord(3) + color(3) + opacity(1); must match the teacher's pre-training input.
teacher_in_channels = 7
feat_teacher_keys = ("coord", "color", "features")

# ----------------------------------------------------------------------------
# Training schedule (paper Sec. 4.1)
# ----------------------------------------------------------------------------
batch_size = 4
num_worker = 16
mix_prob = 0.8
empty_cache = False
enable_amp = True
enable_wandb = False

epoch = 800
eval_epoch = 100

# Sec. 4.1 states lr 0.003; every reported run used 0.002, which is what
# we ship here.
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# Loss weights (Tab. 15)
lambda_b = 0.9  # boundary guidance
lambda_d = 0.4  # appearance distillation, applied by DistillationSchedulerHook

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
model = dict(
    type="SegmentorBSDistill",
    num_classes=20,
    backbone_out_channels=64,
    # Student: PT v3 + B-S block, also returning the encoder bottleneck feature.
    backbone=dict(
        type="PT-v3m4",
        in_channels=6,  # color(3) + normal(3)
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
        # BFANet B-S block: 64-ch decoder features -> 128-d boundary-aware
        # features, 8 heads (Supplementary Sec. A).
        bsblock_cfg=dict(
            in_channels=64,
            semantic_out_channels=64,
            boundary_feature_channels=128,
            num_heads=8,
            dropout=[0.0, 0.0],
            num_semantic_classes=20,
        ),
    ),
    # Teacher: the appearance encoder, frozen. Only embedding + encoder run.
    teacher_backbone=dict(
        type="PT-v3m2",
        in_channels=teacher_in_channels,
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
        traceable=False,
        mask_token=False,
        # Encoder-only: the teacher's decoder is never run, and the appearance
        # encoder is pre-trained encoder-only, so it carries no decoder weights.
        enc_mode=True,
        freeze_encoder=False,
        checkpoint_path=teacher_checkpoint,
        # Sonata stores both branches; the EMA teacher is the stable one.
        checkpoint_source="teacher.backbone",
    ),
    # phi: maps the student bottleneck into the teacher's feature space (Eq. 6).
    mlp_bridge=dict(in_channels=512, out_channels=512, hidden_channels=256),
    criteria=[
        # L_sem + lambda_b * L_bou, applied to both B-S heads.
        # L_sem = CE + Lovasz (Eq. 7), L_bou = BCE + Dice (Eq. 8). The CE is
        # boundary-weighted by semantic_boundary_weight_factor, inherited from
        # BFANet.
        dict(
            type="BSLossWithLovasz",
            semantic_loss_weight=1.0,
            boundary_loss_weight=lambda_b,
            lovasz_loss_weight=1.0,
            ignore_index=-1,
            num_semantic_classes=20,
            semantic_boundary_weight_factor=9.0,
        ),
        # L_distill; the weight is injected by DistillationSchedulerHook below.
        dict(type="FeatureDistillationLoss", loss_weight=0.0, loss_type="CosineSimilarity"),
    ],
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
    dict(type="DistillationSchedulerHook", start_epoch=0, distill_loss_weight=lambda_d),
]

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
dataset_type = "ScanNetDatasetBoundary"
class_names = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture",
]

dataset_common = dict(
    type=dataset_type,
    data_root=data_root,
    features_root=features_root,
    boundary_root=boundary_root,
    features_flag=feature_flag,
    sh_degree=sh_degree,
)

collect_common = dict(
    type="Collect",
    keys=("coord", "grid_coord", "segment", "boundary"),
    feat_keys=("color", "normal"),
    feat_teacher_keys=feat_teacher_keys,
)

data = dict(
    num_classes=20,
    ignore_index=-1,
    names=class_names,
    train=dict(
        **dataset_common,
        split="train",
        test_mode=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            collect_common,
        ],
    ),
    val=dict(
        **dataset_common,
        split="val",
        test_mode=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            collect_common,
        ],
    ),
    test=dict(
        **dataset_common,
        split="val",
        test_mode=True,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "boundary"),
                    feat_keys=("color", "normal"),
                    feat_teacher_keys=feat_teacher_keys,
                ),
            ],
            # 13-view test-time augmentation (rotation x scale, plus a flip).
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[a], axis="z", center=[0, 0, 0], p=1)]
                for a in (0, 1 / 2, 1, 3 / 2)
            ] + [
                [
                    dict(type="RandomRotateTargetAngle", angle=[a], axis="z", center=[0, 0, 0], p=1),
                    dict(type="RandomScale", scale=[s, s]),
                ]
                for s in (0.95, 1.05)
                for a in (0, 1 / 2, 1, 3 / 2)
            ] + [
                [dict(type="RandomFlip", p=1)]
            ],
        ),
    ),
)
