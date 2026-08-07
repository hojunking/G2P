"""Ablation: boundary guidance without appearance distillation (Tab. 7 row 2).

Same data pipeline as the full method, but the model drops the teacher and the
distillation loss, keeping only

    L_total = L_sem + lambda_b * L_bou

``features.npy`` is still loaded because the dataset is shared; it is simply
unused by the model here.
"""

_base_ = ["g2p-scannet-v2.py"]

lambda_b = 0.9

model = dict(
    type="SegmentorBS",
    num_classes=20,
    backbone_out_channels=64,
    # PT-v3m3: B-S block without the encoder-feature output.
    backbone=dict(
        type="PT-v3m3",
        in_channels=6,
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
        bsblock_cfg=dict(
            in_channels=64,
            semantic_out_channels=64,
            boundary_feature_channels=128,
            num_heads=8,
            dropout=[0.0, 0.0],
            num_semantic_classes=20,
        ),
    ),
    criteria=[
        dict(
            type="BSLossWithLovasz",
            semantic_loss_weight=1.0,
            boundary_loss_weight=lambda_b,
            lovasz_loss_weight=1.0,
            ignore_index=-1,
            num_semantic_classes=20,
            semantic_boundary_weight_factor=9.0,
        ),
    ],
)

# No distillation: drop the scheduler hook.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
