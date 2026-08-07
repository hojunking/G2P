import torch
import torch.nn as nn
import torch_scatter

from pointcept.models.losses import (
    build_criteria,
    build_criteria_bs,
    build_criteria_bs_distil,
    build_criteria_distil,
)
from pointcept.models.utils.structure import Point
from pointcept.models.utils import (
    encode,
    load_sonata_backbone_checkpoint,
    offset2batch,
)
from pointcept.utils.logger import get_root_logger
from .builder import MODELS, build_model

def _build_frozen_sonata_teacher(teacher_backbone):
    teacher_backbone_cfg = teacher_backbone.copy()
    checkpoint_path = teacher_backbone_cfg.pop("checkpoint_path", None)
    checkpoint_source = teacher_backbone_cfg.pop(
        "checkpoint_source", "teacher.backbone"
    )
    checkpoint_strict = teacher_backbone_cfg.pop("checkpoint_strict", True)

    teacher = build_model(teacher_backbone_cfg)
    load_report = None
    if checkpoint_path:
        load_report = load_sonata_backbone_checkpoint(
            teacher,
            checkpoint_path,
            source=checkpoint_source,
            map_location="cpu",
            strict=checkpoint_strict,
        )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad = False
    return teacher, load_report


def _forward_sonata_teacher(teacher, input_dict):
    teacher_feat = input_dict["feat_teacher"]
    expected_channels = teacher.embedding.in_channels
    actual_channels = teacher_feat.shape[-1]
    if actual_channels != expected_channels:
        raise ValueError(
            "SONATA teacher input channel mismatch: "
            f"expected {expected_channels}, got {actual_channels}. "
            "Check feat_teacher_keys and teacher_backbone.in_channels."
        )

    teacher_input_dict = {
        "coord": input_dict["coord"],
        "grid_coord": input_dict["grid_coord"],
        "offset": input_dict["offset"],
        "feat": teacher_feat,
        "scene_name": input_dict.get("scene_name"),
    }
    point_teacher = Point(teacher_input_dict)
    point_teacher.serialization(
        order=teacher.order,
        shuffle_orders=teacher.shuffle_orders,
    )
    point_teacher.sparsify()
    point_teacher = teacher.embedding(point_teacher)
    point_teacher = teacher.enc(point_teacher)
    return {
        "feat": point_teacher.feat,
        "grid_coord": point_teacher.grid_coord,
        "batch": point_teacher.batch,
    }


def _pack_student_distillation_features(point_student, feature):
    required_keys = {"distill_grid_coord", "distill_batch"}
    missing_keys = required_keys.difference(point_student.keys())
    if missing_keys:
        raise RuntimeError(
            "Student backbone did not expose bottleneck correspondence metadata: "
            f"missing {sorted(missing_keys)}. "
            "Use a distillation backbone that records its encoder grid_coord and batch."
        )
    return {
        "feat": feature,
        "grid_coord": point_student.distill_grid_coord,
        "batch": point_student.distill_batch,
    }


def _validate_distillation_features(name, features):
    required_keys = {"feat", "grid_coord", "batch"}
    missing_keys = required_keys.difference(features.keys())
    if missing_keys:
        raise RuntimeError(
            f"{name} distillation features are missing {sorted(missing_keys)}."
        )

    feat = features["feat"]
    grid_coord = features["grid_coord"]
    batch = features["batch"]
    if feat.ndim != 2:
        raise RuntimeError(
            f"{name} feature must have shape [N, C], got {tuple(feat.shape)}."
        )
    if grid_coord.ndim != 2 or grid_coord.shape[1] != 3:
        raise RuntimeError(
            f"{name} grid_coord must have shape [N, 3], "
            f"got {tuple(grid_coord.shape)}."
        )
    if batch.ndim != 1:
        raise RuntimeError(
            f"{name} batch must have shape [N], got {tuple(batch.shape)}."
        )
    if not (feat.shape[0] == grid_coord.shape[0] == batch.shape[0]):
        raise RuntimeError(
            f"{name} distillation row count mismatch: "
            f"feat={feat.shape[0]}, grid_coord={grid_coord.shape[0]}, "
            f"batch={batch.shape[0]}."
        )
    if feat.shape[0] == 0:
        raise RuntimeError(f"{name} distillation features are empty.")
    if not (feat.device == grid_coord.device == batch.device):
        raise RuntimeError(
            f"{name} distillation tensors must be on one device: "
            f"feat={feat.device}, grid_coord={grid_coord.device}, "
            f"batch={batch.device}."
        )
    if torch.any(grid_coord < 0) or torch.any(grid_coord >= 2**16):
        raise RuntimeError(
            f"{name} grid_coord is outside the supported [0, 2^16) range."
        )
    if torch.any(batch < 0):
        raise RuntimeError(f"{name} batch contains a negative index.")


def _align_distillation_features(student, teacher):
    """Align bottleneck features by the exact (batch, grid_coord) voxel key."""
    _validate_distillation_features("Student", student)
    _validate_distillation_features("Teacher", teacher)

    if student["feat"].device != teacher["feat"].device:
        raise RuntimeError(
            "Student and teacher distillation features are on different devices: "
            f"student={student['feat'].device}, teacher={teacher['feat'].device}."
        )

    # encode() runs in inference mode; clone here so the resulting sort indices
    # are ordinary tensors and can safely index the gradient-carrying student feat.
    student_key = encode(
        student["grid_coord"], student["batch"], depth=16, order="z"
    ).clone()
    teacher_key = encode(
        teacher["grid_coord"], teacher["batch"], depth=16, order="z"
    ).clone()
    student_key, student_order = torch.sort(student_key)
    teacher_key, teacher_order = torch.sort(teacher_key)

    if student_key.numel() != teacher_key.numel():
        raise RuntimeError(
            "Student/teacher bottleneck voxel count mismatch: "
            f"student={student_key.numel()}, teacher={teacher_key.numel()}. "
            "Check grid sampling and encoder stride settings."
        )
    if torch.unique_consecutive(student_key).numel() != student_key.numel():
        raise RuntimeError("Student bottleneck contains duplicate spatial keys.")
    if torch.unique_consecutive(teacher_key).numel() != teacher_key.numel():
        raise RuntimeError("Teacher bottleneck contains duplicate spatial keys.")

    mismatch = student_key != teacher_key
    if torch.any(mismatch):
        mismatch_index = torch.nonzero(mismatch, as_tuple=False)[0, 0]
        raise RuntimeError(
            "Student/teacher bottleneck spatial keys do not match exactly: "
            f"first mismatch at canonical row {mismatch_index.item()}, "
            f"student_key={student_key[mismatch_index].item()}, "
            f"teacher_key={teacher_key[mismatch_index].item()}. "
            "Check grid sampling and encoder stride settings."
        )

    return (
        student["feat"][student_order],
        teacher["feat"][teacher_order],
        student_key.numel(),
    )


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

@MODELS.register_module()
class SegmentorBS(nn.Module):
    """Segmentor with the boundary-semantic block, without distillation."""
    def __init__(
        self,
        num_classes,
        backbone_out_channels,  # must match the B-S block's semantic_out_channels
        backbone=None,
        criteria=None,  # BoundarySemanticLoss config
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = nn.Identity() 
        self.backbone = build_model(backbone)  # PT-v3m3
        self.criteria = build_criteria_bs(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat  # final semantic feature
        else:
            feat = point # should not happen if backbone returns Point
        
        
        
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        # Training: losses are required.
        if self.training:
            losses_dict = self.criteria(
                initial_sem_logits=point.initial_semantic_logits,     
                initial_bou_logits=point.initial_boundary_logits,     
                final_sem_logits=point.final_semantic_logits,
                final_bou_logits=point.final_boundary_logits,
                gt_semantic_label=input_dict["segment"],
                gt_boundary_label=input_dict["boundary"]
            )
            return_dict.update(losses_dict)

        # Evaluation / test.
        else: # not self.training
            # Evaluation: ground-truth labels are available.
            if "segment" in input_dict.keys():
                # Same losses as training, without gradients.
                losses_dict = self.criteria(
                    initial_sem_logits=point.initial_semantic_logits,     
                    initial_bou_logits=point.initial_boundary_logits,     
                    final_sem_logits=point.final_semantic_logits,
                    final_bou_logits=point.final_boundary_logits,
                    gt_semantic_label=input_dict["segment"],
                    gt_boundary_label=input_dict["boundary"]
                )
                return_dict.update(losses_dict)

            # Predictions are always returned.
            return_dict["seg_logits"] = point.final_semantic_logits 
            return_dict["boundary_logits"] = point.final_boundary_logits 

        return return_dict


@MODELS.register_module()
class SegmentorDistill(nn.Module):
    def __init__(self,
                 backbone,          # student backbone config (PT v3)
                 teacher_backbone,  # appearance-encoder config (pre-trained PT v3)
                 mlp_bridge,        # phi, the distillation mapping MLP
                 criteria,          # segmentation + distillation loss configs
                 num_classes,
                 backbone_out_channels,
                 freeze_backbone=False):
        super().__init__()
        
        # 1. Student.
        self.backbone = build_model(backbone)
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)
        
        
        # 2. Teacher: the frozen appearance encoder.
        self.teacher_backbone, self.teacher_load_report = (
            _build_frozen_sonata_teacher(teacher_backbone)
        )
            
        # 3. phi: student bottleneck -> teacher feature space (Eq. 6).
        self.mlp_bridge = nn.Sequential(
            nn.Linear(mlp_bridge["in_channels"], mlp_bridge["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_bridge["hidden_channels"], mlp_bridge["out_channels"])
        )

        self.criteria = build_criteria_distil(criteria)
        self._distillation_correspondence_verified = False

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen teacher in eval mode; the parent model.train() would
        # otherwise flip it back.
        self.teacher_backbone.eval()
        return self

    def forward(self, input_dict):
        # 1. Student forward pass.
        point_student = Point(input_dict)
        point_student, student_feature_for_distill = self.backbone(point_student)
        student_distillation_features = (
            _pack_student_distillation_features(
                point_student, student_feature_for_distill
            )
            if self.training
            else None
        )
        
        if isinstance(point_student, Point):
            while "pooling_parent" in point_student.keys():
                assert "pooling_inverse" in point_student.keys()
                parent = point_student.pop("pooling_parent")
                inverse = point_student.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_student.feat[inverse]], dim=-1)
                point_student = parent
            feat_for_seg = point_student.feat
        else:
            feat_for_seg = point_student # should not happen

        seg_logits = self.seg_head(feat_for_seg)
        
        # 2. Losses.
        return_dict = {}
        if self.training:
            # Teacher and bridge are only needed while training.
            with torch.no_grad():
                teacher_distillation_features = _forward_sonata_teacher(
                    self.teacher_backbone, input_dict
                )
            (
                student_feature_for_distill,
                teacher_feature_for_distill,
                matched_points,
            ) = _align_distillation_features(
                student_distillation_features, teacher_distillation_features
            )
            if not self._distillation_correspondence_verified:
                get_root_logger().info(
                    "Distillation correspondence verified: "
                    "matched %d student/teacher bottleneck voxels.",
                    matched_points,
                )
                self._distillation_correspondence_verified = True
            student_feature_bridged = self.mlp_bridge(student_feature_for_distill)
            total_semantic_loss, distil_loss = self.criteria(
                seg_logits=seg_logits,
                student_feature_bridged=student_feature_bridged,
                teacher_feature=teacher_feature_for_distill,
                gt_semantic_label=input_dict["segment"]
            )
            
            return_dict["loss_seg"] = total_semantic_loss
            return_dict["loss_distill"] = distil_loss
            return_dict["loss"] = total_semantic_loss + distil_loss
        
        else:  # evaluation / test
            if "segment" in input_dict:
                total_semantic_loss, _ = self.criteria(
                    seg_logits=seg_logits,
                    student_feature_bridged=None,
                    teacher_feature=None,
                    gt_semantic_label=input_dict["segment"]
                )
                
                return_dict["loss"] = total_semantic_loss
            return_dict["seg_logits"] = seg_logits
            
        return return_dict



@MODELS.register_module()
class SegmentorBSDistill(nn.Module):
    def __init__(self,
                 backbone,          # student backbone config (PT v3 + B-S block)
                 teacher_backbone,  # appearance-encoder config (pre-trained PT v3)
                 mlp_bridge,        # phi, the distillation mapping MLP
                 criteria,          # all loss configs
                 **kwargs):
        super().__init__()
        
        # 1. Student: PT v3 + B-S block.
        self.backbone = build_model(backbone)
        
        # 2. Teacher: the frozen appearance encoder.
        self.teacher_backbone, self.teacher_load_report = (
            _build_frozen_sonata_teacher(teacher_backbone)
        )
            
        # 3. phi: student bottleneck -> teacher feature space (Eq. 6).
        self.mlp_bridge = nn.Sequential(
            nn.Linear(mlp_bridge["in_channels"], mlp_bridge["hidden_channels"]),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_bridge["hidden_channels"], mlp_bridge["out_channels"])
        )

        # 4. Loss container.
        self.criteria = build_criteria_bs_distil(criteria)
        self._distillation_correspondence_verified = False

    def train(self, mode=True):
        super().train(mode)
        # Keep the frozen teacher in eval mode; the parent model.train() would
        # otherwise flip it back.
        self.teacher_backbone.eval()
        return self

    def forward(self, input_dict, return_point=False):
        # 1. Student forward pass.
        # Returns the Point (carrying four logit sets) and the encoder feature.
        point_student, student_feature_for_distill = self.backbone(Point(input_dict))
        student_distillation_features = (
            _pack_student_distillation_features(
                point_student, student_feature_for_distill
            )
            if self.training
            else None
        )
        
        if isinstance(point_student, Point):
            while "pooling_parent" in point_student.keys():
                assert "pooling_inverse" in point_student.keys()
                parent = point_student.pop("pooling_parent")
                inverse = point_student.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_student.feat[inverse]], dim=-1)
                point_student = parent
            feat_for_seg = point_student.feat
        else:
            feat_for_seg = point_student # should not happen
        return_dict = dict()
        if return_point:
            return_dict["point"] = point_student
        
        if self.training:
            # Teacher and bridge are only needed while training.
            with torch.no_grad():
                teacher_distillation_features = _forward_sonata_teacher(
                    self.teacher_backbone, input_dict
                )
            (
                student_feature_for_distill,
                teacher_feature_for_distill,
                matched_points,
            ) = _align_distillation_features(
                student_distillation_features, teacher_distillation_features
            )
            if not self._distillation_correspondence_verified:
                get_root_logger().info(
                    "Distillation correspondence verified: "
                    "matched %d student/teacher bottleneck voxels.",
                    matched_points,
                )
                self._distillation_correspondence_verified = True
            student_feature_bridged = self.mlp_bridge(student_feature_for_distill)
            losses_dict = self.criteria(
                point_student=point_student,
                student_feature_bridged=student_feature_bridged,
                teacher_feature=teacher_feature_for_distill,
                input_dict=input_dict
            )
            return_dict.update(losses_dict)
        
        else:  # evaluation
            # Only the semantic + boundary losses are reported at eval time.
            if "segment" in input_dict.keys():
                losses_dict = self.criteria(
                    point_student=point_student,
                    student_feature_bridged=None,
                    teacher_feature=None,
                    input_dict=input_dict
                )
                return_dict = losses_dict
            
            return_dict["seg_logits"] = point_student.final_semantic_logits
            return_dict["boundary_logits"] = point_student.final_boundary_logits
        return return_dict
