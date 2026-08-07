from .builder import build_model
from .default import (
    DefaultSegmentor,
    DefaultClassifier,
    SegmentorBS,
    SegmentorDistill,
)
from .modules import PointModule, PointModel

# Segmentation backbone: PT v3, plus the G2P boundary-semantic / distillation
# variants (PT-v3m3, PT-v3m4).
from .point_transformer_v3 import *

# Prompt-driven normalization, imported by PT v3.
from .point_prompt_training import *

# Appearance encoder (teacher) pre-training.
from .sonata import *
