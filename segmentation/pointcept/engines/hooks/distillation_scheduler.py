# In pointcept/engines/hooks/distillation_scheduler.py

from pointcept.engines.hooks import HookBase
from .builder import HOOKS
from pointcept.models.losses.misc import FeatureDistillationLoss

@HOOKS.register_module()
class DistillationSchedulerHook(HookBase):
    def __init__(self,
                 start_epoch,
                 distill_loss_weight):
        self.start_epoch = start_epoch
        self.distill_loss_weight = distill_loss_weight
        self.activated = False

    def before_epoch(self):
        # trainer.epoch is 0-indexed
        if not self.activated and self.trainer.epoch >= self.start_epoch:
            # Handle both multi-GPU (DDP) and single-GPU cases
            if hasattr(self.trainer.model, "module"):
                model = self.trainer.model.module  # Multi-GPU
            else:
                model = self.trainer.model  # Single-GPU

            # Support both semseg distill models (model.criteria)
            # and PG-v1m4 models (model.distill_criteria).
            if hasattr(model, "criteria") and hasattr(model.criteria, "criteria"):
                criteria_list = model.criteria.criteria
            elif hasattr(model, "distill_criteria") and hasattr(
                model.distill_criteria, "criteria"
            ):
                criteria_list = model.distill_criteria.criteria
            else:
                self.trainer.logger.warning(
                    "DistillationSchedulerHook: no compatible criteria container found."
                )
                return

            for criterion in criteria_list:
                if isinstance(criterion, FeatureDistillationLoss):
                    criterion.loss_weight = self.distill_loss_weight
                    self.trainer.logger.info(
                        f"Epoch {self.trainer.epoch}: Distillation loss weight ACTIVATED and set to {self.distill_loss_weight}."
                    )
                    self.activated = True
                    break
