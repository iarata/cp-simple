"""Loss aliases for the DETR-style baseline."""

from cps.models.detr import DETRCriterion, dice_loss, generalized_box_iou

__all__ = ["DETRCriterion", "dice_loss", "generalized_box_iou"]
