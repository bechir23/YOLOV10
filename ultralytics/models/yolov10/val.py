from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    
    
    def postprocess(self, preds):
            """Apply Non-maximum suppression to prediction outputs."""
            return ops.non_max_suppression(
                preds,
                self.args.conf,
                self.args.iou,
                labels=self.lb,
                multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det,
            )
