import math


def bbox_status(box):
    """COCO xywh validation; a zero-area annotation cannot define spatial geometry."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return "malformed"
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return "malformed"
    if any(v < 0 for v in box):
        return "negative"
    return "degenerate" if box[2] == 0 or box[3] == 0 else "valid"
