def calculate_dice(pred, target):
    smooth = 1.0  # Smoothing factor to avoid division by zero
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return ((2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)).item()


def calculate_precision(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fp = (pred_flat * (1 - target_flat)).sum()

    precision = (tp + smooth) / (tp + fp + smooth)
    return precision.item()


def calculate_recall(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    tp = (pred_flat * target_flat).sum()
    fn = ((1 - pred_flat) * target_flat).sum()

    recall = (tp + smooth) / (tp + fn + smooth)
    return recall.item()


def calculate_iou(pred, target):
    smooth = 1e-6
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = (pred_flat + target_flat).sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()