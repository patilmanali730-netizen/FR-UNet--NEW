    import numpy as np
    import torch
    import cv2
    from sklearn.metrics import roc_auc_score


    class AverageMeter(object):
        def __init__(self):
            self.initialized = False
            self.val = None
            self.avg = None
            self.sum = None
            self.count = None

        def initialize(self, val, weight):
            self.val = val
            self.avg = val
            self.sum = np.multiply(val, weight)
            self.count = weight
            self.initialized = True

        def update(self, val, weight=1):
            if not self.initialized:
                self.initialize(val, weight)
            else:
                self.add(val, weight)

        def add(self, val, weight):
            self.val = val
            self.sum = np.add(self.sum, np.multiply(val, weight))
            self.count = self.count + weight
            self.avg = self.sum / self.count

        @property
        def value(self):
            return np.round(self.val, 4)

        @property
        def average(self):
            return np.round(self.avg, 4)


    def safe_divide(numerator, denominator, eps=1e-8):
        """Avoid division by zero by adding a tiny epsilon."""
        return numerator / (denominator + eps)


    def get_metrics(predict, target, threshold=None, predict_b=None):
        predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
        if predict_b is not None:
            predict_b = predict_b.flatten()
        else:
            predict_b = np.where(predict >= threshold, 1, 0)
        if torch.is_tensor(target):
            target = target.cpu().detach().numpy().flatten()
        else:
            target = target.flatten()

        tp = (predict_b * target).sum()
        tn = ((1 - predict_b) * (1 - target)).sum()
        fp = ((1 - target) * predict_b).sum()
        fn = ((1 - predict_b) * target).sum()

        # Handle edge case for AUC: if target has only 1 class
        try:
            auc = roc_auc_score(target, predict)
        except ValueError:
            auc = 0.5  # neutral AUC if only one class present

        acc = safe_divide(tp + tn, tp + fp + fn + tn)
        pre = safe_divide(tp, tp + fp)
        sen = safe_divide(tp, tp + fn)
        spe = safe_divide(tn, tn + fp)
        iou = safe_divide(tp, tp + fp + fn)
        f1 = safe_divide(2 * pre * sen, pre + sen)

        return {
            "AUC": np.round(auc, 4),
            "F1": np.round(f1, 4),
            "Acc": np.round(acc, 4),
            "Sen": np.round(sen, 4),
            "Spe": np.round(spe, 4),
            "Pre": np.round(pre, 4),
            "IoU": np.round(iou, 4),
        }


    def count_connect_component(predict, target, threshold=None, connectivity=8):
        if threshold is not None:
            predict = torch.sigmoid(predict).cpu().detach().numpy()
            predict = np.where(predict >= threshold, 1, 0)
        if torch.is_tensor(target):
            target = target.cpu().detach().numpy()
        pre_n, _, _, _ = cv2.connectedComponentsWithStats(
            np.asarray(predict, dtype=np.uint8) * 255, connectivity=connectivity
        )
        gt_n, _, _, _ = cv2.connectedComponentsWithStats(
            np.asarray(target, dtype=np.uint8) * 255, connectivity=connectivity
        )
        return safe_divide(pre_n, gt_n)
