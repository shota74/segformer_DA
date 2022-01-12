import numpy as np

def _fast_hist(label_true, label_pred, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) + label_pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist

def scores(label_trues, label_preds, num_classes=19, synthia=False):
    synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    hist = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), num_classes)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    if synthia:
        mean_iu = np.nanmean(iu[synthia_set_16_to_13])
    else:
        valid = hist.sum(axis=1) > 0  # added
        mean_iu = np.nanmean(iu[valid])

    freq = hist.sum(axis=1) / hist.sum()
    cls_iu = dict(zip(range(num_classes), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
    }


def scores_16to13(label_trues, label_preds, num_classes=19):
    synthia_set_16 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
    synthia_set_13 = [0, 1, 2, 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]
    synthia_set_16_to_13 = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    #def Mean_Intersection_over_Union(self, out_16_13=False):
    MIoU = np.diag(self.confusion_matrix) / (
        np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
        np.diag(self.confusion_matrix))
    if self.synthia:
        MIoU_16 = np.nanmean(MIoU[:self.ignore_index])
        MIoU_13 = np.nanmean(MIoU[synthia_set_16_to_13])
        return MIoU_16, MIoU_13
    if out_16_13:
        MIoU_16 = np.nanmean(MIoU[synthia_set_16])
        MIoU_13 = np.nanmean(MIoU[synthia_set_13])
        return MIoU_16, MIoU_13
    MIoU = np.nanmean(MIoU[:self.ignore_index])

    return MIoU