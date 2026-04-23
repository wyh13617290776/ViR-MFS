# codes/utils/metrics.py
import numpy as np

class SegmentationMetric:
    """
    语义分割评估指标工具类
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """添加一个 batch 的预测结果进行累加"""
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def intersection_over_union(self):
        """计算每一类的 IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / np.maximum(1, union)
        return iou

    def mean_intersection_over_union(self):
        """计算 mIoU（忽略没有样本的类别）"""
        iou = self.intersection_over_union()
        # 排除 union 为 0 的类别
        valid_mask = np.sum(self.confusion_matrix, axis=1) > 0
        if not np.any(valid_mask):
            return 0.0
        return np.mean(iou[valid_mask])

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))