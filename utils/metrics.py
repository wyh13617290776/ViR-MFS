# codes/utils/metrics.py
import numpy as np

class SegmentationMetric:
    """Confusion-matrix based semantic segmentation metrics."""

    def __init__(self, num_classes):
        """Create a metric accumulator.

        Args:
            num_classes: Number of semantic classes.

        Returns:
            None.
        """
        self.num_classes = num_classes
        self.reset()

    def _generate_matrix(self, gt_image, pre_image):
        """Build one confusion matrix for a batch.

        Args:
            gt_image: Ground-truth label array.
            pre_image: Predicted label array.

        Returns:
            Confusion matrix with shape ``[num_classes, num_classes]``.
        """
        mask = (gt_image >= 0) & (gt_image < self.num_classes)
        label = self.num_classes * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_classes**2)
        confusion_matrix = count.reshape(self.num_classes, self.num_classes)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        """Accumulate predictions from one batch.

        Args:
            gt_image: Ground-truth label array.
            pre_image: Predicted label array.

        Returns:
            None.
        """
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def intersection_over_union(self):
        """Compute per-class IoU values.

        Args:
            None.

        Returns:
            NumPy array containing one IoU value per class.
        """
        intersection = np.diag(self.confusion_matrix)
        union = np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - intersection
        iou = intersection / np.maximum(1, union)
        return iou

    def mean_intersection_over_union(self):
        """Compute mean IoU over classes with ground-truth samples.

        Args:
            None.

        Returns:
            Mean IoU as a float.
        """
        iou = self.intersection_over_union()
        valid_mask = np.sum(self.confusion_matrix, axis=1) > 0
        if not np.any(valid_mask):
            return 0.0
        return np.mean(iou[valid_mask])

    def reset(self):
        """Reset the confusion matrix.

        Args:
            None.

        Returns:
            None.
        """
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
