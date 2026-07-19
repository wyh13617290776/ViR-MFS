import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.common import RGB2YCrCb

class VIFSDataset(Dataset):
    """Visible-infrared fusion and segmentation dataset."""

    def __init__(
        self,
        mode,
        vi_dir,
        ir_dir,
        label_dir,
        resize_size=(640, 480),
        label_resize_interpolation="nearest",
    ):
        """Create a dataset instance.

        Args:
            mode: Dataset mode. ``train`` returns tensors for optimization;
                ``test`` also returns file names and chroma channels.
            vi_dir: Directory containing visible images.
            ir_dir: Directory containing infrared images.
            label_dir: Directory containing semantic labels.
            resize_size: Output size as ``(W, H)``. Use ``(None, None)`` to
                keep original image sizes.
            label_resize_interpolation: Interpolation mode used when resizing
                semantic labels. ``nearest`` preserves class ids. ``default``
                uses PIL's default resize behavior.

        Returns:
            None.
        """
        super().__init__()
        self.mode = mode
        self.vi_dir = vi_dir
        self.ir_dir = ir_dir
        self.label_dir = label_dir
        self.resize_size = resize_size
        self.label_resize_interpolation = label_resize_interpolation
        self.to_tensor = transforms.ToTensor()

        if not os.path.exists(self.vi_dir):
            raise FileNotFoundError(f"Directory does not exist: {self.vi_dir}")
        self.file_list = sorted(os.listdir(self.vi_dir))

    def _resize_label(self, label_image):
        """Resize a semantic label image.

        Args:
            label_image: PIL image containing integer semantic class ids.

        Returns:
            Resized PIL label image.
        """
        if self.label_resize_interpolation == "default":
            return label_image.resize(self.resize_size)
        if self.label_resize_interpolation == "nearest":
            return label_image.resize(self.resize_size, resample=Image.NEAREST)
        raise ValueError(
            "label_resize_interpolation must be either 'nearest' or 'default'"
        )

    def __len__(self):
        """Return the number of visible-image samples.

        Args:
            None.

        Returns:
            Dataset length.
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """Load one visible/infrared/label sample.

        Args:
            index: Sample index.

        Returns:
            For ``train``: ``(vi_y, ir, label)``. For ``test``:
            ``(vi_y, ir, label, file_name, cb, cr)``.
        """
        file_name = self.file_list[index]
        
        # Build paths for aligned visible, infrared, and label files.
        vi_path = os.path.join(self.vi_dir, file_name)
        ir_path = os.path.join(self.ir_dir, file_name)
        label_path = os.path.join(self.label_dir, file_name)

        if self.mode == 'train':
            vi_image = Image.open(vi_path).convert('L')
            ir_image = Image.open(ir_path).convert('L')
            label_image = Image.open(label_path)
            
            if self.resize_size[0] is not None:
                vi_image = vi_image.resize(self.resize_size)
                ir_image = ir_image.resize(self.resize_size)
                label_image = self._resize_label(label_image)
                
            vi_image = self.to_tensor(vi_image)
            ir_image = self.to_tensor(ir_image)
            
            # Labels must remain integer class IDs for cross-entropy.
            label_np = np.array(label_image, dtype=np.uint8)
            label_tensor = torch.from_numpy(label_np).long()
            
            return vi_image, ir_image, label_tensor

        elif self.mode == 'test':
            # Keep visible chroma channels during testing for RGB recovery.
            vi_image_color = Image.open(vi_path)
            ir_image = Image.open(ir_path).convert('L')
            label_image = Image.open(label_path)
            
            if self.resize_size[0] is not None:
                vi_image_color = vi_image_color.resize(self.resize_size)
                ir_image = ir_image.resize(self.resize_size)
                label_image = self._resize_label(label_image)

            label_np = np.array(label_image, dtype=np.uint8)
            label_tensor = torch.from_numpy(label_np).long()

            vi_image_tensor = self.to_tensor(vi_image_color)
            ir_image_tensor = self.to_tensor(ir_image)
            
            # Split visible RGB into luminance and chroma components.
            vi_y, cb, cr = RGB2YCrCb(vi_image_tensor)
            
            # File name and chroma tensors are required for output recovery.
            return vi_y, ir_image_tensor, label_tensor, file_name, cb, cr
        else:
            raise ValueError(f"Unsupported dataset mode: {self.mode}")
