import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.SSIM import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .InterpolateGMFSS import InterpolateGMFSSTorch
from .InterpolateRIFE import InterpolateRifeTorch,  InterpolateRIFEDRBA
from .InterpolateIFRNET import InterpolateIFRNetTorch


class InterpolateFactory:
    @staticmethod
    def build_interpolation_method(interpolate_model_path, backend, drba=False):
        ad = ArchDetect(interpolate_model_path)
        base_arch = ad.getArchBase()
        if base_arch == "rife":
            if drba:
                return InterpolateRIFEDRBA
            return InterpolateRifeTorch
        elif base_arch == "gmfss":
            return InterpolateGMFSSTorch
        elif base_arch == "ifrnet":
            return InterpolateIFRNetTorch  # IFRNet is a RIFE based architecture
