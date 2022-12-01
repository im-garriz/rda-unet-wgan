from imgaug import augmenters as iaa
import numpy as np
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


"""
Script de configuracion de dataset en Pytorch ##
    Iñaki Martinez: 18/02/2021
"""


class AugmentationPipeline:
    """
    Clase que continene un iaa.Sequential con todas las etapas de data
    augmentation
    """
    def __init__(self):
        """
        Constructor: genera la secuencia de augmentation
        """
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.0),
            iaa.Affine(rotate=(-0, 0), mode='symmetric'),
            #iaa.PerspectiveTransform(scale=(0.01, 0.15))
            #iaa.Sometimes(0.25, iaa.Dropout(p=(0, 0.1))),
        ])

    def __call__(self, img, gt):
        """
        Llamada al objeto: aplica la augmentation a las imagenes de entrada

        :param img: imagenes a las que aplicar augmentation
        :return: imagenes aumentadas
        """

        ground_truth = SegmentationMapsOnImage(gt == 255, shape=gt.shape)
        return self.aug(image=img, segmentation_maps=ground_truth)
