# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DRIVEDataset(BaseSegDataset):
    """DRIVE dataset.

    In segmentation map annotation for DRIVE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '_manual1.png'.
    """
    METAINFO = dict(
        # classes=('background', 'vessel'),
        # palette=[[120, 120, 120], [6, 230, 230]])
        classes=('background', 'Arytenoid', 'A_Carotid_L', 'A_Carotid_R',
                 'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R',
                 'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR',
                 'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R','Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thryoid',
                 'Glottis', 'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L',
                 'Partoid_R', 'Pituitary', 'SpinalCord', 'train', 'motorcycle', 'bicycle'),#modified time:2024/09/15

        palette=[[0, 0, 0], [128, 64, 128], [0, 0, 0], [0, 0, 0], [0, 0, 0], [111, 74, 0],
                 [81, 0, 81], [0,0,0], [244, 35, 232], [250, 170, 160],
                 [230, 150, 140], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                 [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153],
                 [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35],
                 [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
                 [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110],
                 [0, 80, 100], [0, 0, 230], [119, 11, 32]])

    def __init__(self,
                 img_suffix='.png',
                #  seg_map_suffix='_manual1.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)
