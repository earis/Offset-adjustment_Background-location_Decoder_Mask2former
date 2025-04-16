# Copyright (c) OpenMMLab. All rights reserved.
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

# Generate distinct colors for visualization ZX 0831-2217
import colorsys
def generate_distinct_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        lightness = (50 + 10 * (i % 2)) / 100.0  # Alternate lightness for better distinction
        saturation = 0.9  # High saturation for vivid colors
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = tuple(int(c * 255) for c in rgb)
        colors.append(rgb)
    return colors


@DATASETS.register_module()
class HRFDataset(BaseSegDataset):
    """HRF dataset.

    In segmentation map annotation for HRF, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        # classes=('background', 'vessel','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','black','1','2'),
        classes=('background','Arytenoid', 'A_Carotid_L', 'A_Carotid_R',
                 'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R',
                 'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR',
                 'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R','Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thryoid',
                 'Glottis', 'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L',
                 'Partoid_R', 'Pituitary', 'SpinalCord',"unknown1"),#modified time:2024/09/15 , "unknown1",'unknown2','unknwon3','unknown4'
        palette=[[120,120,9], [6, 230, 230], [230, 6, 230], [230, 230, 6], [6, 6, 230], [6, 230, 6], [230, 6, 6], [230, 230, 230],\
                    [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
                    [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
                    [192, 192, 192], [128, 128, 128], [153, 50, 204], [255, 140, 0],
                    [75, 0, 130], [255, 20, 147], [34, 139, 34], [255, 215, 0],
                    [0, 191, 255], [139, 69, 19], [255, 69, 0], [46, 139, 87],
                    [123, 104, 238], [255, 105, 180], [0, 255, 127],[0, 255, 127]],)
    #     palette = [
    # [0, 0, 0], [180, 170, 160], [130, 250, 244], [121, 220, 241],
    # [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
    # [255, 0, 255], [0, 255, 255], [128, 0, 0], [0, 128, 0],
    # [0, 0, 128], [128, 128, 0], [128, 0, 128], [0, 128, 128],
    # [192, 192, 192], [128, 128, 128], [153, 50, 204], [255, 140, 0],
    # [75, 0, 130], [255, 20, 147], [34, 139, 34], [255, 215, 0],
    # [0, 191, 255], [139, 69, 19], [255, 69, 0], [46, 139, 87],
    # [123, 104, 238], [255, 105, 180], [0, 255, 127], [70, 130, 180]
    # ])

    def __init__(self,
                 img_suffix='.png',
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
