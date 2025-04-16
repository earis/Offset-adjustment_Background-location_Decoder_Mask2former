# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CityscapesDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        # classes=('unlabeled', 'ego vehicle', 'rectification border', 'out of roi',
        #          'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking',
        #          'rail track', 'building', 'wall', 'fence', 'guard rail', 'bridge',
        #          'tunnel', 'pole', 'polegroup', 'traffic light', 'traffic sign',
        #          'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck',
        #          'bus', 'caravan', 'trailer', 'train', 'motorcycle', 'bicycle'),
        # classes=('background', 'Arytenoid', 'A_Carotid_L', 'A_Carotid_R',
        #          'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R',
        #          'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR',
        #          'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R','Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thryoid',
        #          'Glottis', 'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L',
        #          'Partoid_R', 'Pituitary', 'SpinalCord', 'train', 'motorcycle', 'bicycle'),#modified time:2024/09/15        
        # classes=('background', 'Arytenoid', 'A_Carotid_L', 'A_Carotid_R',
        #          'Bone_Mandible', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R',
        #          'Cricopharyngeus', 'Esophagus_S', 'Eye_AL', 'Eye_AR', 'Eye_PL', 'Eye_PR',
        #          'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R','Glnd_Submand_L', 'Glnd_Submand_R', 'Glnd_Thryoid',
        #          'Glottis', 'Larynx_SG', 'Lips', 'OpticChiasm', 'OpticNrv_L', 'OpticNrv_R', 'Parotid_L',
        #          'Partoid_R', 'Pituitary', 'SpinalCord','None'),#modified time:2024/09/15
        classes = ("background",
            "Brain",
    "BrainStem",
    "Chiasm",
    "TemporalLobe_L",
    "TemporalLobe_R",
    "TemporalLobe_Hippocampus_OverLap_L",
    "TemporalLobe_Hippocampus_OverLap_R",
    "Hippocampus_L",
    "Hippocampus_R",
    'Eye_L',
    'Eye_R',
    "Lens_L",
    "Lens_R",
    "OpticNerve_L",
    "OpticNerve_R",
    "MiddleEar_L",
    "MiddleEar_R",
    "IAC_L",
    "IAC_R",
    "MiddleEar_TympanicCavity_OverLap_L",
    "MiddleEar_TympanicCavity_OverLap_R",
    "TympanicCavity_L",
    "TympanicCavity_R",
    "MiddleEar_VestibulSemi_OverLap_L",
    "MiddleEar_VestibulSemi_OverLap_R",
    "VestibulSemi_L",
    "VestibulSemi_R",
    "Cochlea_L",
    "Cochlea_R",
    "MiddleEar_ETbone_OverLap_L",
    "MiddleEar_ETbone_OverLap_R",
    "ETbone_L",
    "ETbone_R",
    "Pituitary",
    "OralCavity",
    "Mandible_L",
    "Mandible_R",
    "Submandibular_L",
    "Submandibular_R",
    "Parotid_L",
    "Parotid_R",
    "Mastoid_L",
    "Mastoid_R",
    "TMjoint_L",
    "TMjoint_R",
    "SpinalCord",
    "Esophagus",
    "Larynx",
    "Larynx_Glottic",
    "Larynx_Supraglot",
    "Larynx_PharynxConst_OverLap",
    "PharynxConst",
    "Thyroid",
    "Trachea",
    "None"#: 55
    ),

        # palette=[[0, 0, 0], [128, 64, 128], [129, 3, 233], [102, 102, 255], [30, 21, 10], [111, 74, 0],
        #          [81, 0, 81], [0,109,102], [244, 35, 232], [250, 170, 160],
        #          [230, 150, 140], [70, 70, 70], [102, 102, 156], [190, 153, 153],
        #          [180, 165, 180], [150, 100, 100], [150, 120, 90], [153, 153, 153],
        #          [153, 153, 153],
        #          [250, 170, 30], [220, 220, 0], [107, 142, 35],
        #          [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
        #          [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 0, 90], [0, 0, 110],
        #          [0, 80, 100], [0, 0, 230], [119, 11, 32]])
        # palette=[[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #          [190, 153, 153], [153, 153, 153], [250, 170,
        #                                             30], [220, 220, 0],
        #          [107, 142, 35], [152, 251, 152], [70, 130, 180],
        #          [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        #          [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
        #          [255,182,193],[255,105,180],[255,20,147],[199,21,133],
        #          [123,104,238],[106,90,205],[72,61,139],[186,85,211],
        #          [148,0,211],[75,0,130],[138,43,226],[147,112,219],
        #          [30,144,255],[176,224,230],
        #          ])
        # palette=[[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
        #          [190, 153, 153], [153, 153, 153], [250, 170,
        #                                             30], [220, 220, 0],
        #          [107, 142, 35], [152, 251, 152], [70, 130, 180],
        #          [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
        #          [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
        #          [255,182,193],[255,105,180],[255,20,147],
        #          [123,104,238],[106,90,205],
        #          [148,0,211],[75,0,130],[138,43,226],[147,112,219],
        #          [30,144,255],[176,224,230],[0,0,13]
        #          ])
        palette=[[0,0,0],[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
                 [190, 153, 153], [153, 153, 153], [250, 170,
                                                    30], [220, 220, 0],
                 [107, 142, 35], [152, 251, 152], [70, 130, 180],
                 [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                 [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
                 [255,182,193],[255,105,180],[255,20,147],
                 [123,104,238],[106,90,205],
                 [148,0,211],[75,0,130],[138,43,226],[147,112,219],
                 [30,144,255],[176,224,230],[25,25,112],
                 [65,105,225],[100,149,237],[70,130,180],#1
                 [95,158,160],[47,79,79],[102,205,170],
                 [0,255,127],[46,139,87],[144,238,144],
                 [250,250,210],[255,255,0],[128,128,0],
                 [218,165,32],[	255,165,0],[255,69,0],
                 [128,0,0],[255,228,225],[188,143,143],
                 [255,0,0],[105,105,105],[184,134,11],
                [240,128,128],[72,61,139],
                 
                 
                 
                 
                 [0,0,13]
                 ])

    def __init__(self,
                #  img_suffix='_leftImg8bit.png',
                img_suffix = '.png',
                #  seg_map_suffix='_gtFine_labelTrainIds.png',
                seg_map_suffix = '.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix,reduce_zero_label=False, **kwargs)
