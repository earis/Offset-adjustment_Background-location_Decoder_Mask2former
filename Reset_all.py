import os
import shutil
shutil.move('/modifications/encoder_decoder.py','../mmsegmentation/mmseg/models/segmentors/encoder_decoder.py')
shutil.move('/modifications/msdeformattn_pixel_decoder_origin.py','../mmdet/models/layers/msdeform_attn_pixel_decoder.py')
shutil.move('/modifications/multi_scale_deform_attn_0.py','../mmcv/ops/multi_scale_deform_attn.py')
shutil.move('/modifications/mask2former_layers_progressive_fusion.py','../mmdet/models/layers/transformer/mask2former_layers.py')


