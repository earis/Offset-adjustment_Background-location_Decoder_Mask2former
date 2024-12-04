import shutil

shutil.move('/modifications/msdeformattn_pixel_decoder_mask2former_fusion.py','../mmdet/models/layers/msdeform_attn_pixel_decoder.py')
shutil.move('/modifications/mask2former_layers_mask2former_fusion.py','../mmdet/models/layers/transformer/mask2former_layers.py')
