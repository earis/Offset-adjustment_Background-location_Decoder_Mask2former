import os
import shutil
shutil.move('/modifications/encoder_decoder_1.py','../mmsegmentation/mmseg/models/segmentors/encoder_decoder.py')
shutil.move('/modifications/dice_loss.py','../mmsegmentation/mmseg/models/losses/dice_loss.py')