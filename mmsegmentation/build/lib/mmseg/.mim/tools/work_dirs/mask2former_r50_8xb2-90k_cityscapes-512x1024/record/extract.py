import re

# Define the regular expression pattern to match numbers inside '{{}}'
pattern = re.compile(r'tensor\((\d+\.\d+),')

# Open the file and read its contents
with open('/home/zhangxin/mmsegmentation/tools/work_dirs/mask2former_r50_8xb2-90k_cityscapes-512x1024/record/log2/sampling_offsets_0_x.txt',\
           'r') as file:
    lines = file.readlines()

# Extract numbers from each line
for line in lines:
    match = pattern.search(line)
    if match:
        number = match.group(1)
        print(number)