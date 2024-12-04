**:wave:Hello !** **This is the hub for using our modified Mask2former added offset adjustment and feature fusion with Background and Location sensitive branch :rocket: for small organs segmentation with 2D CTs and better than using 3D volumes sometimes:heart: !.**

## **Start**

To start with, we require 

[Python]: https://www.python.org/

 installed with 3.10 and CUDA installed.

And the environment requires 

[mmsegmentation]: https://mmsegmentation.readthedocs.io/zh-cn/latest/get_started.html

 installed with mmcv, mmdet, and mmengine for its running support. You can download our respiratory and run the command in folder OABLS-Mask2former:

```bash
pip install requirements.txt
```

And mmsegmentation can be cloned directly from: https://github.com/open-mmlab/mmsegmentation.

Now copy the  OABLS-Mask2former in your mmsegmentation folder, and enter it and initiate the environment with running 

```bash
cd mmsegmentation/OABLS-Mask2former
py Basic_setups_BLS_decoder.py
```

### Configs

Please remember where your mmdet and mmcv is installed, usually under the folder your python installed say python3.10, then please enter site-packages folder inside it and copy the OABLS-Mask2former in. 

The folder tree in python3.10/site-packages seems like:

python3.10

​	|------------ ...

​	|------------site-packages

​				|------------mmcv

​				|------------mmengine

​				|------------OABLS-Mask2former

​				|------------numpy

​				|------------ ...

In fact in our OABLS-Mask2former/modifications folder there are just some modified version of original mmsegmentation support files, so we just need to replace them with the corresponding one. If you are familiar with mmsegmentation and mmdet, you can do it mannually or revise the files with different sets. If not, you can finish the process by running some scripts listed in OABLS-Mask2former. Here are some examples: 

Now after entering OABLS-Mask2former in site-packages, we can start by running the script:

```bash
py OA_1.py
```

to make Offset adjustment strategy 1 set in the environment. 

If we want to adapt progressive fusion policy in the meanwhile, we can run additionally:

```bash
py Progressive_fusion.py
```

All this set then we can just use mmsegmentation as usual.

Say running the config /mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py or with /mmsegmentation/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x10241.py.

And to use BLS_decoder branch with pixel enhancement , we just add a auxiliary_head in model dict:

```python
#in ~/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py
model = dict(
auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=0,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=34,
        align_corners=False,
        loss_decode=dict(
            type='DiceLoss_background', use_sigmoid=False, loss_weight=0.4)),
...
)
```

Nothing is special except for a DiceLoss_background loss function defined  by us.

You can try other models too, and we suppport GCNet, PSPNet, DANet and maybe other models too. Only things you need to change is the `in_channels` , `channels` ,`in_index`for the specific models, say `1024`,`2048`, `2` for PSPNet. 

## Give up

You can restore your original mmsegmentation by running 

```bash
py Reset_all.py 
```

in site-packages/OABLS-Mask2former AND mmsegmentation/OABLS-Mask2former.

Then the auxiliary head and other options are not influenced by our codes any more. (But the code is still revised in some places)

Further use of changing parameters in different Offset strategies and fusion ways and the pixel target in loss demands better understanding of codes in the mmsegmentation and our scripts. To change them, you can set values in the files in modifications provided by us.

Thanks for mmsegmentation for help!