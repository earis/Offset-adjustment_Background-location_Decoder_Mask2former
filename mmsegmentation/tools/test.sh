config=$1
ckpt=$2
figure=$3
python3 test.py /home/zhangxin/mmsegmentation/configs/mask2former/$config /home/zhangxin/mmsegmentation/tools/work_dirs/mask2former_r50_8xb2-160k_ade20k-512x512/$ckpt --show-dir /home/zhangxin/mmsegmentation/tools/records/ade
# python3 /home/zhangxin/mmsegmentation/tools/analysis_tools/visualization_cam.py --out-file /home/zhangxin/mmsegmentation/tools/records/ade/prediction.png $figure /home/zhangxin/mmsegmentation/configs/mask2former/$config /home/zhangxin/mmsegmentation/tools/work_dirs/mask2former_r50_8xb2-160k_ade20k-512x512/$ckpt 