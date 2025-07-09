
# Single-camera Tracking on multicam

python3 tools/multicam_track.py video \
./test-site022 \
./experiments/yolox/yolox_tiny_8x8_300e_coco.py \
./experiments/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth \
--tp_weight ./experiments/tp/tp_best.pth \
--save_result ./test-site022 --save_vid True --track_buffer 150 --device cpu 
