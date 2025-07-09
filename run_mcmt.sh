
# Multi-camera Tracking (Tracklet Association)

python3 tools/multicam_association.py \
./test-site022/track_vis \
./experiments/mcmt/homography_list.pkl \
./experiments/mcmt/yolov7-w6-pose.onnx \
--device cpu \
--save_txt_path ./test-site022
