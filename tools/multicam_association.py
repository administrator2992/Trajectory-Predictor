import torch
import numpy as np
import cv2
import os
import os.path as osp
from PIL import Image
import pandas as pd
import time
import natsort
import scipy
import lap
import warnings
import argparse
import pickle
import operator

from tqdm import tqdm
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from scipy.sparse import csgraph
from collections import defaultdict, Counter

from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint

# Add ONNX Runtime import
import onnxruntime as ort

def make_parser():
    parser = argparse.ArgumentParser("MCMT Tracking Demo!")
    parser.add_argument('result_path', help='SCMT Tracking result path')
    parser.add_argument('homography_path', help='Homography file')
    parser.add_argument('pose_weight', help='Checkpoint file or ONNX model')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--save_out_path', default='./MCMT_result', help='Path to save text results')
    return parser

def load_SCMT_tracklet(gts, vid_width, vid_height):
    result = []
    for gt in gts:
        data_dict = {}
        for row in gt:
            row_values = row.strip().split(',')
            key = int(row_values[0])
            track_id = int(row_values[1])
            x = int(max(0, int(float(row_values[2]))))
            y = int(max(0, int(float(row_values[3]))))

            if int(float(row_values[4])) + x > vid_width:
                obj_w = vid_width - x - 1
            else:
                obj_w = int(float(row_values[4]))
            if int(float(row_values[5])) + y > vid_height:
                obj_h = vid_height - y - 1
            else:
                obj_h = int(float(row_values[5]))

            values = [x, y, x + obj_w, y + obj_h, float(row_values[6]), track_id]
            
            if key in data_dict:
                data_dict[key].append(values)
            else:
                data_dict[key] = [values]
        result.append(data_dict)
    return result

def check_dict_keys(arr, key_list, channel_id):
    result = []
    key_set = set(key_list)
    for elem in arr:
        if elem not in key_set:
            result.append([int(channel_id), int(elem)])
    return result

def make_square(img):
    height, width = img.shape[:2]
    square_size = max(width, height)
    new_img = np.zeros((square_size, square_size, 3), dtype=np.uint8)
    x_offset = (square_size - width) // 2
    y_offset = (square_size - height) // 2
    new_img[y_offset:y_offset+height, x_offset:x_offset+width] = img
    return new_img

def run_keypoint(frame_bbox, frame_img, onnx_session, input_name):
    thres = 0.5
    padding_size = 10
    square_size = 256
    H, W, _ = frame_img.shape
    
    patch_x1 = int(max(0, frame_bbox[0] - padding_size))
    patch_y1 = int(max(0, frame_bbox[1] - padding_size))
    patch_x2 = int(min(W - 1, frame_bbox[2] + padding_size))
    patch_y2 = int(min(H - 1, frame_bbox[3] + padding_size))

    patch = frame_img[patch_y1:patch_y2, patch_x1:patch_x2, :]
    if patch.size == 0:
        return False, None
        
    patch_squared = make_square(patch)
    resized_rgb = cv2.resize(cv2.cvtColor(patch_squared, cv2.COLOR_BGR2RGB), (square_size, square_size))
    
    # Preprocess image for ONNX
    input_img = resized_rgb.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))  # HWC to CHW
    input_img = np.expand_dims(input_img, axis=0)    # Add batch dimension
    
    # Run inference
    outputs = onnx_session.run(None, {input_name: input_img})
    output_tensor = torch.tensor(outputs[0])
    
    # Apply NMS - adjust nc/nkpt based on your model
    output_data = non_max_suppression_kpt(output_tensor, 0.25, 0.65, nc=1, nkpt=9, kpt_label=True)
    output = output_to_keypoint(output_data)

    state_bit = False
    avg_foot_coor = None
    max_area = 10
    
    for o in output:
        if len(o) < 33:  # Ensure we have enough elements
            continue
            
        obj_area = o[4] * o[5]
        # Adjust indices for ankle keypoints based on your model
        # Assuming new indices: left_ankle=25,26,27; right_ankle=28,29,30
        left_ankle_conf = o[27]  # Index for left ankle confidence
        right_ankle_conf = o[30]  # Index for right ankle confidence
        
        # Check if any leg joint is visible
        leg_joints_visible = any(conf > thres for conf in [o[18], o[21], o[24], o[27]])
        
        if (obj_area > max_area and 
            left_ankle_conf > thres and 
            right_ankle_conf > thres and 
            leg_joints_visible):
            max_area = obj_area
            state_bit = True
            patch_height, patch_width = patch.shape[:2]
            if patch_height > patch_width:
                # Get ankle coordinates - adjust indices based on your model
                left_ankle_x = o[25]  # X coordinate for left ankle
                left_ankle_y = o[26]  # Y coordinate for left ankle
                right_ankle_x = o[28]  # X coordinate for right ankle
                right_ankle_y = o[29]  # Y coordinate for right ankle
                
                # Calculate average foot position
                c_x = ((patch_x1 + patch_x2) / 2) - ((patch_y2 - patch_y1) / 2) + \
                      ((left_ankle_x + right_ankle_x) / 2) / 256 * (patch_y2 - patch_y1)
                bottom_y = patch_y1 + ((left_ankle_y + right_ankle_y) / 2) / 256 * (patch_y2 - patch_y1)
                avg_foot_coor = (int(c_x), int(bottom_y))
            else:
                state_bit = False
                
    return state_bit, avg_foot_coor

def extract_global_position(avg_foot_point, H):
    if avg_foot_point is None:
        return -1, -1
    c_x = avg_foot_point[0]
    bottom_y = avg_foot_point[1]
    point1 = np.array([[c_x, bottom_y, 1]])
    point2 = np.dot(H, point1.T)
    point2 /= point2[2]
    return int(point2[0]), int(point2[1])

def euclidean_distance_1d(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def euclidean_distance_2d(p1, p2):
    return int(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

def cosine_distance(x, y):
    return cosine(x, y)
    
def find_n_for_clustering(data):
    err_dist = 100
    tracklets = []
    distance_valid_bit = True
    for camera in data:
        camera_tracklets = {}
        for track_id, coords in camera.items():
            if len(coords) == 3:
                camera_tracklets[track_id] = coords[-1]
        tracklets.append(camera_tracklets)
    
    clusters = defaultdict(set)
    cluster_id = 1
    for i in range(len(tracklets)):
        camera_i = tracklets[i]
        for track_id_i, coord_i in camera_i.items():
            for j in range(len(tracklets)):
                camera_j = tracklets[j]
                for track_id_j, coord_j in camera_j.items():
                    if i == j and track_id_i == track_id_j:
                        continue
                    distance = euclidean_distance_2d(coord_i, coord_j)
                    if i == j and distance < err_dist*1.1:
                        distance_valid_bit = False
                    elif i != j and distance < err_dist:
                        clusters[cluster_id].add((i+1, track_id_i))
                        clusters[cluster_id].add((j+1, track_id_j))
                        cluster_id += 1
    # Merge clusters with common elements
    merged_clusters = []
    for key, value in clusters.items():
        for cluster in merged_clusters:
            if value.intersection(cluster):
                cluster.update(value)
                break
        else:
            merged_clusters.append(value)

    return len(merged_clusters), distance_valid_bit, merged_clusters

def run_spectral_clustering(position_features, k):
    # Calculate the Euclidean distance similarity matrix
    position_similarity_matrix = np.zeros((len(position_features), len(position_features)))
    for i in range(len(position_features)):
        for j in range(len(position_features)):
            if i != j:
                position_similarity_matrix[i, j] = euclidean_distance_1d(position_features[i], position_features[j])
                
    position_similarity_matrix = position_similarity_matrix / np.max(position_similarity_matrix)
    position_similarity_matrix = np.abs(position_similarity_matrix - 1)
    np.fill_diagonal(position_similarity_matrix, 0)
    
    pos_sim_matrix = position_similarity_matrix

    # Compute the Laplacian matrix
    laplacian = csgraph.laplacian(pos_sim_matrix, normed=True)

    # Compute the first k eigenvectors
    eigvals, eigvecs = np.linalg.eig(laplacian)
    eigvecs = eigvecs[:, np.argsort(eigvals)[:k]]

    # Normalize each row of the eigenvector matrix
    normalized_eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1)[:, np.newaxis]

    # Perform k-means clustering on the rows of the normalized matrix
    kmeans = KMeans(n_clusters=k, n_init=10).fit(normalized_eigvecs)
    
    # Cluster assignments
    clusters = kmeans.labels_

    return clusters

def offline_tracking(channel_loss, channel_id_mapping, channel_final_output):
    delete_key_list = []
    for idx, loss_obj in enumerate(channel_loss):
        for key in list(loss_obj.keys()):
            if key in channel_id_mapping[idx]:
                global_track_id = channel_id_mapping[idx][key]
                for row in loss_obj[key]:
                    row[1] = global_track_id
                    channel_final_output[idx].append(row)
                delete_key_list.append((idx, key))
    for key_obj in delete_key_list:
        del channel_loss[key_obj[0]][key_obj[1]]
    
    return channel_final_output, channel_loss

def most_frequent_last_called(lst):
    counter = Counter(lst)
    max_count = max(counter.values())
    most_frequent = [k for k, v in counter.items() if v == max_count]
    return most_frequent[-1] if most_frequent else None

def save_txt_result(channel_final_output, tracking_result_path, channel_list):
    for idx, channel_result in enumerate(channel_final_output):
        if not channel_result:
            continue
        sorted_result = sorted(channel_result, key=lambda x: x[2])
        save_dir = osp.join(tracking_result_path, channel_list[idx])
        os.makedirs(save_dir, exist_ok=True)
        
        with open(osp.join(save_dir, 'sorted_associated.txt'), 'w') as f:
            for row in sorted_result:
                row_str = ','.join(str(int(elem)) for elem in row[:3]) + ',' + ','.join(str(int(elem)) for elem in row[3:7]) + ',' + ','.join(str(int(elem)) for elem in row[7:]) + '\n'
                f.write(row_str)

def main(args):
    # Load homography matrix
    with open(args.homography_path, 'rb') as file:
        H_list = pickle.load(file)
    
    # Initialize ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.device.startswith('cuda') else ['CPUExecutionProvider']
    onnx_session = ort.InferenceSession(args.pose_weight, providers=providers)
    input_name = onnx_session.get_inputs()[0].name
    
    tracking_result_path = osp.join(args.result_path)
    channel_list = [file for file in os.listdir(tracking_result_path) 
                   if not file.startswith(".") and osp.isdir(osp.join(tracking_result_path, file))]
    channel_list = natsort.natsorted(channel_list)

    caps = []
    gts = []
    for channel in channel_list:
        video_path = osp.join(tracking_result_path, channel, 'inference_vid.mp4')
        label_path = osp.join(tracking_result_path, channel, 'label.txt')
        cap = cv2.VideoCapture(video_path)
        caps.append(cap)
        with open(label_path) as label_file:
            gt = label_file.readlines()
        gts.append(gt)
        
    vid_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_length = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    channel_track_result = load_SCMT_tracklet(gts, vid_width, vid_height)
    
    # ======================================================================
    # VIDEO WRITER INITIALIZATION (ADDED HERE)
    # ======================================================================
    video_writers = []
    os.makedirs(args.save_out_path, exist_ok=True)
    for idx, channel in enumerate(channel_list):
        save_dir = osp.join(args.save_out_path, channel_list[idx])
        output_video_path = osp.join(save_dir, f'{channel}_output.mp4')
        fps = caps[idx].get(cv2.CAP_PROP_FPS)
        width = int(caps[idx].get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(caps[idx].get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        video_writers.append(video_writer)
    # ======================================================================
    
    # Initialize tracking variables
    channel_id_mapping = [{} for _ in range(len(caps))]
    channel_id_position = [{} for _ in range(len(caps))]
    channel_loss = [{} for _ in range(len(caps))]
    channel_final_output = [[] for _ in range(len(caps))]
    new_id = 1
    
    # Processing loop
    t = tqdm(total=vid_length)
    for frame_count in range(vid_length):
        new_id_exist = []
        for idx, cap in enumerate(caps):
            ret, frame_img = cap.read()
            if not ret:
                continue
            
            # ======================================================================
            # VISUALIZATION PREPARATION (ADDED HERE)
            # ======================================================================
            vis_img = frame_img.copy()
            # ======================================================================
                
            defined_track_id = []
            if frame_count in channel_track_result[idx]:
                track_bbox = channel_track_result[idx][frame_count]
                for frame_bbox in track_bbox:
                    state_bit, avg_foot_point = run_keypoint(frame_bbox, frame_img, onnx_session, input_name)
                    if state_bit:
                        track_id = frame_bbox[-1]
                        defined_track_id.append(track_id)
                        projected_x, projected_y = extract_global_position(avg_foot_point, H_list[idx])
                        if track_id not in channel_id_position[idx]:
                            channel_id_position[idx][track_id] = [(projected_x, projected_y)]
                        elif len(channel_id_position[idx][track_id]) == 3:
                            channel_id_position[idx][track_id].pop(0)
                            channel_id_position[idx][track_id].append((projected_x, projected_y))
                        else:
                            channel_id_position[idx][track_id].append((projected_x, projected_y))
            
                # Cleanup position history
                delete_position_list = []
                for key in list(channel_id_position[idx].keys()):
                    if key not in defined_track_id:
                        delete_position_list.append((idx, key))
                for obj in delete_position_list:
                    del channel_id_position[obj[0]][obj[1]]

                # Check for new IDs
                channel_track_id = np.array([bbox[-1] for bbox in track_bbox])
                key_list = list(channel_id_mapping[idx].keys())
                new_id_exist += check_dict_keys(channel_track_id, key_list, idx)
            
            # ======================================================================
            # VISUALIZATION DRAWING (ADDED HERE)
            # ======================================================================
            if frame_count in channel_track_result[idx]:
                for frame_bbox in track_bbox:
                    x1, y1, x2, y2 = map(int, frame_bbox[:4])
                    track_id = int(frame_bbox[5])
                    global_id = channel_id_mapping[idx].get(track_id, -1)
                    
                    # Draw bounding box
                    color = (0, 255, 0)  # Green
                    cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw ID information
                    id_text = f"L:{track_id} G:{global_id}"
                    cv2.putText(vis_img, id_text, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Write frame to video
            video_writers[idx].write(vis_img)
            # ======================================================================
            
        # Association and ID management
        if new_id_exist:
            keypoint_valid_bit = False
            for new_id_obj in new_id_exist:
                if new_id_obj[1] in channel_id_position[new_id_obj[0]] and len(channel_id_position[new_id_obj[0]][new_id_obj[1]]) == 3:
                    keypoint_valid_bit = True

            if keypoint_valid_bit:
                N, distance_valid_bit, merged_clusters = find_n_for_clustering(channel_id_position)
    
                if distance_valid_bit:
                    matched_id_exist = []
                    for idx, tracklets_dic in enumerate(channel_id_position):
                        for key in tracklets_dic.keys():
                            if len(tracklets_dic[key]) == 3:
                                matched_id_exist.append([idx, key])
                    
                    new_matched_id_exist = [tup for tup in new_id_exist if tup in matched_id_exist]
                    
                    # Update channel_id_mapping using global_id 
                    for id_obj in new_matched_id_exist:
                        find_id_obj = (id_obj[0]+1, id_obj[1])
                        other_tuples = []
                        for group in merged_clusters:
                            if find_id_obj in group:
                                for tup in group:
                                    if tup != find_id_obj:
                                        other_tuples.append((tup[0]-1, tup[1]))
                                break
                        
                        target_id_list = []
                        for tup in other_tuples:
                            if tup[1] in channel_id_mapping[tup[0]]:
                                target_id_list.append(channel_id_mapping[tup[0]][tup[1]])
                    
                        if target_id_list:
                            target_id = most_frequent_last_called(target_id_list)
                            
                            if not all(x == target_id_list[0] for x in target_id_list):
                                dup_tuples = []
                                target_id_list_final = []
                                for idx, mapping_dic in enumerate(channel_id_mapping):
                                    for key, value in mapping_dic.items():
                                        if value in target_id_list and key in channel_id_position[idx]:
                                            dup_tuples.append((idx, key))
                                            target_id_list_final.append(value)
                                
                                dup_position = []
                                for tup in dup_tuples:
                                    dup_position.append(channel_id_position[tup[0]][tup[1]][-1])
                                dup_position.append(channel_id_position[id_obj[0]][id_obj[1]][-1])
                                
                                cluster_k = len(set(target_id_list_final))
                                cluster_result = run_spectral_clustering(np.array(dup_position), cluster_k)
                                
                                cluster_label_list = cluster_result[:-1]
                                connection_count = defaultdict(lambda: defaultdict(int))

                                for label, gt_id in zip(cluster_label_list, target_id_list_final):
                                    connection_count[label][gt_id] += 1

                                label_mapping = {}
                                used_gt_ids = set()
                                sorted_connections = sorted(
                                    [(count, label, gt_id) for label, connections in connection_count.items() for gt_id, count in connections.items()],
                                    key=lambda x: x[0], 
                                    reverse=True
                                )

                                for count, label, gt_id in sorted_connections:
                                    if label not in label_mapping and gt_id not in used_gt_ids:
                                        label_mapping[label] = gt_id
                                        used_gt_ids.add(gt_id)
                                
                                for cluster_id in set(cluster_result):
                                    if cluster_id not in label_mapping:
                                        label_mapping[cluster_id] = new_id
                                        new_id += 1
                                
                                for x, tup in enumerate(dup_tuples):
                                    channel_id_mapping[tup[0]][tup[1]] = label_mapping[cluster_result[x]]
                                target_id = label_mapping[cluster_result[-1]]
                                
                        else:
                            target_id = None

                        if target_id is not None:
                            channel_id_mapping[id_obj[0]][id_obj[1]] = target_id
                        else:
                            channel_id_mapping[id_obj[0]][id_obj[1]] = new_id
                            new_id += 1 
                        
        # Update final output
        for idx in range(len(caps)):
            if frame_count in channel_track_result[idx]:
                track_bbox = channel_track_result[idx][frame_count]
                for frame_bbox in track_bbox:
                    global_x, global_y = -1, -1
                    track_id = frame_bbox[-1]
                    if track_id in channel_id_position[idx]:
                        global_x, global_y = channel_id_position[idx][track_id][-1]

                    global_track_id = -1
                    if track_id in channel_id_mapping[idx]:
                        global_track_id = channel_id_mapping[idx][track_id]

                    x, y, w, h = frame_bbox[0], frame_bbox[1], frame_bbox[2]-frame_bbox[0], frame_bbox[3]-frame_bbox[1]

                    if global_track_id != -1:
                        channel_final_output[idx].append([idx, global_track_id, frame_count, x, y, w, h, global_x, global_y])
                    else:
                        if track_id in channel_loss[idx]:
                            channel_loss[idx][track_id].append([idx, global_track_id, frame_count, x, y, w, h, global_x, global_y])
                        else:
                            channel_loss[idx][track_id] = [[idx, global_track_id, frame_count, x, y, w, h, global_x, global_y]]
                
        channel_final_output, channel_loss = offline_tracking(channel_loss, channel_id_mapping, channel_final_output) 
        t.update(1)
    
    # Release video captures and writers
    for cap in caps:
        cap.release()
    
    # ======================================================================
    # RELEASE VIDEO WRITERS (ADDED HERE)
    # ======================================================================
    for writer in video_writers:
        writer.release()
    # ======================================================================
        
    # Save results
    save_txt_result(channel_final_output, args.save_out_path, channel_list)

if __name__ == "__main__":
    args = make_parser().parse_args()
    main(args)