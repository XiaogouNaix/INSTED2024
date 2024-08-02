# The scoring program compute scores from:
# - The ground truth
# - The predictions made by the candidate model

# Imports
import json
import os
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import SimpleITK as sitk
import pickle
from medpy.metric import hd95
from skimage import io, measure, morphology
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import pdist, squareform

# Path
input_dir = '/app/input'    # Input from ingestion program
output_dir = '/app/output/' # To write the scores
reference_dir = os.path.join(input_dir, 'ref')  # Ground truth data
prediction_dir = os.path.join(input_dir, 'res') # Prediction made by the model

# input_dir = 'test_set_validation/image'
# output_dir = './'
# reference_dir = 'test_set_validation/seg'
# prediction_dir = 'pred_example/'

score_file = os.path.join(output_dir, 'scores.json')          # Scores
html_file = os.path.join(output_dir, 'detailed_results.html') # Detailed feedback

def box3dVolume(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])

def rotz(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  -s,  0],
                    [s,  c,  0],
                    [0, 0,  1]])


def vocAP(rec, prec, use_07_metric=False):
    """ ap = vocAP(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get3dIoU(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two 3D bounding boxes.
    box1 and box2 should be in the format [x1, y1, z1, x2, y2, z2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    z1_inter = max(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])
    y2_inter = min(box1[4], box2[4])
    z2_inter = min(box1[5], box2[5])

    if x1_inter < x2_inter and y1_inter < y2_inter and z1_inter < z2_inter:
        inter_volume = (x2_inter - x1_inter) * (y2_inter - y1_inter) * (z2_inter - z1_inter)
    else:
        inter_volume = 0

    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    iou_value = inter_volume / (box1_volume + box2_volume - inter_volume)
    return iou_value

def evalDetectionClass(pred, gt, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for a single class.
        Input:
            pred: map of {img_id: [(bbox, score)]} where bbox is numpy array
            gt: map of {img_id: [bbox]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if True use VOC07 11 point method
        Output:
            rec: numpy array of length nd
            prec: numpy array of length nd
            ap: scalar, average precision
    """

    # construct gt objects
    class_recs = {} # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
        bbox = np.array(gt[img_id])
        det = [False] * len(bbox)
        npos += len(bbox)
        class_recs[img_id] = {'bbox': bbox, 'det': det}

        # import matplotlib.pyplot as plt
        # from mpl_toolkits.mplot3d import Axes3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for bbox_ in bbox:
        #     ax.plot(*bbox_.T,'.')
        # ax.set_zlim3d([0,200])
        # plt.show()
    # pad empty list to all other imgids
    for img_id in pred.keys():
        if img_id not in gt:
            class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    BB = []
    for img_id in pred.keys():
        for box,score in pred[img_id]:
            image_ids.append(img_id)
            confidence.append(score)
            BB.append(box)
    confidence = np.array(confidence)
    BB = np.array(BB) # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence, kind='stable')
    sorted_scores = np.sort(-confidence, kind='stable')
    BB = BB[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d,...].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            for j in range(BBGT.shape[0]):
                iou = get3dIoU(bb, BBGT[j,...])
                if iou > ovmax:
                    ovmax = iou
                    jmax = j

        if ovmax > ovthresh:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = vocAP(rec, prec, use_07_metric)

    return rec, prec, ap

def evalDetectionClassWrapper(arguments):
    pred, gt, ovthresh, use_07_metric = arguments
    rec, prec, ap = evalDetectionClass(pred, gt, ovthresh, use_07_metric)
    return (rec, prec, ap)

def evalDetection(pred_all, gt_all, ovthresh=0.25, use_07_metric=False):
    """ Generic functions to compute precision/recall for object detection
        for multiple classes.
        Input:
            pred_all: map of {img_id: [(classname, bbox, score)]}
            gt_all: map of {img_id: [(classname, bbox)]}
            ovthresh: scalar, iou threshold
            use_07_metric: bool, if true use VOC07 11 point method
        Output:
            rec: {classname: rec}
            prec: {classname: prec_all}
            ap: {classname: scalar}
    """
    pred = {} # map {classname: pred}
    gt = {} # map {classname: gt}
    for img_id in pred_all.keys():
        for classname, bbox, score in pred_all[img_id]:
            if classname not in pred: pred[classname] = {}
            if img_id not in pred[classname]:
                pred[classname][img_id] = []
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            pred[classname][img_id].append((bbox,score))
    for img_id in gt_all.keys():
        for classname, bbox in gt_all[img_id]:
            if classname not in gt: gt[classname] = {}
            if img_id not in gt[classname]:
                gt[classname][img_id] = []
            gt[classname][img_id].append(bbox)

    rec = {}
    prec = {}
    ap = {}
    for classname in gt.keys():
        print('Computing AP for class: ', classname)
        if classname not in pred:
            ap[classname] = 0
            rec[classname] = 0
            prec[classname] = 0
            continue
        rec[classname], prec[classname], ap[classname] = evalDetectionClass(pred[classname], gt[classname], ovthresh, use_07_metric)
        print(classname, ap[classname])

    return rec, prec, ap

def convert_output_to_detection(output):
    """
    Convert the output to the format of the detection.
    :param output: {img_id: [bbox]}
    :param wanted_class_num: 1 or 2
    :return: {img_id: [(bbox, score)]}
    """
    detection = {}
    for img_id, bbox_list in output.items():
        detection[img_id] = []
        for bbox in bbox_list:
            if bbox[2] >= bbox[3]:
                class_name = 'IA'
            else:
                class_name = 'stenosis'
            detection[img_id].append((class_name, bbox[0], bbox[1]))
    return detection

def convert_gt_to_detection(gt):
    """
    Convert the ground truth to the format of the detection.
    :param gt: {img_id: [[bbox, class_num]]}
    :return: {img_id: [(classname, bbox)]}
    """
    detection = {}
    for img_id, bbox_list in gt.items():
        detection[img_id] = []
        for bbox in bbox_list:
            if bbox[1] == 1:
                class_name = 'IA'
            elif bbox[1] == 2:
                class_name = 'stenosis'
            detection[img_id].append((class_name, bbox[0]))
    return detection

def write_file(file, content):
    """ Write content in file.
    """
    with open(file, 'a', encoding="utf-8") as f:
        f.write(content)

def print_bar():
    """ Display a bar ('----------')
    """
    print('-' * 10)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    fig_b64 = base64.b64encode(buf.getvalue()).decode('ascii')
    return fig_b64

def load_data():  # reference_dir: ["1seg.nii.gz", "2seg.nii.gz", "3seg.nii.gz", "4seg.nii.gz", ...]
    lst = os.listdir(reference_dir)
    reference_masks = {}
    pred_masks = {}
    for i in lst:
        if i.endswith("seg.nii.gz"):
            reference_masks[i[:-10] + ".nii.gz"] = sitk.ReadImage(os.path.join(reference_dir, i))
            pred_masks[i[:-10] + ".nii.gz"] = sitk.ReadImage(os.path.join(prediction_dir, i[:-10] + 'pred.nii.gz'))
    assert len(reference_masks) == len(pred_masks)
    reference_bbox = pickle.load(open(os.path.join(reference_dir, 'bbox_gt' + '.pkl'), 'rb'))
    pred_bbox = pickle.load(open(os.path.join(prediction_dir, 'bbox_pred' + '.pkl'), 'rb'))
    assert len(reference_bbox) == len(pred_bbox) == len(reference_masks)
    return reference_masks, pred_masks, reference_bbox, pred_bbox

def calc_num_instances(gt_dic):
    num_instances_IA = 0
    num_instances_stenosis = 0
    for k, v in gt_dic.items():
        for instance in v:
            if instance[1] == 1:
                num_instances_IA += 1
            elif instance[1] == 2:
                num_instances_stenosis += 1
    return num_instances_IA, num_instances_stenosis

def partition_gt_dic(gt_dic):
    gt_dic_IA = {}
    gt_dic_stenosis = {}
    gt_dic_hc = {}
    for k, v in gt_dic.items():
        lst_IA = [instance for instance in v if instance[1] == 1]
        lst_stenosis = [instance for instance in v if instance[1] == 2]
        if len(lst_IA) > 0:
            gt_dic_IA[k] = lst_IA
        if len(lst_stenosis) > 0:
            gt_dic_stenosis[k] = lst_stenosis
        if len(lst_IA) == 0 and len(lst_stenosis) == 0:
            gt_dic_hc[k] = v
    return gt_dic_IA, gt_dic_stenosis, gt_dic_hc

def convert_output_to_wanted_class_detection(output, wanted_class_num=1):
    """
    Convert the output of the model to a list of detections.
    Set wanted_class_num to the class number you want to detect. 1 is IA and 2 is stenosis.
    """
    detections = {}
    for k, v in output.items():  # detections of each image
        lst_this_image = []
        for j in range(len(v)):  # detections in this image
            lst2 = v[j][2:]
            max_idx = np.argmax(lst2)
            if max_idx + 1 == wanted_class_num:
                lst_this_image.append((v[j][0], v[j][1]))
        detections[k] = lst_this_image
    return detections

def prediction_dicts_to_lists(pred_dict, gt_dict, wanted_class_num=1):
    """
    Convert the prediction and ground truth dictionaries to lists.
    """
    pred_lst = []
    gt_lst = []
    for k, v in gt_dict.items():
        gt_to_append = []
        for instance in v:
            if instance[1] == wanted_class_num:
                gt_to_append.append(instance[0])
        gt_lst.append(gt_to_append)
        if k not in pred_dict:
            raise ValueError("Key {} in gt dictionary not found in pred dictionary.".format(k))
        pred_lst.append(pred_dict[k])
    return pred_lst, gt_lst

def iou_3d(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two 3D bounding boxes.
    box1 and box2 should be in the format [x1, y1, z1, x2, y2, z2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    z1_inter = max(box1[2], box2[2])
    x2_inter = min(box1[3], box2[3])
    y2_inter = min(box1[4], box2[4])
    z2_inter = min(box1[5], box2[5])

    if x1_inter < x2_inter and y1_inter < y2_inter and z1_inter < z2_inter:
        inter_volume = (x2_inter - x1_inter) * (y2_inter - y1_inter) * (z2_inter - z1_inter)
    else:
        inter_volume = 0

    box1_volume = (box1[3] - box1[0]) * (box1[4] - box1[1]) * (box1[5] - box1[2])
    box2_volume = (box2[3] - box2[0]) * (box2[4] - box2[1]) * (box2[5] - box2[2])

    iou_value = inter_volume / (box1_volume + box2_volume - inter_volume)
    return iou_value

# def compute_precision_recall(pred_boxes_lst, gt_boxes_lst, iou_threshold=0.5):
#     """
#     :param pred_boxes_lst: list of instances, each instance is a list of predicted boxes, each box is a tuple of [bbox, confidence]. 
#     e.g. [[([x1, y1, z1, x2, y2, z2], confidence), ...], ...]
#     :param gt_boxes_lst: list of instances, each instance is a list of ground truth boxes
#     e.g. [[[x1, y1, z1, x2, y2, z2], ...], ...]
#     """
#     all_tp = []
#     all_fp = []
#     all_probs = []
#     total_gt_boxes = 0
#     for pred_boxes, gt_boxes in zip(pred_boxes_lst, gt_boxes_lst):
        
#         pred_boxes = sorted(pred_boxes, key=lambda x: x[1], reverse=True)
#         tp = np.zeros(len(pred_boxes))
#         fp = np.zeros(len(pred_boxes))
#         probs = np.zeros(len(pred_boxes))
#         total_gt_boxes_this = len(gt_boxes)
#         total_gt_boxes += total_gt_boxes_this

#         matched_gt = []

#         for pred_idx, pred in enumerate(pred_boxes):
#             best_iou = 0
#             best_gt_idx = -1
#             for gt_idx, gt in enumerate(gt_boxes):
#                 if gt_idx in matched_gt:
#                     continue
#                 iou_value = iou_3d(pred[0], gt)
#                 if iou_value > best_iou:
#                     best_iou = iou_value
#                     best_gt_idx = gt_idx

#             if best_iou >= iou_threshold:
#                 tp[pred_idx] = 1
#                 matched_gt.append(best_gt_idx)
#             else:
#                 fp[pred_idx] = 1
#             probs[pred_idx] = pred[1]

#         all_tp.append(tp)
#         all_fp.append(fp)
#         all_probs.append(probs)
    
#     tp = np.concatenate(all_tp)
#     fp = np.concatenate(all_fp)
#     probs = np.concatenate(all_probs)

#     sorted_indices = np.argsort(-probs)
#     tp = tp[sorted_indices]
#     fp = fp[sorted_indices]
#     probs = probs[sorted_indices]

#     cumulative_tp = np.cumsum(tp)
#     cumulative_fp = np.cumsum(fp)
#     precision = cumulative_tp / (cumulative_tp + cumulative_fp)
#     recall = cumulative_tp / total_gt_boxes

#     return precision, recall, probs

# def compute_ap(precision, recall, eleven_points_avg=False):
#     """
#     Compute Average Precision (AP) given precision and recall arrays
#     """
#     precision = np.concatenate(([1.0], precision, [0.0]))
#     recall = np.concatenate(([0.0], recall, [1.0]))

#     for i in range(len(precision) - 1, 0, -1):  # Smoothing the precision curve
#         precision[i - 1] = np.maximum(precision[i - 1], precision[i])

#     # Compute the area under the curve
#     if not eleven_points_avg:
#         indices = np.where(recall[1:] != recall[:-1])[0]
#         ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
#     else:
#         # Compute average precision at 11 recall points
#         recall_points = np.linspace(0, 1, 11)
#         ap = 0
#         for recall_point in recall_points:
#             if np.sum(recall >= recall_point) == 0:
#                 precision_point = 0
#             else:
#                 precision_point = np.max(precision[recall >= recall_point])
#             ap += precision_point
#         ap /= 11

#     return ap

# def calculate_ap(predictions, ground_truths, iou_threshold=0.5, eleven_points_avg=False):
#     """
#     Calculate AP from prediction and ground truth boxes for multiple instances
#     predictions: list of list of predictions per image, each prediction is a list of [bbox, confidence]
#     ground_truths: list of list of ground truth boxes per image
#     """
#     all_precisions, all_recalls, _ = compute_precision_recall(predictions, ground_truths, iou_threshold)
#     if len(all_precisions) == 0:
#         return 0

#     return compute_ap(all_precisions, all_recalls, eleven_points_avg=eleven_points_avg)

def get_tp_bboxes(pred_boxes, gt_boxes, iou_thres=0.15):
    """
    Get the true positive bounding boxes. Any box with IoU >= iou_thres and probability > 0 is considered a true positive.
    """
    tp_bboxes = {}
    for k, v in gt_boxes.items():
        # if k not in pred_boxes:
        #     raise ValueError("Key {} in pred dictionary not found in ground truth dictionary.".format(k))
        pred_boxes_this = pred_boxes[k]
        gt_boxes_this = gt_boxes[k]
        tp_bboxes_this = []
        matched_gt = []
        for pred_idx, pred in enumerate(pred_boxes_this):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_boxes_this):
                if gt_idx in matched_gt:
                    continue
                iou_value = iou_3d(pred[0], gt[0])
                if iou_value > best_iou:
                    best_iou = iou_value
                    best_gt_idx = gt_idx

            if best_iou >= iou_thres:
                tp_bboxes_this.append(gt_boxes_this[best_gt_idx])
                matched_gt.append(best_gt_idx)
        tp_bboxes[k] = tp_bboxes_this
    return tp_bboxes

def dice_score(pred, gt):
    """
    Calculate Dice score between two binary masks
    """
    pred = np.bool_(pred)
    gt = np.bool_(gt)
    intersection = np.count_nonzero(pred & gt)
    union = np.count_nonzero(pred | gt)
    dice = 2 * intersection / (np.count_nonzero(pred) + np.count_nonzero(gt))
    return dice

def hausdorff_distance_unified(pred, gt, baseline, voxel_spacing):
    """
    Calculate Hausdorff distance between two binary masks, then unify the result to (0-1) with a baseline
    """
    pred = np.bool_(pred)
    gt = np.bool_(gt)
    if np.sum(pred) == 0:
        pred[pred.shape[0] // 2, pred.shape[1] // 2, pred.shape[2] // 2] = 1
    hd = hd95(pred, gt, voxel_spacing)
    hd_baseline = hd95(baseline, gt, voxel_spacing)
    hd = 1 - hd / hd_baseline
    if hd < 0:
        hd = 0
    return hd

def max_and_min_diameters(segmentation_image, spacing):
    """
    :param segmentation_image: Binary segmentation image with the vessel as 1 and background as 0.
    """
    binary_image = segmentation_image > 0
    binary_image = get_max_connective_field(binary_image)
    skeleton = morphology.skeletonize(binary_image)
    distance_transform = distance_transform_edt(binary_image, sampling=spacing)

    # Get the coordinates of the skeleton
    skeleton_coords = np.column_stack(np.where(skeleton))

    # Calculate the diameter at each point in the skeleton
    diameters = [2 * distance_transform[tuple(coord)] for coord in skeleton_coords]
    if len(diameters) == 0:
        return np.max(distance_transform), np.max(distance_transform)

    return np.max(diameters), np.min(diameters)

def max_diameter_short_radius(arr):
    if np.sum(arr) == 0:
        return 0, 0
    # 将2D数组转换为1D数组
    flat_arr = np.ravel(arr)
    # 找到所有值为1的索引
    indices = np.where(flat_arr == 1)[0]
    # 将索引转换为2D坐标
    coordinates = np.column_stack(np.unravel_index(indices, arr.shape))
    # 计算所有点之间的距离
    distances = squareform(pdist(coordinates))
    # 找到距离最远的两个点
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    # 计算最大直径
    max_diameter = distances[i, j]
    # 计算垂直于最大直径的两个点
    midpoint = (coordinates[i] + coordinates[j]) / 2
    vector = coordinates[j] - coordinates[i]
    perp_vector = np.array([-vector[1], vector[0]])
    # 计算垂直于最大直径的两个点的坐标
    k = np.argmax(np.abs(np.dot(coordinates - midpoint, perp_vector)))
    l = np.argmin(np.abs(np.dot(coordinates - midpoint, perp_vector)))
    # 计算垂直于最大直径的长度
    short_radius = np.linalg.norm(coordinates[k] - coordinates[l])
    return max_diameter, short_radius

def get_2d_diameters(label_arr,nodule_spacing):
    if np.sum(label_arr) == 0:
        return 0, 0
    mask = label_arr
    mask = get_max_connective_field(mask)
    largest_z = np.argmax(np.sum(mask, axis=(1,2)))
    lag_z = mask[largest_z]
    max_diameter, short_diameter =max_diameter_short_radius(lag_z)
    return max_diameter*nodule_spacing[1],short_diameter*nodule_spacing[1]

def get_max_connective_field(cube):
    if np.sum(cube) == 0:
        return cube
    result = measure.label(cube)
    result1 = result.reshape([-1])
    lst = np.bincount(result1)
    lst[0] = 0
    a = np.argmax(lst)
    result[result != a] = 0
    result[result == a] = 1
    return result

def get_arr_in_bbox(arr, bbox, expansion=0):
    return arr[max(0, bbox[0] - expansion):min(arr.shape[0], bbox[3] + expansion), max(0, bbox[1] - expansion):min(arr.shape[1], bbox[4] + expansion), max(0, bbox[2] - expansion):min(arr.shape[2], bbox[5] + expansion)]
    

def get_segmentation_metrics(pred_masks, gt_masks, tp_dic, gt_dic):
    metrics_dic = {
        "dice_IA": [],
        "hd_IA": [],
        "dice_stenosis": [],
        "hd_stenosis": [],
    }
    metrics_dic["IA_long_axis"] = []
    metrics_dic["IA_short_axis"] = []
    metrics_dic["stenosis_percentage"] = []
    IA_num_instances, stenosis_num_instances = calc_num_instances(gt_dic)
            
    for k, v in gt_masks.items():
        if k not in pred_masks:
            raise ValueError("Key {} in pred dictionary not found in ground truth dictionary.".format(k))
        pred_mask = pred_masks[k]
        gt_mask = gt_masks[k]
        tp_bboxes = tp_dic[k]
        pred_mask_arr = sitk.GetArrayFromImage(pred_mask)
        gt_mask_arr_loaded = sitk.GetArrayFromImage(gt_mask)  # stenosis: 1 for stenosis part and 2 for normal part nearby
        gt_mask_arr = gt_mask_arr_loaded == 1
        pred_mask_arr = pred_mask_arr == 1
        voxel_spacing = list(reversed(pred_mask.GetSpacing()))
        for bbox_gt in tp_bboxes:
            if bbox_gt[1] == 1:
                lesion_type = "IA"
            elif bbox_gt[1] == 2:
                lesion_type = "stenosis"
            else:
                raise ValueError("Invalid lesion type.")
            bbox = bbox_gt[0]
            pred_img_in_bbox = get_arr_in_bbox(pred_mask_arr, bbox, expansion=0)
            label_img_in_bbox = get_arr_in_bbox(gt_mask_arr, bbox, expansion=0)
            baseline_pred_img_in_bbox = np.ones_like(pred_img_in_bbox)

            dice_this = dice_score(pred_img_in_bbox, label_img_in_bbox)
            hd_this = hausdorff_distance_unified(pred_img_in_bbox, label_img_in_bbox, baseline_pred_img_in_bbox, voxel_spacing)
            metrics_dic["dice_{}".format(lesion_type)].append(dice_this)
            metrics_dic["hd_{}".format(lesion_type)].append(hd_this)

            if lesion_type == "stenosis":
                gt_mask_arr_with_normal = gt_mask_arr_loaded >= 1
                normal_mask = gt_mask_arr_loaded == 2
                # pred_mask_arr_with_normal_mask = pred_mask_arr | normal_mask
                pred_mask_arr_in_bbox = get_arr_in_bbox(pred_mask_arr, bbox, expansion=3)
                label_img_in_bbox_with_normal = get_arr_in_bbox(gt_mask_arr_with_normal, bbox, expansion=3)
                gt_max, _ = max_and_min_diameters(label_img_in_bbox_with_normal, voxel_spacing)  # note that we have labelled the ordinary vessel beside the stenosis site in our test set GT using another label, so we can calculate the diameter of the ordinary vessel
                _, gt_min = max_and_min_diameters(label_img_in_bbox, voxel_spacing)
                _, pred_min = max_and_min_diameters(pred_mask_arr_in_bbox, voxel_spacing)
                gt_percentage = (gt_max - gt_min) / gt_max
                pred_percentage = (gt_max - pred_min) / gt_max
                metrics_dic["stenosis_percentage"].append(max(0, 1 - abs(gt_percentage - pred_percentage)))
            elif lesion_type == "IA":
                gt_max, gt_min = get_2d_diameters(label_img_in_bbox, voxel_spacing)
                pred_max, pred_min = get_2d_diameters(pred_img_in_bbox, voxel_spacing)
                metrics_dic["IA_long_axis"].append(max(0, 1 - abs(gt_max - pred_max) / gt_max))
                metrics_dic["IA_short_axis"].append(max(0, 1 - abs(gt_min - pred_min) / gt_min))

    tp_perc_dic = {}
    tp_perc_dic["IA_tp_perc"] = len(metrics_dic["dice_IA"]) / IA_num_instances
    tp_perc_dic["stenosis_tp_perc"] = len(metrics_dic["dice_stenosis"]) / stenosis_num_instances

    for k, v in metrics_dic.items():
        if len(v) == 0:
            metrics_dic[k] = 0
        else:
            if k in ["dice_IA", "hd_IA", "IA_long_axis", "IA_short_axis"]:
                metrics_dic[k] = np.sum(v) / IA_num_instances
            elif k in ["dice_stenosis", "hd_stenosis", "stenosis_percentage"]:
                metrics_dic[k] = np.sum(v) / stenosis_num_instances
            else:
                raise ValueError("Invalid key.")

    metrics_dic.update(tp_perc_dic)
    return metrics_dic

def main():
    """ The scoring program.
    """
    print_bar()
    print('Scoring program.')
    # Initialized detailed results
    write_file(html_file, '<h1>Detailed results</h1>') # Create the file to give real-time feedback
    scores = {}
    # Evaluate IA
    masks_gt, masks_pred, bbox_gt, bbox_pred = load_data()
    bbox_IA_pred = convert_output_to_wanted_class_detection(bbox_pred, 1)
    bbox_stenosis_pred = convert_output_to_wanted_class_detection(bbox_pred, 2)

    bbox_pred_for_ap = convert_output_to_detection(bbox_pred)
    bbox_gt_for_ap = convert_gt_to_detection(bbox_gt)

    _, _, ap15 = evalDetection(bbox_pred_for_ap, bbox_gt_for_ap, ovthresh=0.15, use_07_metric=False)
    _, _, ap25 = evalDetection(bbox_pred_for_ap, bbox_gt_for_ap, ovthresh=0.25, use_07_metric=False)
    ap15_IA = ap15["IA"]
    ap25_IA = ap25["IA"]
    ap15_stenosis = ap15["stenosis"]
    ap25_stenosis = ap25["stenosis"]
    ap_IA = (ap15_IA + ap25_IA) / 2
    ap_stenosis = (ap15_stenosis + ap25_stenosis) / 2
    scores["ap_IA"] = ap_IA
    scores["ap_stenosis"] = ap_stenosis

    bbox_gt_IA, bbox_gt_stenosis, bbox_gt_hc = partition_gt_dic(bbox_gt)
    tp_bboxes_dic = get_tp_bboxes(bbox_IA_pred, bbox_gt_IA, iou_thres=0.15)
    tp_bboxes_dic_stenosis = get_tp_bboxes(bbox_stenosis_pred, bbox_gt_stenosis, iou_thres=0.15)
    tp_bboxes_dic.update(tp_bboxes_dic_stenosis)  
    tp_bboxes_dic.update(bbox_gt_hc)  # all tp bboxes
    metrics_dic = get_segmentation_metrics(masks_pred, masks_gt, tp_bboxes_dic, bbox_gt)
    scores.update(metrics_dic)

    total_score = 0.6 * (0.5 * scores["ap_IA"] + 0.5 * scores["ap_stenosis"]) + 0.1 * (0.5 * scores["dice_IA"] + 0.5 * scores["dice_stenosis"]) + 0.1 * (0.5 * scores["hd_IA"] + 0.5 * scores["hd_stenosis"]) + 0.2 * (1 / 3 * scores["IA_long_axis"] + 1 / 3 * scores["IA_short_axis"] + 1 / 3 * scores["stenosis_percentage"])
    scores["total_score"] = total_score

    # Write scores
    print_bar()
    print('Scoring program finished. Writing scores.')
    print(scores)
    write_file(score_file, json.dumps(scores))
    # Create a figure for detailed results
    # figure = fig_to_b64(make_figure(scores))
    # write_file(html_file, f'<img src="data:image/png;base64,{figure}">')

if __name__ == '__main__':
    main()