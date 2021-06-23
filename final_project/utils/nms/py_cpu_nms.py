# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        inter_x1 = np.maximum(x1[i], x1[order[1:]])
        inter_y1 = np.maximum(y1[i], y1[order[1:]])
        inter_x2 = np.minimum(x2[i], x2[order[1:]])
        inter_y2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, inter_x2 - inter_x1)
        h = np.maximum(0.0, inter_y2 - inter_y1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # diounms, delete will be normal nms
        center_dist_x = np.square((x1[i] + x1[order[1:]]) / 2 - (x2[i] + x2[order[1:]]) / 2)
        center_dist_y = np.square((y1[i] + y1[order[1:]]) / 2 - (y2[i] + y2[order[1:]]) / 2)
        center_dist = center_dist_x + center_dist_y
        
        outer_x1 = np.minimum(x1[i], x1[order[1:]])
        outer_y1 = np.minimum(y1[i], y1[order[1:]])
        outer_x2 = np.maximum(x2[i], x2[order[1:]])
        outer_y2 = np.maximum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, outer_x2 - outer_x1)
        h = np.maximum(0.0, outer_y2 - outer_y1)
        bounding = w**2 + h**2
        
        ovr = ovr - center_dist/bounding        
        # diounms end

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
