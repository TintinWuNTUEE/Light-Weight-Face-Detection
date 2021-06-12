import os
import tqdm
import pickle
import argparse
import numpy as np
from bbox import bbox_overlaps
from math import sqrt

def get_gt_boxes(gt_file):
    with open(gt_file,'rb') as handle:
        gt = pickle.load(handle)

    facebox_list = gt['face_bbx_list']
    file_list = gt['file_list'] #image name
    ignore_list = gt['ignore_idx']
    facelm_list = gt['face_lm_list']
    return facebox_list, file_list, ignore_list, facelm_list

def get_preds(pred_file):
    print('Reading solution file...')
    begin = True
    preds = dict()
    f = open(pred_file,'r')
    while True:
        line = f.readline().rstrip('\n\r')
        if not line:
            break
        if line.startswith('#'):
            if begin:
                begin = False
            else:
                preds[img_name] = np.array(preds[img_name],dtype='float')
                assert(len(preds[img_name])==n_bboxes)
                # new start
            img_name = line.split(' ')[-1]
            n_bboxes = int(f.readline().rstrip('\n\r'))
            preds[img_name] = []
        else:
            pred = line.split(' ')
            preds[img_name].append(pred)
    preds[img_name] = np.array(preds[img_name],dtype='float')
    assert(len(preds[img_name])==n_bboxes)
    f.close()
    return preds


def norm_score(pred):
    max_score = 0
    min_score = 1

    for _, v in pred.items():
        if len(v) == 0:
            continue
        _min = np.min(v[:, 4])
        _max = np.max(v[:, 4])
        max_score = max(_max, max_score)
        min_score = min(_min, min_score)

    diff = max_score - min_score
    for _, v in pred.items():
        if len(v) == 0:
            continue
        v[:, 4] = (v[:, 4] - min_score)/diff

def calculate_NME(pred,lm_gt,gt):
    NME = 0
    normalize = sqrt((gt[2]-gt[0])*(gt[3]-gt[1]))
    diff = (pred - lm_gt)**2
    for i in range(5):
        NME += sqrt(diff[2*i] + diff[2*i+1])
    NME /= (5*normalize)
    return NME

def image_eval(pred, gt, keep, iou_thresh=0.5,is_lm=False, lm_gt=None):

    _pred = pred.copy()
    _gt = gt.copy()
    _lm_gt = lm_gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    # convert to x1,y1, x2,y2
    proposal_list = np.ones(_pred.shape[0])
    _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    # pairwise iou
    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    nme_list = []
    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if keep[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1
                #### lm #####
                if is_lm:
                    assert _pred.shape[1] == 15,"The format of Landmark is incorrect"
                    _pred_lm = _pred[h,5:]
                    if not (-1 in _lm_gt[max_idx]):
                        nme = calculate_NME(_pred_lm,_lm_gt[max_idx],_gt[max_idx])
                        nme_list.append(nme)
        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list, nme_list


def img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t+1)/thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[:r_index+1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def voc_ap(rec, prec):

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


def evaluation(pred, gt_path, is_lm, iou_thresh=0.5):
    pred = get_preds(pred) 
    norm_score(pred)
    
    facebox_list, file_list, ignore_list, facelm_list = get_gt_boxes(gt_path)
    thresh_num = 1000
    
    num_images = len(pred)
    count_face = 0
    pr_curve = np.zeros((thresh_num, 2)).astype('float')

    all_nme = []
    pbar = tqdm.tqdm(range(num_images),ncols=100)
    for i in pbar:
        if is_lm:
            pbar.set_description('Processing AP & NME')
        else:
            pbar.set_description('Processing AP')
        img_name = file_list[i]

        pred_info = pred[img_name]
        gt_boxes = facebox_list[i].astype('float')
        ignore_idx = ignore_list[i]
        ##
        gt_lm = facelm_list[i].astype('float')

        count_face += len(gt_boxes) - len(ignore_idx)
        if len(gt_boxes) == 0 or len(pred_info) == 0:
            continue

        keep = np.ones(gt_boxes.shape[0])
        if len(ignore_idx) != 0:
            keep[ignore_idx] = 0

        pred_recall, proposal_list ,nme_list = image_eval(pred_info, gt_boxes, keep, iou_thresh,is_lm, gt_lm)
        all_nme += nme_list
        _img_pr_info = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve += _img_pr_info
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)

    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]

    ap = voc_ap(recall, propose)

    print("==================== Results ====================")
    print("Val AP: {}".format(ap))
    print("=================================================")

    if is_lm:
        NME_score = 100*sum(all_nme)/len(all_nme)
        print("==================== Results ====================")
        print("Val NME: {}".format(NME_score))
        print("=================================================")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pred', default="./solution.txt")
    parser.add_argument('-g', '--gt', default='./ground_truth/val_gt.pkl')
    parser.add_argument('-lm',action='store_true',help='whether evaluating the landmark')

    args = parser.parse_args()
    evaluation(args.pred, args.gt,args.lm)












