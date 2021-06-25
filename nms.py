import numpy as np
import cv2
import torch

def calc_IoU(a, b):
    # step1:
    inter_x1 = np.maximum(np.min(a[:,0]), np.min(b[:,0]))
    inter_x2 = np.minimum(np.max(a[:,0]), np.max(b[:,0]))
    inter_y1 = np.maximum(np.min(a[:,1]), np.min(b[:,1]))
    inter_y2 = np.minimum(np.max(a[:,1]), np.max(b[:,1]))
    # print("hello")
    if inter_x1>=inter_x2 or inter_y1>=inter_y2:
        return 0.
    x1 = np.minimum(np.min(a[:,0]), np.min(b[:,0]))
    x2 = np.maximum(np.max(a[:,0]), np.max(b[:,0]))
    y1 = np.minimum(np.min(a[:,1]), np.min(b[:,1]))
    y2 = np.maximum(np.max(a[:,1]), np.max(b[:,1]))
    if x1>=x2 or y1>=y2 or (x2-x1)<2 or (y2-y1)<2:
        return 0.
    else:
        mask_w = np.int(np.ceil(x2-x1))
        mask_h = np.int(np.ceil(y2-y1))
        mask_a = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        mask_b = np.zeros(shape=(mask_h, mask_w), dtype=np.uint8)
        a[:,0] -= x1
        a[:,1] -= y1
        b[:,0] -= x1
        b[:,1] -= y1
        mask_a = cv2.fillPoly(mask_a, pts=np.asarray([a], 'int32'), color=1)
        mask_b = cv2.fillPoly(mask_b, pts=np.asarray([b], 'int32'), color=1)
        inter = np.logical_and(mask_a, mask_b).sum()
        union = np.logical_or(mask_a, mask_b).sum()
        iou = float(inter)/(float(union)+1e-12)
        # print(iou)
        # cv2.imshow('img1', np.uint8(mask_a*255))
        # cv2.imshow('img2', np.uint8(mask_b*255))
        # k = cv2.waitKey(0)
        # if k==ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()
        return iou

# def calc_ciou(bboxes1, bboxes2):
#     rows = bboxes1.shape[0]
#     cols = bboxes2.shape[0]
#     cious = torch.zeros((rows, cols))
#     if rows * cols == 0:
#         return cious
#     exchange = False
#     if bboxes1.shape[0] > bboxes2.shape[0]:
#         bboxes1, bboxes2 = bboxes2, bboxes1
#         cious = torch.zeros((cols, rows))
#         exchange = True
#
#     w1 = bboxes1[:, 2] - bboxes1[:, 0]
#     h1 = bboxes1[:, 3] - bboxes1[:, 1]
#     w2 = bboxes2[:, 2] - bboxes2[:, 0]
#     h2 = bboxes2[:, 3] - bboxes2[:, 1]
#
#     area1 = w1 * h1
#     area2 = w2 * h2
#
#     center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
#     center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
#     center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
#     center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2
#
#     inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
#     inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
#     out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
#     out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])
#
#     inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
#     inter_area = inter[:, 0] * inter[:, 1]
#     inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
#     outer = torch.clamp((out_max_xy - out_min_xy), min=0)
#     outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
#     union = area1+area2-inter_area
#     u = (inter_diag) / outer_diag
#     iou = inter_area / union
#     with torch.no_grad():
#         arctan = torch.atan(w2 / h2) - torch.atan(w1 / h1)
#         v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w2 / h2) - torch.atan(w1 / h1)), 2)
#         S = 1 - iou
#         alpha = v / (S + v)
#         w_temp = 2 * w1
#     ar = (8 / (math.pi ** 2)) * arctan * ((w1 - w_temp) * h1)
#     cious = iou - (u + alpha * ar)
#     cious = torch.clamp(cious,min=-1.0,max = 1.0)
#     if exchange:
#         cious = cious.T
#     return cious


def calc_Diou(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))
    if rows * cols == 0:  #
        return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    # x1min = np.min(bboxes1[:,0])
    # x1max = np.max(bboxes1[:,0])
    # w1 = x1max - x1min
    # x2min = np.min(bboxes2[:,0])
    # x2max = np.max(bboxes2[:,0])
    # w2 = x2max - x2min
    # y1min = np.min(bboxes1[:,1])
    # y1max = np.max(bboxes1[:,1])
    # h1 = y1max - y1min
    # y2min = np.min(bboxes2[:,1])
    # y2max = np.max(bboxes2[:,1])
    # h2 = y2max - y2min

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # center_x1 = (x1max + x1min) / 2
    # center_y1 = (y1max + y1min) / 2
    # center_x2 = (x2max + x2min) / 2
    # center_y2 = (y2max + y2min) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    # inter_max_xy = torch.min((x1max,y1max ),(x2max,y2max))
    # inter_min_xy = torch.max((x1min,y1min ),(x2min,y2min))
    # out_max_xy = torch.max((x1max,y1max ),(x2max,y2max))
    # out_min_xy = torch.min((x1min,y1min ),(x2min,y2min))

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    return dious

def draw_image(pts, image):
    cen_pts = np.mean(pts, axis=0)
    tt = pts[0, :]
    rr = pts[1, :]
    bb = pts[2, :]
    ll = pts[3, :]
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(tt[0]), int(tt[1])), (0, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(rr[0]), int(rr[1])), (255, 0, 255), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(bb[0]), int(bb[1])), (0, 255, 0), 2, 1)
    cv2.line(image, (int(cen_pts[0]), int(cen_pts[1])), (int(ll[0]), int(ll[1])), (255, 0, 0), 2, 1)
    return image


def NMS_numpy_exboxes(exboxes, conf, nms_thresh=0.5, image=None):
    if len(exboxes)==0:
        return None
    sorted_index = np.argsort(conf)      # Ascending order
    keep_index = []
    # print(type(exboxes))
    while len(sorted_index)>0:
        curr_index = sorted_index[-1]
        keep_index.append(curr_index)
        if len(sorted_index)==1:
            break
        sorted_index = sorted_index[:-1]
        IoU = []
        for index in sorted_index:
            iou = calc_IoU(exboxes[index,:,:].copy(), exboxes[curr_index,:,:].copy())
            IoU.append(iou)
        IoU = np.asarray(IoU, np.float32)
        sorted_index = sorted_index[IoU<=nms_thresh]
    return keep_index



# def NMS_numpy_bbox(bboxes, nms_thresh=0.5):
#     """
#     bboxes: num_insts x 5 [x1,y1,x2,y2,conf]
#     """
#     if len(bboxes)==0:
#         return None
#     x1 = bboxes[:,0]
#     y1 = bboxes[:,1]
#     x2 = bboxes[:,2]
#     y2 = bboxes[:,3]
#     conf = bboxes[:,4]
#     area_all = (x2-x1)*(y2-y1)
#     sorted_index = np.argsort(conf)      # Ascending order
#     keep_index = []
#
#     while len(sorted_index)>0:
#         # get the last biggest values
#         curr_index = sorted_index[-1]
#         keep_index.append(curr_index)
#         if len(sorted_index)==1:
#             break
#         # pop the value
#         sorted_index = sorted_index[:-1]
#         # get the remaining boxes
#         yy1 = np.take(y1, indices=sorted_index)
#         xx1 = np.take(x1, indices=sorted_index)
#         yy2 = np.take(y2, indices=sorted_index)
#         xx2 = np.take(x2, indices=sorted_index)
#         # get the intersection box
#         yy1 = np.maximum(yy1, y1[curr_index])
#         xx1 = np.maximum(xx1, x1[curr_index])
#         yy2 = np.minimum(yy2, y2[curr_index])
#         xx2 = np.minimum(xx2, x2[curr_index])
#         # calculate IoU
#         w = xx2-xx1
#         h = yy2-yy1
#         w = np.maximum(0., w)
#         h = np.maximum(0., h)
#         inter = w*h
#         rem_areas = np.take(area_all, indices=sorted_index)
#         union = (rem_areas-inter)+area_all[curr_index]
#         IoU = inter/union
#         sorted_index = sorted_index[IoU<=nms_thresh]
#
#     return keep_index



def NMS_numpy_bbox(boxes, sigma=0.5, Nt=0.1, threshold=0.001, method=1):
    N = boxes.shape[0]
    pos = 0
    maxscore = 0
    maxpos = 0

# boxes = np.array([[100, 100, 150, 168, 0.63], [166, 70, 312, 190, 0.55], [
                #  221, 250, 389, 500, 0.79], [12, 190, 300, 399, 0.9], [28, 130, 134, 302, 0.3]])
    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

    # add max box as a detection
        boxes[i, 0] = boxes[maxpos, 0]
        boxes[i, 1] = boxes[maxpos, 1]
        boxes[i, 2] = boxes[maxpos, 2]
        boxes[i, 3] = boxes[maxpos, 3]
        boxes[i, 4] = boxes[maxpos, 4]

    # swap ith box with position of max box
        boxes[maxpos, 0] = tx1
        boxes[maxpos, 1] = ty1
        boxes[maxpos, 2] = tx2
        boxes[maxpos, 3] = ty2
        boxes[maxpos, 4] = ts

        tx1 = boxes[i, 0]
        ty1 = boxes[i, 1]
        tx2 = boxes[i, 2]
        ty2 = boxes[i, 3]
        ts = boxes[i, 4]

        pos = i + 1
    # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) *
                               (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua  # iou between max box and detection box

                    if method == 1:  # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2:  # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else:  # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                    print(boxes[:, 4])

            # if box score falls below threshold, discard the box by swapping with last box
            # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos, 0] = boxes[N-1, 0]
                        boxes[pos, 1] = boxes[N-1, 1]
                        boxes[pos, 2] = boxes[N-1, 2]
                        boxes[pos, 3] = boxes[N-1, 3]
                        boxes[pos, 4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1
    keep = [i for i in range(N)]
    return keep


