import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import shapely

from decoder import DecDecoder
from shapely.geometry import Polygon, MultiPoint
import cmath
import math
from torch.autograd import Variable
from torch.nn import Module
from torch.autograd import Function



def template_pixels(height, width):
    xv, yv = torch.meshgrid(
        [torch.arange(-100, width + 100), torch.arange(-100, height + 100)])
    xy = torch.stack((xv, yv), -1)
    grid_xy = xy.reshape(-1, 2).float() + 0.5

    return grid_xy


def template_w_pixels(width):
    x = torch.tensor(torch.arange(-100, width + 100))
    grid_x = x.float() + 0.5
    return grid_x

def kernel_function(dis, k, t):
    # clamp to avoid nan
    factor = torch.clamp(-k * (dis - t), -50, 50)
    return 1.0 - 1.0 / (torch.exp(factor) + 1)


def pixel_weights(loc, grid_xy, k):

    xx = torch.pow(loc[:, :, 0:2], 2).sum(2)
    yy = torch.pow(grid_xy, 2).sum(2)
    dis = xx + yy
    # dis - 2 * x * yT
    dis.addmm_(1, -2, loc[:, 0, 0:2], grid_xy[0].t())


    dis = dis.clamp(min=1e-9).sqrt()  # for numerical stability

    a1 = loc[:, :, -1] - torch.acos((grid_xy[:, :, 0] - loc[:, :, 0]) / dis)
    a2 = loc[:, :, -1] + torch.acos((grid_xy[:, :, 0] - loc[:, :, 0]) / dis)
    a = torch.where(loc[:, :, 1] > grid_xy[:, :, 1], a1, a2)

    dis_w = dis * torch.abs(torch.cos(a))
    dis_h = dis * torch.abs(torch.sin(a))
    # print(dis_h)
    # return dis_h
    pixel_weights = kernel_function(
        dis_w, k, loc[:, :, 2] / 2.) * kernel_function(dis_h, k, loc[:, :, 3] / 2.)

    return pixel_weights

def PIoU(loc_p, loc_t, grid_xy, k=10):

    num = loc_p.size(0)
    dim = grid_xy.size(0)

    loc_pp = loc_p.unsqueeze(1).expand(num, dim, 5)
    loc_tt = loc_t.unsqueeze(1).expand(num, dim, 5)
    grid_xyxy = grid_xy.unsqueeze(0).expand(num, dim, 2)

    pixel_p_weights = pixel_weights(loc_pp, grid_xyxy, k)
    pixel_t_weights = pixel_weights(loc_tt, grid_xyxy, k)
    # print(torch.sum(pixel_p_weights,1))

    inter_pixel_area = pixel_p_weights * pixel_t_weights
    intersection_area = torch.sum(inter_pixel_area, 1)
    # print('intersection_area', intersection_area)
    union_pixel_area = pixel_p_weights + pixel_t_weights - inter_pixel_area
    union_area = torch.sum(union_pixel_area, 1)
    print('uniou_area',union_area + 0.000001)
    pious = intersection_area / (union_area + 0.000001)
    return torch.sum(1 - pious), pious

def cal_boxwh(a):
    listwh = []
    listxy = []
    cen_pt = np.asarray([a[0], a[1]], np.float32)
    tt = np.asarray([a[2], a[3]], np.float32)
    rr = np.asarray([a[4], a[5]], np.float32)
    bb = np.asarray([a[6], a[7]], np.float32)
    ll = np.asarray([a[8], a[9]], np.float32)

    if a[2] - a[6] == 0:
        theta = 3.141593
    else:
        th = a[3] - a[7]  / a[2] - a[6]
        theta = math.atan(th)
    h = cmath.sqrt((a[3] - a[7]) ** 2 + (a[2] - a[6]) ** 2)
    w = cmath.sqrt((a[5] - a[9]) ** 2 + (a[4] - a[8]) ** 2)
    w = float(abs(w))
    h = float(abs(h))
    listwh = [w,h]
    listxy = cen_pt.tolist()

    return listxy, listwh, theta


def cal_box(a):
    # cen_pt = np.asarray([a[0], a[1]], np.float32)
    # tt = np.asarray([a[2], a[3]], np.float32)
    # rr = np.asarray([a[4], a[5]], np.float32)
    # bb = np.asarray([a[6], a[7]], np.float32)
    # ll = np.asarray([a[8], a[9]], np.float32)

    tt = np.asarray([a[0], a[1]], np.float32)
    rr = np.asarray([a[2], a[3]], np.float32)
    bb = np.asarray([a[4], a[5]], np.float32)
    ll = np.asarray([a[6], a[7]], np.float32)
    cen_pt = (tt + bb) / 2


    tl1 = tt + ll
    bl1 = bb + ll
    tr1 = tt + rr
    br1 = bb + rr

    tl = tl1 - cen_pt
    bl = bl1 - cen_pt
    tr = tr1 - cen_pt
    br = br1 - cen_pt

    # print(tr1, br1, bl1, tl1)
    box = np.asarray([tr, br, bl, tl], np.float32)
    box = box.reshape(1,-1)
    box = np.squeeze(box)
    box = box.tolist()
    return  box

def iouloss(pred,target):
    giou = 0
    iou = 0
    a = np.array(pred).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(a).convex_hull)  # 可以打印看看是不是这样子


    b = np.array(target).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            union_area = poly1.area + poly2.area - inter_area
            # union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                iou = 0

            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0

            ##Diou
            # cnt = numpy.array([[x1,y1],[x3,y3],[x4,y4],[x2,y2]]) # 必须是array数组的形式
    union_poly = np.array(union_poly, dtype=np.float32)
    rect = cv2.minAreaRect(union_poly) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    cen, bbox ,theta = rect
    bbox_w = bbox[0]
    bbox_h = bbox[1]
    enclose_area  = bbox_w * bbox_h
    giou = iou - (enclose_area - union_area) / enclose_area


    return iou

def diou(pred , target):
    cent_ax = pred[0]
    cent_ay = pred[1]
    cent_bx = target[0]
    cent_by = target[1]
    pred = pred[2:]
    target = target[2:]
    # cent_ax, cent_ay = cal_cent(pred)
    # cent_bx, cent_by = cal_cent(target)
    a = np.array(pred).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(a).convex_hull)  # 可以打印看看是不是这样子


    b = np.array(target).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # print(Polygon(b).convex_hull)
    union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
        # print(union_poly)
    # print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            # print(inter_area)
            union_area = poly1.area + poly2.area - inter_area
                # union_area = MultiPoint(union_poly).convex_hull.area
            # print(union_area)
            if union_area == 0:
                    iou = 0
            iou = float(inter_area) / union_area
                # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
                # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
                # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:

            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0


    # # cnt = numpy.array([[x1,y1],[x3,y3],[x4,y4],[x2,y2]]) # 必须是array数组的形式
    # union_poly = np.array(union_poly, dtype=np.float32)
    # rect = cv2.minAreaRect(union_poly) # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    # cen, bbox ,theta = rect
    # bbox_w = bbox[0]
    # bbox_h = bbox[1]
    # # cen_x, cen_y, bbox_w, bbox_h, theta = rect
    # outer_diag = bbox_w * bbox_w + bbox_h * bbox_h
    # # print('w:', bbox_w)
    # # print('h',bbox_h)
    # # print('outer_diag:', outer_diag)
    #
    #
    # # minx = np.min(np.hstack((a[:, 0],b[:, 0])))
    # # miny = np.min(np.hstack((a[:, 1],b[:, 1])))
    # # maxy = np.max(np.hstack((a[:, 1],b[:, 1])))
    # # maxx = np.max(np.hstack((a[:, 0],b[:, 0])))
    # # print(minx,miny,maxx,maxy)
    # # outer_diag = (maxx - minx) **2 + (maxy - miny) **2
    # # print('second:', outer_diag)
    #
    #
    #
    # inter_diag = (cent_ax - cent_bx)**2 + (cent_ay + cent_by)**2
    #
    # dious = iou - (inter_diag) / outer_diag
    # # print('iou:',iou)
    # # print('diag:',(inter_diag) / outer_diag)
    # dious = torch.tensor(dious)
    # dious = torch.clamp(dious,min=-1.0,max = 1.0)
    # # if exchange:
    # #     dious = dious.T
    return iou

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return 0.



class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()

            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')

            return loss
        else:
            return 0.

class WHOffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(WHOffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            p = pred.masked_select(mask)
            t = target.masked_select(mask)


            l = len(t) / 10
                # avlist = []
                # bvlist = []
                # athetalist = []
                # bthetalist = []
                # print(l)
            loss = 0

            for i in range(int(l)):
                with torch.no_grad():
                    a = p[i * 10: (i + 1) * 10]
                    b = t[i * 10: (i + 1) * 10]
                    w = torch.pow((a[2] - a[6]), 2) + torch.pow((a[3] - a[7]),2)
                    h = torch.pow((a[0] - a[4]), 2) + torch.pow((a[1] - a[5]),2)
                    wt = torch.pow((b[2] - b[6]), 2) + torch.pow((b[3] - a[7]),2)
                    ht = torch.pow((b[0] - b[4]), 2) + torch.pow((b[1] - b[5]),2)
                    th = (a[3] - a[7]) / (a[2] - a[6])
                    tth = (b[3] - b[7]) / (b[2] - b[6])

                    v = (4 / (math.pi ** 2)) * (torch.pow((torch.atan(wt / ht) - torch.atan(w / h)), 2) + torch.pow((torch.atan(th) - torch.atan(tth)), 2))
                    # v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(b[8] / b[9]) - torch.atan(a[8] / a[9])), 2)
                    # print('v',v)
                loss = loss + v
            # print('loss', loss)
            #         avlist.append(av)
            #         bvlist.append(bv)
            #         if a[0] == 0:
            #             atheta = 0
            #
            #         else:
            #             th = a[1] / a[0]
            #             atheta = math.atan(th)
            #         athetalist.append(atheta)
            #         if b[0] == 0:
            #             btheta = 0
            #         else:
            #             th = a[1] / a[0]
            #             btheta = math.atan(th)
            #         bthetalist.append(btheta)
            #     avlist = torch.Tensor(avlist)
            #     bvlist = torch.Tensor(bvlist)
            #     athetalist = torch.Tensor(athetalist)
            #     bthetalist = torch.Tensor(bthetalist)
            #
            #
            # # loss = F.smooth_l1_loss(p,t,reduction='mean')
            # loss1 = F.smooth_l1_loss(avlist,bvlist,reduction='mean')
            # loss2 = F.smooth_l1_loss(athetalist, bthetalist, reduction='mean')


            return loss
        else:
            return 0.


# class IouLoss(nn.Module):
#     def __init__(self):
#         super(IouLoss, self).__init__()
#
#     def _gather_feat(self, feat, ind, mask=None):
#         dim = feat.size(2)
#         ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
#         feat = feat.gather(1, ind)
#         if mask is not None:
#             mask = mask.unsqueeze(2).expand_as(feat)
#             feat = feat[mask]
#             feat = feat.view(-1, dim)
#         return feat
#
#     def _tranpose_and_gather_feat(self, feat, ind):
#         feat = feat.permute(0, 2, 3, 1).contiguous()
#         feat = feat.view(feat.size(0), -1, feat.size(3))
#         feat = self._gather_feat(feat, ind)
#         return feat
#
#     def forward(self, output, mask, ind, target):
#         # torch.Size([1, 2, 152, 152])
#         # torch.Size([1, 500])
#         # torch.Size([1, 500])
#         # torch.Size([1, 500, 2])
#         pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
#         a = []
#         b = []
#
#         if mask.sum():
#             mask = mask.unsqueeze(2).expand_as(pred).bool()
#             p = pred.masked_select(mask)
#             g = target.masked_select(mask)
#             l = list(p.size())
#             l = l[0]/10
#             loss = 0
#             for i in range(int(l)):
#                 a = p[i * 10 : (i+1) * 10 - 2]
#                 b = g[i * 10 : (i+1) * 10 - 2]
#                 pred_box = cal_box(a)
#                 target_box = cal_box(b)
#                 iou_loss = iouloss(pred_box,target_box)
#                 loss = loss + iou_loss
#             # print('p size: {}.' .format(p.size()))
#             # print('g size: {}.' .format(g.size()))
#
#             return loss
#         else:
#             return 0.


class IouLoss(nn.Module):
    def __init__(self):
        super(IouLoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, pred, gt):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        output = pred['wh']
        ind = gt['ind']
        mask = gt['reg_mask']
        target = gt['wh']

        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        a = []
        b = []

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            p = pred.masked_select(mask)
            g = target.masked_select(mask)
            l = list(p.size())
            l = l[0]/10
            loss = 0
            for i in range(int(l)):
                a = p[i * 10 : (i+1) * 10 - 2]
                b = g[i * 10 : (i+1) * 10 - 2]
                pred_box = cal_box(a)
                target_box = cal_box(b)
                with torch.no_grad():
                    # arctan = torch.atan(wt / ht) - torch.atan(w / h)
                    w = (torch.pow((a[2] - a[6]), 2) + torch.pow((a[3] - a[7]), 2)) ** 0.5
                    h = (torch.pow((a[0] - a[4]), 2) + torch.pow((a[1] - a[5]), 2)) ** 0.5
                    wt = (torch.pow((b[2] - b[6]), 2) + torch.pow((b[3] - a[7]), 2)) ** 0.5
                    ht = (torch.pow((b[0] - b[4]), 2) + torch.pow((b[1] - b[5]), 2)) ** 0.5

                    # w = torch.pow((a[2] - a[6]), 2) + torch.pow((a[3] - a[7]), 2)
                    # h = torch.pow((a[0] - a[4]), 2) + torch.pow((a[1] - a[5]), 2)
                    # wt = torch.pow((b[2] - b[6]), 2) + torch.pow((b[3] - a[7]), 2)
                    # ht = torch.pow((b[0] - b[4]), 2) + torch.pow((b[1] - b[5]), 2)

                    th1 = (a[3] - a[7]) / (a[2] - a[6])
                    tth1 = (b[3] - b[7]) / (b[2] - b[6])
                    th = (a[1] - a[5]) / (a[0] - a[4])
                    tth = (b[1] - b[5]) / (b[0] - b[4])
                    n = torch.pow((torch.atan(th) - torch.atan(tth)), 2)
                    n1 = torch.pow((torch.atan(th1) - torch.atan(tth1)), 2)
                    if n1 < n :
                        n = n1
                    # v = (4 / (math.pi ** 2)) * ((torch.pow((torch.atan(wt / ht) - torch.atan(w / h)), 2) + torch.pow((torch.atan(th) - torch.atan(tth)), 2)))
                    v = (4 / (math.pi ** 2)) * (torch.pow((torch.atan(wt / ht) - torch.atan(w / h)), 2))
                    s = (4 / (math.pi ** 2)) * n
                    w_temp = 2 * w
                # ar = (8 / (math.pi ** 2)) * arctan * ((w - w_temp) * h)
                # print(len(pred_box))
                # print(len(target_box))
                iou = iouloss(pred_box,target_box)
                alpha = (v + s)/(1-iou+v+s)
                loss = alpha * (v + 0.7 * s)
                # loss = 0.8 * s
                # loss = iou - alpha * v - s
                # loss = torch.clamp(loss, min=-1.0, max=1.0)
                # loss = 1-loss
            return loss
        else:
            return 0.
        # print(len(gg))
        # l = list(gg.size())
        # if mask.sum():
        #     mask = mask.unsqueeze(2).expand_as(tt).bool()
        #     pp = pp.masked_select(mask)
        #     tt = tt.masked_select(mask)
        #     l = len(tt) / 10
        #     loss = 0
        #     for i in range(int(l)):
        #         a = pp[i * 10 + 2: (i + 1) * 10 ]
        #         b = tt[i * 10 + 2: (i + 1) * 10 ]
        #         pred_box = cal_box(a)
        #         target_box = cal_box(b)
        #         iou_loss = iouloss(pred_box, target_box)
        #         loss = loss + iou_loss
        #
        #     # print('p size: {}.' .format(p.size()))
        #     # print('g size: {}.' .format(g.size()))
        #
        #     return loss
        # else:
        #     return 0.


class DiouLoss(nn.Module):
    def __init__(self):
        super(DiouLoss, self).__init__()
        self.K = 500
        self.conf_thresh = 0.18
        self.num_classes = 1

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def ctdet_decode(self, pr_decs,gt):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']

        gt_heat = gt['hm']
        gt_wh = gt['wh']
        gt_reg = gt['reg']
        gt_cls_theta = gt['cls_theta']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        gtscores, gtinds, gtclses, gtys, gtxs = self._topk(gt_heat)


        inds = gt['ind']
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        gtxs = gtxs.view(batch, self.K, 1) + gt_reg[:, :, 0:1]
        gtys = gtys.view(batch, self.K, 1) + gt_reg[:, :, 1:2]


        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)

        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        gtmask = (gt_cls_theta>0.8).float().view(batch, self.K, 1)
        #
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)

        gttt_x = (gtxs+gt_wh[..., 0:1])*gtmask + (gtxs)*(1.-gtmask)
        gttt_y = (gtys+gt_wh[..., 1:2])*gtmask + (gtys-gt_wh[..., 9:10]/2)*(1.-gtmask)
        gtrr_x = (gtxs+gt_wh[..., 2:3])*gtmask + (gtxs+gt_wh[..., 8:9]/2)*(1.-gtmask)
        gtrr_y = (gtys+gt_wh[..., 3:4])*gtmask + (gtys)*(1.-gtmask)
        gtbb_x = (gtxs+gt_wh[..., 4:5])*gtmask + (gtxs)*(1.-gtmask)
        gtbb_y = (gtys+gt_wh[..., 5:6])*gtmask + (gtys+gt_wh[..., 9:10]/2)*(1.-gtmask)
        gtll_x = (gtxs+gt_wh[..., 6:7])*gtmask + (gtxs-gt_wh[..., 8:9]/2)*(1.-gtmask)
        gtll_y = (gtys+gt_wh[..., 7:8])*gtmask + (gtys)*(1.-gtmask)
        #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y],
                               dim=2)

        gtbox = torch.cat([gtxs,                      # cen_x
                           gtys,                      # cen_y
                           gttt_x,
                           gttt_y,
                           gtrr_x,
                           gtrr_y,
                           gtbb_x,
                           gtbb_y,
                           gtll_x,
                           gtll_y],
                           dim=2)

        return detections, gtbox

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, pred, gt):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        mask = gt['reg_mask']
        pp,tt = self.ctdet_decode(pred,gt)

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(tt).bool()
            tt = tt.masked_select(mask)
            pp = pp.masked_select(mask)
            # l = len(tt) / 10
            # loss = 0
            # for i in range(int(l)):
            #     a = pp[i * 10  : (i+1) * 10]
            #     b = tt[i * 10  : (i+1) * 10]
            #     pred_box = cal_box(a)
            #     target_box = cal_box(b)
            #     Diou_loss = 1 - diou(pred_box,target_box)
            #     loss += Diou_loss
            # print('p size: {}.' .format(p.size()))
            # print('g size: {}.' .format(g.size()))
            loss = F.smooth_l1_loss(pp, tt, reduction='mean')
            return loss
        else:
            return 0.

class PiouLoss(nn.Module):
    def __init__(self):
        super(PiouLoss, self).__init__()
        self.K = 500
        self.conf_thresh = 0.18
        self.num_classes = 1

    def _topk(self, scores):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), self.K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds // width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), self.K)
        topk_clses = (topk_ind // self.K).int()
        topk_inds = self._gather_feat( topk_inds.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, self.K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, self.K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def ctdet_decode(self, pr_decs,gt):
        heat = pr_decs['hm']
        wh = pr_decs['wh']
        reg = pr_decs['reg']
        cls_theta = pr_decs['cls_theta']

        gt_heat = gt['hm']
        gt_wh = gt['wh']
        gt_reg = gt['reg']
        gt_cls_theta = gt['cls_theta']

        batch, c, height, width = heat.size()
        heat = self._nms(heat)

        scores, inds, clses, ys, xs = self._topk(heat)
        gtscores, gtinds, gtclses, gtys, gtxs = self._topk(gt_heat)


        inds = gt['ind']
        reg = self._tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, self.K, 2)
        xs = xs.view(batch, self.K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, self.K, 1) + reg[:, :, 1:2]
        gtxs = gtxs.view(batch, self.K, 1) + gt_reg[:, :, 0:1]
        gtys = gtys.view(batch, self.K, 1) + gt_reg[:, :, 1:2]


        wh = self._tranpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, self.K, 10)
        # add
        cls_theta = self._tranpose_and_gather_feat(cls_theta, inds)
        cls_theta = cls_theta.view(batch, self.K, 1)

        mask = (cls_theta>0.8).float().view(batch, self.K, 1)
        gtmask = (gt_cls_theta>0.8).float().view(batch, self.K, 1)
        #
        tt_x = (xs+wh[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+wh[..., 1:2])*mask + (ys-wh[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+wh[..., 2:3])*mask + (xs+wh[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+wh[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+wh[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+wh[..., 5:6])*mask + (ys+wh[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+wh[..., 6:7])*mask + (xs-wh[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+wh[..., 7:8])*mask + (ys)*(1.-mask)

        gttt_x = (gtxs+gt_wh[..., 0:1])*gtmask + (gtxs)*(1.-gtmask)
        gttt_y = (gtys+gt_wh[..., 1:2])*gtmask + (gtys-gt_wh[..., 9:10]/2)*(1.-gtmask)
        gtrr_x = (gtxs+gt_wh[..., 2:3])*gtmask + (gtxs+gt_wh[..., 8:9]/2)*(1.-gtmask)
        gtrr_y = (gtys+gt_wh[..., 3:4])*gtmask + (gtys)*(1.-gtmask)
        gtbb_x = (gtxs+gt_wh[..., 4:5])*gtmask + (gtxs)*(1.-gtmask)
        gtbb_y = (gtys+gt_wh[..., 5:6])*gtmask + (gtys+gt_wh[..., 9:10]/2)*(1.-gtmask)
        gtll_x = (gtxs+gt_wh[..., 6:7])*gtmask + (gtxs-gt_wh[..., 8:9]/2)*(1.-gtmask)
        gtll_y = (gtys+gt_wh[..., 7:8])*gtmask + (gtys)*(1.-gtmask)
        #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y],
                               dim=2)

        gtbox = torch.cat([gtxs,                      # cen_x
                           gtys,                      # cen_y
                           gttt_x,
                           gttt_y,
                           gtrr_x,
                           gtrr_y,
                           gtbb_x,
                           gtbb_y,
                           gtll_x,
                           gtll_y],
                           dim=2)

        return detections, gtbox

    def _nms(self, heat, kernel=3):
        hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=(kernel - 1) // 2)
        keep = (hmax == heat).float()
        return heat * keep

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, pred, gt):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        mask = gt['reg_mask']
        pp,tt = self.ctdet_decode(pred,gt)

        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(tt).bool()
            tt = tt.masked_select(mask)
            pp = pp.masked_select(mask)
            loc_pxy = []
            loc_pwh = []
            loc_pa = []
            loc_txy = []
            loc_twh = []
            loc_ta = []
            l = len(tt) / 10
            # print(l)
            loss = 0
            for i in range(int(l)):
                a = pp[i * 10: (i + 1) * 10]
                b = tt[i * 10: (i + 1) * 10]
                listpxy, listpwh, ptheat = cal_boxwh(a)
                listtxy, listtwh, ttheat = cal_boxwh(b)
                loc_pxy.append(listpxy)
                loc_pwh.append(listpwh)
                loc_pa.append(ptheat)
                loc_txy.append(listtxy)
                loc_twh.append(listtwh)
                loc_ta.append(ttheat)
            loc_pxy = torch.tensor(loc_pxy)
            loc_pwh = torch.tensor(loc_pwh)
            loc_pa = torch.tensor(loc_pa)
            loc_pa = loc_pa.unsqueeze(-1)
            # print(loc_pwh)
            loc_txy = torch.tensor(loc_txy)
            loc_twh = torch.tensor(loc_twh)
            loc_ta = torch.tensor(loc_ta)
            loc_ta = loc_ta.unsqueeze(-1)
            # print(loc_pxy.shape)
            # print(loc_pwh.shape)
            # print(loc_pa.shape)
            loc_p = torch.cat((loc_pxy, loc_pwh, loc_pa), -1)
            loc_t = torch.cat((loc_txy, loc_twh, loc_ta), -1)

            grid_xy = template_pixels(512, 512)
            grid_x = template_w_pixels(512)
            num = loc_p.size(0)
            dim = grid_xy.size(0)
            loc_p = loc_p.cuda()
            loc_t = loc_t.cuda()
            grid_xy = grid_xy.cuda()
            loc_p = Variable(loc_p, requires_grad=True)

            piou_loss, pious_big = PIoU(loc_p, loc_t.data, grid_xy.data, 10)

            return piou_loss
        else:
            return 0.

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      # print('pred :' ,pred.shape)
      # print('gt size :' ,gt.shape)
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss




def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        # self.L_wh1 = WHOffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        self.L_iou = IouLoss()
        self.L_Diou = DiouLoss()
        self.L_Piou = PiouLoss()

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        # wh1_loss = self.L_wh1(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        # iou_loss = self.L_Diou(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        # iou_loss = self.L_iou(pr_decs, gt_batch,pp,tt)
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])

        # if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
        #     print('hm loss is {}'.format(hm_loss))
        #     print('wh loss is {}'.format(wh_loss))
        #     print('off loss is {}'.format(off_loss))
        #     print('iou loss is {}'.format(iou_loss))

        # print(hm_loss)
        # print(wh_loss)
        # print(off_loss)
        # print(cls_theta_loss)
        # print('-----------------')
        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss
        # loss =  hm_loss + wh_loss + off_loss + cls_theta_loss  +iou_loss
        # loss = hm_loss  + off_loss + cls_theta_loss + 10 * iou_loss
        # print('hm loss is {}'.format(hm_loss))
        # print('wh loss is {}'.format(wh_loss))
        # print('off loss is {}'.format(off_loss))
        # print('iou loss is {}'.format(iou_loss))
        return loss
        # return loss, hm_loss, wh_loss, off_loss, cls_theta_loss ,iou_loss




