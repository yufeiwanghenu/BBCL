import torch
import torch.nn as nn
import os
import numpy as np
import loss
import cv2
import func_utils
from data import dataset_hrsc
from data.hrsc_evaluation_task1 import voc_eval


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict

class TrainModule(object):
    def \
            __init__(self, dataset, num_classes, model, decoder, down_ratio):
        torch.manual_seed(317)
        self.dataset = dataset
        self.dataset_phase = {'dota': ['train'],
                              'hrsc': ['train', 'test']}
        self.num_classes = num_classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.decoder = decoder
        self.down_ratio = down_ratio


    def save_model(self, path, epoch, model, optimizer):
        if isinstance(model, torch.nn.DataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss
        }, path)

    def load_model(self, model, optimizer, resume, strict=True):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['model_state_dict']
        state_dict = {}
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
        model_state_dict = model.state_dict()
        if not strict:
            for k in state_dict:
                if k in model_state_dict:
                    if state_dict[k].shape != model_state_dict[k].shape:
                        print('Skip loading parameter {}, required shape{}, ' \
                              'loaded shape{}.'.format(k, model_state_dict[k].shape, state_dict[k].shape))
                        state_dict[k] = model_state_dict[k]
                else:
                    print('Drop parameter {}.'.format(k))
            for k in model_state_dict:
                if not (k in state_dict):
                    print('No param {}.'.format(k))
                    state_dict[k] = model_state_dict[k]
        model.load_state_dict(state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        epoch = checkpoint['epoch']
        # loss = checkpoint['loss']
        return model, optimizer, epoch

    def train_network(self, args):

        self.optimizer = torch.optim.Adam(self.model.parameters(), args.init_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.96, last_epoch=-1)
        save_path = 'weights_'+args.dataset
        start_epoch = 1
        
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if args.ngpus>1:
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
                self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

        criterion = loss.LossAll()
        print('Setting up data...')


        dataset_module = self.dataset[args.dataset]

        dsets = {x: dataset_module(data_dir=args.data_dir,
                                   phase=x,
                                   input_h=args.input_h,
                                   input_w=args.input_w,
                                   down_ratio=self.down_ratio)
                 for x in self.dataset_phase[args.dataset]}

        dsets_loader = {}

        dsets_loader['train'] = torch.utils.data.DataLoader(dsets['train'],
                                                           batch_size=args.batch_size,
                                                           shuffle=True,
                                                           num_workers=args.num_workers,
                                                           pin_memory=True,
                                                           drop_last=True,
                                                           collate_fn=collater)

        print('Starting training...')
        train_loss = []
        ap_list = []

        # # 加载模型，继续训练
        # log_dir = 'E:\wyf\BBAVectors\weights_dota\model_15.pth'
        # if os.path.exists(log_dir):
        #     checkpoint = torch.load(log_dir)
        #     self.model.load_state_dict(checkpoint['model_state_dict'])
        #     self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     start_epoch = checkpoint['epoch']
        #     print('加载 epoch {} 成功！'.format(start_epoch))

        for epoch in range(start_epoch, args.num_epoch+1):
            print('-'*10)
            print('Epoch: {}/{} '.format(epoch, args.num_epoch))
            epoch_loss = self.run_epoch(phase='train',
                                        data_loader=dsets_loader['train'],
                                        criterion=criterion)
            # print(type(dsets_loader['train']))
            train_loss.append(epoch_loss)
            self.scheduler.step(epoch)

            # np.savetxt(os.path.join(save_path, 'loss.txt'), train_loss, fmt='%.6f')

            if epoch % 3 == 0 or epoch > 40:
                self.save_model(os.path.join(save_path, 'model_{}.pth'.format(epoch)),
                                epoch,
                                self.model,
                                self.optimizer)

            if 'test' in self.dataset_phase[args.dataset] and (epoch % 5 == 0 or epoch > 40):

                mAP = self.dec_eval(args, dsets['test'])
                ap_list.append(mAP)
                print(ap_list)
                np.savetxt(os.path.join(save_path, 'v+05saplist.txt'), ap_list, fmt='%s')
                np.savetxt(os.path.join(save_path, 'ap.txt'), ap_list, fmt='%.6f')

            self.save_model(os.path.join(save_path, 'model_last.pth'),
                            epoch,
                            self.model,
                            self.optimizer)

    def run_epoch(self, phase, data_loader, criterion):
        if phase == 'train':
            self.model.train()
        else:
            self.model.eval()
        running_loss = 0.
        hm_loss1= 0.
        wh_loss1= 0.
        wh_loss2= 0.
        off_loss1= 0.
        cls_theta_loss1= 0.
        Diou_loss1= 0.
        iou_loss1 = 0.
        for data_dict in data_loader:
            for name in data_dict:
                data_dict[name] = data_dict[name].to(device=self.device, non_blocking=True)
            if phase == 'train':
                self.optimizer.zero_grad()
                with torch.enable_grad():
                    torch.cuda.synchronize(self.device)
                    pr_decs = self.model(data_dict['input'])
                    # predictions,gt = self.tdecoder.ctdet_decode(pr_decs,data_dict)
                    # print('pred shape:',predictions.shape)
                    # print('gt shape:', gt.shape)
                    # print('gg shape:', gg.shape)
                    # print('tt shape:', tt.shape)
                    # gt = self.tdecoder.ctdet_decode(data_dict,data_dict)
                    loss = criterion(pr_decs, data_dict)

                    # loss, hm_loss, off_loss, cls_theta_loss, iou_loss = criterion(pr_decs, data_dict)

                    # loss,hm_loss,wh_loss,off_loss,cls_theta_loss,iou_loss= criterion(pr_decs, data_dict)

                    # print(pr_decs.keys(),data_dict.keys())
                    # pwh = pr_decs['wh']
                    # twh = data_dict['wh']
                    # phm = pr_decs['reg']
                    # thm = data_dict['reg']
                    # regmask = data_dict['reg_mask']
                    # print('pwh size :' , pwh.shape)
                    # print('twh size :', twh.shape)
                    # print('preg size :', phm.shape)
                    # print('treg size :', thm.shape)
                    # print('regmask size :', regmask.shape)
                    # print(twh.shape)
                    # print(twh[1, : ,1,1])
                    # print(pr_decs['hm'])
                    # print(data_dict['hm'])
                    loss.backward()
                    self.optimizer.step()
            else:
                with torch.no_grad():
                    pr_decs = self.model(data_dict['input'])
                    # loss = criterion(pr_decs, data_dict)
                    loss ,hm_loss,wh_loss,off_loss,cls_theta_loss,wh1_loss,iou_loss= criterion(pr_decs, data_dict)

            running_loss += loss.item()


            # hm_loss1 += hm_loss
            # wh_loss1 += wh_loss
            # off_loss1 += off_loss
            # cls_theta_loss1 += cls_theta_loss
            # # wh_loss2 += wh1_loss
            # iou_loss1 += iou_loss
            # Diou_loss1 += Diou_loss
            # print('wh_loss2',wh_loss2)

        epoch_loss = running_loss / len(data_loader)


        # hm_loss2 = hm_loss1 / len(data_loader)
        # wh_loss1 = wh_loss1 / len(data_loader)
        # off_loss2 = off_loss1 / len(data_loader)
        # cls_theta_loss2 = cls_theta_loss1 / len(data_loader)
        # # wh_loss2 = wh_loss2 / len(data_loader)
        # iou_loss2 = iou_loss1 / len(data_loader)
        # Diou_loss2 = Diou_loss1 / len(data_loader)


        print('-----------------')
        print('{} loss: {}'.format(phase, epoch_loss))

        # print('hm_loss loss: {}'.format(hm_loss2))
        # print('wh_loss loss: {}'.format(wh_loss1))
        # print('off_loss loss: {}'.format(off_loss2))
        # print('cls_theta_loss loss: {}'.format(cls_theta_loss2))
        # # print('wh_loss2 loss: {}'.format(wh_loss2))
        # print('iou_loss loss: {}'.format(iou_loss2))
        # print('Diou_loss loss: {}'.format(Diou_loss2))

        return epoch_loss


    def dec_eval(self, args, dsets):
        result_path = 'result_'+args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        self.model.eval()
        func_utils.write_results(args,
                                 self.model,dsets,
                                 self.down_ratio,
                                 self.device,
                                 self.decoder,
                                 result_path)
        ap = dsets.dec_evaluation(result_path)

        # detpath = os.path.join(result_path, 'Task1_{}.txt')
        # # label_path =
        # annopath = os.path.join(self.label_path,
        #                         '{}.xml')  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
        # imagesetfile = os.path.join(self.data_dir, 'test.txt')
        # classaps = []
        # map = 0
        # for classname in self.category:
        #     if classname == 'background':
        #         continue
        #     print('classname:', classname)
        #     rec, prec, ap = voc_eval(detpath,
        #                              annopath,
        #                              imagesetfile,
        #                              classname,
        #                              ovthresh=0.5,
        #                              use_07_metric=True)
        #     map = map + ap
        #     # print('rec: ', rec, 'prec: ', prec, 'ap: ', ap)
        #     print('{}:{} '.format(classname, ap * 100))
        #     classaps.append(ap)
        #     # umcomment to show p-r curve of each category
        #     # plt.figure(figsize=(8,4))
        #     # plt.xlabel('recall')
        #     # plt.ylabel('precision')
        #     # plt.plot(rec, prec)
        # # plt.show()
        # map = map / len(self.category)
        # print('map:', map * 100)

        return ap


