import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration, double_threshold_iteration_fast
from utils.metrics import AverageMeter, get_metrics, count_connect_component
import ttach as tta


class Tester(Trainer):
    def __init__(self, model, loss, CFG, test_loader, dataset_path, show=False):
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show

        if self.show:
            dir_exists("save_picture")
            remove_files("save_picture")

        cudnn.benchmark = True

    def test(self):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = img.cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                pre = self.model(img)
                loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)

                # ✅ Always define H, W
                if self.dataset_path.endswith("DRIVE"):
                    H, W = 584, 565
                elif self.dataset_path.endswith("CHASEDB1"):
                    H, W = 960, 999
                elif self.dataset_path.endswith("DCA1"):
                    H, W = 300, 300
                else:
                    H, W = img.shape[2], img.shape[3]

                if not self.dataset_path.endswith("CHUAC"):
                    img = TF.crop(img, 0, 0, H, W)
                    gt = TF.crop(gt, 0, 0, H, W)
                    pre = TF.crop(pre, 0, 0, H, W)

                img = img[0, 0, ...]
                gt = gt[0, 0, ...]
                pre = pre[0, 0, ...]

                # -------------------------------------------------------------------------------------------------------------------------------------------------------------
                #  <--- START OF NEW CODE: POST-PROCESSING --->
                # -------------------------------------------------------------------------------------------------------------------------------------------------------------

                # Convert PyTorch tensor to NumPy array and apply threshold
                binary_mask = (torch.sigmoid(pre).cpu().detach().numpy() > self.CFG.threshold).astype(np.uint8)

                # Define a structuring element (kernel)
                kernel_size = 5
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

                # Apply morphological closing: dilation followed by erosion
                processed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

                # -------------------------------------------------------------------------------------------------------------------------------------------------------------
                #  <--- END OF NEW CODE: POST-PROCESSING --->
                # -------------------------------------------------------------------------------------------------------------------------------------------------------------


                if self.show:
                    cv2.imwrite(f"save_picture/img{i}.png", np.uint8(img.cpu().numpy() * 255))
                    cv2.imwrite(f"save_picture/gt{i}.png", np.uint8(gt.cpu().numpy() * 255))
                    cv2.imwrite(f"save_picture/pre{i}.png", np.uint8(processed_mask * 255))
                    cv2.imwrite(f"save_picture/pre_b{i}_processed.png", np.uint8(processed_mask * 255))

                if self.CFG.DTI:
                    if self.CFG.fast_DTI:
                        pre_DTI = double_threshold_iteration_fast(
                            i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    else:
                        pre_DTI = double_threshold_iteration(
                            i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(*get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(*get_metrics(pre, gt, threshold=self.CFG.threshold, predict_b=processed_mask).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(processed_mask, gt))

                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} '
                    'Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(),
                        self.batch_time.average, self.data_time.average))
                tic = time.time()

        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average:.4f}')
        logger.info(f'     loss:  {self.total_loss.average:.4f}')
        if self.CFG.CCC:
         logger.info(f'     CCC:  {self.CCC.average:.4f}')
        for k, v in self._metrics_ave().items():
         logger.info(f'{str(k):5s}: {v:.4f}')
