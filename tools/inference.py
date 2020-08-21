import os
import sys
sys.path.append('/app/')

from tools.parse_arg_test import TestOptions
from lib.models.diverse_depth_model import RelDepthModel
from lib.utils.net_tools import load_ckpt
import torch
import os
import numpy as np
from lib.core.config import cfg, merge_cfg_from_file
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

import cv2
import json

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W]
    """
    img = np.transpose(img, (2, 0, 1))
    img = img[::-1, :, :]
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS, cfg.DATASET.RGB_PIXEL_VARS)(img)
    return img

if __name__ == '__main__':
    test_args = TestOptions().parse()
    test_args.thread = 1
    test_args.batchsize = 1
    test_args.load_ckpt = "/app/model.pth"
    merge_cfg_from_file(test_args)
    
    # load model
    model = RelDepthModel()

    model.eval()

    # load checkpoint
    if test_args.load_ckpt:
        load_ckpt(test_args, model)
    
    model.cuda()
    model = torch.nn.DataParallel(model)

    with torch.no_grad():
        rgb = cv2.imread('/app/unnamed.jpg')


        img_torch = scale_torch(rgb, 255)
        img_torch = img_torch[None, :, :, :].cuda()


        import time
        start_time = time.time()

        pred_depth, _ = model.module.depth_model(img_torch)

        delta_time = (time.time() - start_time) 
        print("Model execution took %f s" % delta_time)

        #pred_depth = torch.nn.functional.tanh(pred_depth) + 1

        pred_depth = pred_depth.cpu().numpy().squeeze()
        
        #pred_depth_metric = recover_metric_depth(pred_depth, depth)

        # smoothed_criteria = evaluate_rel_err(pred_depth_metric, depth, smoothed_criteria, scale=1.0)

        plt.imshow(pred_depth,  cmap='hot')
        plt.show()
