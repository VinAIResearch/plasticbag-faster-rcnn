#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Vinh Nguyen, based on demo.py
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__', 'person', 'bicycle', 'car', 'motorcycle', 
           'airplane', 'bus', 'train', 'truck', 'boat', 
           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
           'bird', 'cat', 'dog', 'horse', 'sheep', 
           'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 
           'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 
           'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 
           'spoon', 'bowl', 'banana', 'apple', 'sandwich', 
           'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
           'donut', 'cake', 'chair', 'couch', 'potted plant', 
           'bed', 'dining table', 'toilet', 'tv', 'laptop', 
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 
           'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
           'toothbrush')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_1190000.ckpt',)}
DATASETS= {'coco': ('coco_2014_train+coco_2014_valminusminival',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""

    PERSON_THRESH = 0.8

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        text = '{:s} {:.2f}'.format(class_name, score)
        if class_name == 'person' and score < PERSON_THRESH:
            continue
        cv2.rectangle(im, 
                    (bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                    ((0, 255, 0) if class_name == 'person' else (43,0,255)), 
                    2)
        cv2.putText(im, 
                    text, 
                    (int(bbox[0]) + 1, int(bbox[1]) + 10), 
                    cv2.FONT_HERSHEY_PLAIN, 
                    1, 
                    ((0, 255, 0) if class_name == 'person' else (43,0,255)),
                    lineType=cv2.LINE_AA)

def demo(sess, net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        if cls != 'person' and cls != 'backpack' and cls != 'handbag' and cls != 'suitcase':
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    
    # Display the image with bouding boxes
    cv2.imshow('', im)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [coco]',
                        choices=DATASETS.keys(), default='coco')
    args = parser.parse_args()

    return args

def write_output():
    ''' Capture video '''
    cap = cv2.VideoCapture(os.path.join(cfg.DATA_DIR, 'demo', 'pets2006_1.avi'))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video
    out = cv2.VideoWriter(os.path.join(cfg.DATA_DIR, 'demo', 'output_pets2006_1.avi'), cv2.VideoWriter_fourcc(*'XVID'), 10, (width, height))

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == 0: break
        # Display the frame with bounding boxes
        demo(sess, net, frame)
        # Write on the output video
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture and video write objects
    cap.release()
    out.release()

    # Close all the frames
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[4, 8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    write_output()

