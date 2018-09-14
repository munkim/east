import os, argparse, time
import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn
from mxnet import gluon
from model import FCN
import train_utils
import time

import cv2
import matplotlib.pyplot as plt


class ScoreLoss(gluon.loss.Loss):
    def __init__(self, eps=1e-5, **kwargs):
        super(ScoreLoss, self).__init__(None, 0, **kwargs)
        self.eps = eps

    def hybrid_forward(self, F_score_true, F_score_pred):
        # Dice Loss Function (Better than Balanced Cross Entropy)
        intersection = mx.nd.sum(F_score_true * F_score_pred)
        union = mx.nd.sum(F_score_true) + mx.nd.sum(F_score_pred) + self.eps
        loss = 1. - (2 * intersection / union)

        return loss


class GeometryLoss(gluon.loss.Loss):
    def __init__(self, lambda_value=20, **kwargs):
        super(GeometryLoss, self).__init__(None, 0, **kwargs)
        self.lambda_value = lambda_value

    def hybrid_forward(self, F_geo_true, F_geo_pred):
        top_true, right_true, bottom_true, left_true, theta_true = nd.split(F_geo_true, axis=3, num_outputs=5)
        top_pred, right_pred, bottom_pred, left_pred, theta_pred = nd.split(F_geo_pred, axis=3, num_outputs=5)
        area_true = (top_true + bottom_true) * (right_true + left_true)
        area_pred = (top_pred + bottom_pred) * (right_pred + left_pred)
        w_union = mx.nd.minimum(right_true, right_pred) + mx.nd.minimum(left_true, left_pred)
        h_union = mx.nd.minimum(top_true, top_pred) + mx.nd.minimum(bottom_true, bottom_pred)
        area_intersect = w_union * h_union
        area_union = area_true + area_pred - area_intersect
        L_AABB = -nd.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - nd.cos(theta_pred - theta_true)
        L_geo = L_AABB + self.lambda_value * L_theta
        loss = mx.nd.sum(L_geo * F_geo_true)

        return loss


def get_iterators(args, data_shape, batch_size, resize_mode='force', mean_pixels=[123.68, 116.779, 103.939]):

    """
    Parameters:
    -----------
    data:
        dataset
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list
        mean values for red/green/blue
    """
    mean_pixels = [123.68, 116.779, 103.939]

    # train_iter = mx.io.ImageDetRecordIter(
    #     path_imgrec=args.path_train_rec,
    #     path_imgidx=args.path_train_idx,
    #     path_imglist=args.path_train_lst,
    #     batch_size=batch_size,
    #     data_shape=data_shape,
    #     label_width=-1,
    #     label_pad_width=-1,
    #     mean_r=mean_pixels[0],
    #     mean_g=mean_pixels[1],
    #     mean_b=mean_pixels[2],
    #     resize_mode=resize_mode,
    #     rand_crop_prob=0,
    #     shuffle=args.shuffle)

    train_iter = mx.image.ImageDetIter(
        path_imgrec=args.path_train_rec,
        path_imgidx=args.path_train_idx,
        path_imglist=args.path_train_lst,
        batch_size=batch_size,
        data_shape=data_shape,
        shuffle=False)

    valid_iter = mx.io.ImageDetRecordIter(
        path_imgrec=args.path_valid_rec,
        path_imgidx=args.path_valid_idx,
        path_imglist=args.path_valid_lst,
        batch_size=batch_size,
        data_shape=data_shape,
        label_width=-1,
        label_pad_width=-1,
        mean_r=mean_pixels[0],
        mean_g=mean_pixels[1],
        mean_b=mean_pixels[2],
        resize_mode=resize_mode,
        rand_crop_prob=0.5)

    return train_iter, valid_iter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-path_train_rec', default='./data/train.rec', type=str)
    parser.add_argument('-path_train_idx', default='./data/train.idx', type=str)
    parser.add_argument('-path_train_lst', default='./data/train.lst', type=str)
    parser.add_argument('-path_valid_rec', default='./data/valid.rec', type=str)
    parser.add_argument('-path_valid_idx', default='./data/valid.idx', type=str)
    parser.add_argument('-path_valid_lst', default='./data/valid.lst', type=str)
    parser.add_argument('-batch_size', default=32, type=int)
    parser.add_argument('-use_gpu', default=False, action='store_true')
    parser.add_argument('-shuffle', default=False, action='store_true')
    parser.add_argument('-save_path', default='./save')
    parser.add_argument('-lr_rate', default=0.00025, type=float)
    parser.add_argument('-num_epochs', default=150, type=int)
    args = parser.parse_args()

    model = FCN()
    ctx = mx.gpu() if args.use_gpu else mx.cpu()
    model.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
    model.collect_params().reset_ctx(ctx)
    criterion1 = ScoreLoss()
    criterion2 = GeometryLoss()

    data_shape = (3, 512, 512)
    train_data, test_data = get_iterators(args, data_shape, args.batch_size)

    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': .1})

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    log_file = open('%s/log.txt' % args.save_path, 'w')
    train_data.reset()

    for epoch in range(args.num_epochs):
        for i, batch in enumerate(train_data):
            images = batch.data[0].as_in_context(ctx)
            labels = batch.label[0].as_in_context(ctx)
            # print (batch)
            print (labels)
            exit()
            t = time.time()
            F_score_true, F_geo_true = train_utils.get_score_and_geo(images, labels, data_iter_type=train_data.__class__.__name__)


            with autograd.record():
                F_score_pred, F_geo_pred = model(images)
                score_loss = criterion1(F_score_pred, F_score_true)
                geo_loss = criterion2(F_geo_pred, F_geo_true)

            lambda_geo = 1  # Lambda_geo weighs the importance between two losses
            loss = score_loss + lambda_geo * geo_loss
            loss.backward()
            trainer.step(images.shape[0])

            train_loss_log = ('\nTrain Loss: %.4f' % loss)
            train_score_log = ('\nTrain F_Score and F_Geo: %.4f \t %.4f' % (F_score_pred, F_geo_pred))
            log_file.write(train_loss_log)
            log_file.write(train_score_log)
            print (train_loss_log)
            print (train_score_log)

        # To-do: Get Validation Score
        F_score_pred = 0
        F_geo_pred = 0
        validation_score_log = ('\n---- Validation F_Score and F_Geo: %.4f\n' % (F_score_pred, F_geo_pred))
        log_file.write(validation_score_log)
        log_file.flush()
        print (validation_score_log)


if __name__ == '__main__':
    main()

