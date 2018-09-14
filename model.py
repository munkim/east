import os, subprocess, time
from PIL import Image
from collections import namedtuple
import numpy as np

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet import ndarray as F

class Stem(nn.Block):
    def __init__(self, **kwargs):
        super(Stem, self).__init__(**kwargs)
        # Path to pre-trained model
        path_pretrained_model = './pretrained_resnext-101/'
        pretrained_model_name = 'resnext-101-64x4d'
        if not os.path.exists(path_pretrained_model):  # Download if the pre-trained model doesn't exist
            os.chmod('./download.sh', 0777)
            subprocess.call(['./download.sh'])

        # Get symbols and parameters of the pre-trained model
        sym, arg_params, aux_params = mx.model.load_checkpoint(path_pretrained_model + pretrained_model_name, 0)
        all_layers = sym.get_internals()  # Get internal symbols (layer-wise)
        stage1_sym = all_layers['stage1_unit3_relu_output']
        stage2_sym = all_layers['stage2_unit4_relu_output']
        stage3_sym = all_layers['stage3_unit23_relu_output']
        stage4_sym = all_layers['stage4_unit3_relu_output']
        self.Batch = namedtuple('Batch', ['data'])

        # Make symbols into modules
        self.stage1 = mx.mod.Module(symbol=stage1_sym, context=mx.cpu(), label_names=None)
        self.stage2 = mx.mod.Module(symbol=stage2_sym, context=mx.cpu(), label_names=None)
        self.stage3 = mx.mod.Module(symbol=stage3_sym, context=mx.cpu(), label_names=None)
        self.stage4 = mx.mod.Module(symbol=stage4_sym, context=mx.cpu(), label_names=None)

        self.stage1.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        self.stage2.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        self.stage3.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])
        self.stage4.bind(for_training=False, data_shapes=[('data', (1, 3, 224, 224))])

        self.stage1.set_params(arg_params, aux_params)
        self.stage2.set_params(arg_params, aux_params)
        self.stage3.set_params(arg_params, aux_params)
        self.stage4.set_params(arg_params, aux_params)

    def forward(self, inputs):
        self.stage1.forward(self.Batch([mx.nd.array(inputs)]))
        self.stage2.forward(self.Batch([mx.nd.array(inputs)]))
        self.stage3.forward(self.Batch([mx.nd.array(inputs)]))
        self.stage4.forward(self.Batch([mx.nd.array(inputs)]))

        conv1_features = self.stage1.get_outputs()[0]
        conv2_features = self.stage2.get_outputs()[0]
        conv3_features = self.stage3.get_outputs()[0]
        conv4_features = self.stage4.get_outputs()[0]
        stem_out = [conv1_features, conv2_features, conv3_features, conv4_features]

        return stem_out

class Branch(nn.Block):
    def __init__(self, **kwargs):
        super(Branch, self).__init__(**kwargs)
        with self.name_scope():
            # Branch 1
            self.unpool1 = nn.Conv2DTranspose(512, 2, strides=2, use_bias=False)  # Deconvolution
            self.unpool1.initialize('Bilinear')  # Initialize Upsampling (or Deconv) weights to Bilinear
            self.conv1_1 = nn.Conv2D(128, 1)
            self.conv1_2 = nn.Conv2D(128, 3, padding=1)

            # Branch 2
            self.unpool2 = nn.Conv2DTranspose(512, 2, strides=2, use_bias=False)  # Deconvolution
            self.unpool2.initialize('Bilinear')  # Initialize Upsampling (or Deconv) weights to Bilinear
            self.conv2_1 = nn.Conv2D(64, 1)
            self.conv2_2 = nn.Conv2D(64, 3, padding=1)

            # Branch 3
            self.unpool3 = nn.Conv2DTranspose(512, 2, strides=2, use_bias=False)  # Deconvolution
            self.unpool3.initialize('Bilinear')  # Initialize Upsampling (or Deconv) weights to Bilinear
            self.conv3_1 = nn.Conv2D(32, 1)
            self.conv3_2 = nn.Conv2D(32, 3, padding=1)

            self.output = nn.Conv2D(32, 3, padding=1)

    def forward(self, stem_out, *args):
        out = self.unpool1(stem_out[3])  # Branch 1
        out = F.concatenate([out, stem_out[2]], axis=1)
        out = self.conv1_1(out)
        out = self.conv1_2(out)

        out = self.unpool2(out)  # Branch 2
        out = F.concatenate([out, stem_out[1]], axis=1)
        out = self.conv2_1(out)
        out = self.conv2_2(out)

        out = self.unpool3(out)  # Branch 3
        out = F.concatenate([out, stem_out[0]], axis=1)
        out = self.conv3_1(out)
        out = self.conv3_2(out)

        out = self.output(out)  # Branch 4

        return out

class FCN(nn.Block):
    def __init__(self, **kwargs):
        super(FCN, self).__init__(**kwargs)
        self.stem = Stem()
        self.branch = Branch()
        self.scores = nn.Conv2D(1, 1)
        self.boxes = nn.Conv2D(4, 1)
        self.angles = nn.Conv2D(1, 1)

    def forward(self, inputs, *args):
        stem_out = self.stem(inputs)
        branch_out = self.branch(stem_out)

        # Score Map
        F_score = self.scores(branch_out)

        # Geometric Map (RBOX)
        boxes = self.boxes(branch_out)  # 4 channels of axis-aligned bounding box
        rot_angles = (self.angles(branch_out) - 0.5) * np.pi / 2  # 1 channel rotation angle, which is between [-45, 45]
        F_geo = F.concatenate([boxes, rot_angles], axis=1)

        return F_score, F_geo

def test():
    net = FCN()
    net.collect_params().initialize(mx.init.Xavier(), ctx=mx.cpu())
    sample_data = mx.nd.random_normal(shape=(10, 3, 512, 512))  # dummy data
    F_score, F_geo = net(sample_data)

    print ("\nOUTPUTS SHAPE")
    print (F_score.shape)
    print (F_geo.shape)

if __name__ == '__main__':
    test()