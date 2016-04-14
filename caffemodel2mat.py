import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
caffe_root = '/home/gavinpan/workspace/caffe-master/'
caffe_python = caffe_root+'python'
#print caffe_python
sys.path.insert(0, caffe_python)
import caffe
import scipy.io as sio
net=caffe.Classifier('examples/SuperResolution/SR_run.prototxt','examples/SuperResolution/snapshot/SR_iter_7105000.caffemodel')
sio.savemat('examples/SuperResolution/saveddata_7105000.mat',{'conv1_weights':net.params['conv1'][0].data,'conv1_biases':net.params['conv1'][1].data,'conv2_weights':net.params['conv2'][0].data,'conv2_biases':net.params['conv2'][1].data,'conv3_weights':net.params['conv3'][0].data,'conv3_biases':net.params['conv3'][1].data})
