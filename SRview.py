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
import math
from PIL import Image
from skimage.color import rgb2gray

sys.path.insert(0,'python')
plt.rcParams['figure.figsize']=(10,10)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

imageName='1'
LRName=imageName+'LR1.bmp'
SRResultName=imageName+'SR1.bmp'

inputs=[caffe.io.load_image('examples/SuperResolution/SR_prepare_data/test/LR_image/'+LRName,False)]

imageHeight=len(inputs[0])
imageWidth=len(inputs[0][0])
dim=len(inputs[0][0][0])
print imageHeight,imageWidth,dim

resultHeight=imageHeight-12
resultWidth=imageWidth-12

#inputs=[rgb2gray(input) for input in inputs]


infile=open('examples/SuperResolution/SR_run.prototxt')
outfile=open('examples/SuperResolution/SR_run_1.prototxt','w')
outfile.write('name: "SuperResolution"\ninput: "lowdata"\ninput_dim: 1\ninput_dim: 1\n')
outfile.write('input_dim:')
outfile.write(str(imageHeight))
outfile.write('\ninput_dim:')
outfile.write(str(imageWidth))
outfile.write('\n')
while(1):
   line=infile.readline()
   if not line:
      break
   outfile.write(line)
infile.close()
outfile.close()
net=caffe.Classifier('examples/SuperResolution/SR_run.prototxt','examples/SuperResolution/snapshot/SR_iter_75000.caffemodel')
#net.set_phase_test()
caffe.set_mode_gpu()
caffe.set_input_scale('lowdata',1)
#net.set_raw_scale('lowdata',1)
caffe.set_channel_swap('lowdata',[0])
SRResult=net.predict(inputs,0)
SRResult=SRResult.reshape(1,resultHeight,resultWidth)
SRResultImage=Image.new("RGB",(resultHeight,resultWidth))
scale=0.00390625
for x in range(0,resultHeight):
   for y in range(0,resultWidth):
     SRResultImage.putpixel([y,x],(int(math.floor(SRResult[0][x][y]*256)),int(math.floor(SRResult[0][x][y]*256)),int(math.floor(SRResult[0][x][y]*256))))
     SRResultImage.show()
     SRResultImage.save('examples/SuperResolution/Result/'+SRResultName)

