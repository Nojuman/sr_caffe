import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
import glob
import caffe
import scipy.io as sio
from pylab import *

matfn='examples/SuperResolution/ResumeCaffeModel/x3.mat'
Mat=sio.loadmat(matfn)
outfile=open('examples/SuperResolution/ResumeCaffeModel/x2data.txt','w')

data=Mat['weight_conv1']
for i in range(0,len(data))
   for j in range(0,len(data[0]))
      outfile.write(str(data[i][j])+'')
		
data=Mat['biases_conv1']
for i in range(0,len(data))
     outfile.write(str(data[i][0])+'')
		
data=Mat['weight_conv2']
for i in range(0,len(data))
   for j in range(0,len(data[0][0]))
      outfile.write(str(data[i][j])+'')
		
data=Mat['biases_conv2']
for i in range(0,len(data))
   outfile.write(str(data[i][0])+'')
		
data=Mat['weight_conv3']
for i in range(0,len(data))
   for j in range(0,len(data[0]))
      outfile.write(str(data[i][j])+'')
		
data=Mat['biases_conv3']
outfile.write(str(data[0][0])+'')

outfile.close()
