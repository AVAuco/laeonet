"""
Demo code for testing a trained model

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net: revisiting people Looking At Each Other in videos  
Intl Conference in Computer Vision and Pattern Recognition (CVPR), 2019 

(c) MJMJ/2019
"""

__author__ = "Manuel J Marin-Jimenez"

import os, sys, getopt
import numpy as np
import cv2

homedir = "/home/mjmarin/research/laeonet/"

# Add custom directories with source code
sys.path.insert(0, homedir + "datasets")

os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpu_rate = 0.45 # CHANGE ME!!!

theSEED = 1330

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Tensorflow config
tf.set_random_seed(theSEED)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_rate
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
# for reproducibility
np.random.seed(theSEED)

from keras.models import load_model

# Get command line parameters: could be used to select the input model and/or test file
argv = sys.argv[1:]

winlen = 10            # Fixed: temporal length

# Define path to file containing the model
 # Model trained on UCO
modelpath = homedir + "/models/model-hmaps-truco.hdf5"
 # Model trained on AVA
#modelpath = homedir + "/models/model-hmaps-trava.hdf5"

# Load model into memory
model = load_model(modelpath)
model.summary()

# Load test data (already cropped for speed)
imagesdir= homedir +"/data/ava_val_crop/"

# Select the example
# ===========================
# The followings are LAEO
basename = "om_83F5VwTQ_01187_0000_pair_51_49"
#basename = "covMYDBa5dk_01024_0000_pair_37_35"
#basename = "7T5G0CmwTPo_00936_0000_pair_20_19"
#basename = "914yZXz-iRs_01549_0000_pair_192_194"

# The followings are not LAEO
#basename = "914yZXz-iRs_01569_0000_pair_196_195"
#basename = "SCh-ZImnyyk_00902_0000_pair_1_0"

pairspath = os.path.join(imagesdir, basename + ".jpg")
mapspath = os.path.join(imagesdir, basename + "_map.jpg")

imgpairs = cv2.imread(pairspath)
imgmaps = cv2.imread(mapspath)

# cv2.imshow("Pairs", imgpairs)
# cv2.waitKey()

# Load mean head and mean map
meanpath = homedir+"/models/meanhead.npy"
meansample = np.load(meanpath)

meanfile = os.path.join(homedir, "models", "meanmap.npy")
mean_map5 = np.load(meanfile)

# Prepare inputs
ncols = imgpairs.shape[1]
ncols_2 = int(ncols / 2)

sampleL = np.zeros((winlen, ncols_2, ncols_2, 3))
sampleR = np.zeros((winlen, ncols_2, ncols_2, 3))

# Separate into two head tracks
for t in range(0, winlen):
    sampleL[t,] = (imgpairs[t * ncols_2:(t + 1) * ncols_2, 0:ncols_2, ] / 255.0) - meansample
    sampleR[t,] = (imgpairs[t * ncols_2:(t + 1) * ncols_2, ncols_2:ncols, ] / 255.0) - meansample

headmapnorm = (imgmaps - mean_map5) / 255.0

# Run inference
X0 = np.expand_dims(sampleL, axis=0)
X1 = np.expand_dims(sampleR, axis=0)

M = np.expand_dims(headmapnorm, axis=0)

X = [X0, X1, M]

prediction = model.predict(X)

print("Probability of LAEO is: {:.2f}%".format(prediction[0][1]*100))

print("End of test.")