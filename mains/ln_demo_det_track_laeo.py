"""
Demo code for detecting, tracking and testing a trained model.
This demo assumes that the input videos contain just one shot

Reference:
MJ. Marin-Jimenez, V. Kalogeiton, P. Medina-Suarez, A. Zisserman
LAEO-Net: revisiting people Looking At Each Other in videos
Intl Conference in Computer Vision and Pattern Recognition (CVPR), 2019

(c) MJMJ/2019
"""

__author__ = "Manuel J Marin-Jimenez"

import os, sys
import numpy as np
import cv2
#import deepdish as dd

from os.path import expanduser
homedir = expanduser("~")
mainsdir = os.path.dirname(os.path.abspath(__file__))

# Add custom directories with source code
sys.path.insert(0, os.path.join(mainsdir,"../laeonet-head-det-track")) # CHANGE ME
sys.path.insert(0, os.path.join(mainsdir,"../utils")) # CHANGE ME


import socket
hostname = socket.gethostname()
if hostname == "sylar":
   os.environ["CUDA_VISIBLE_DEVICES"]="0" # "-1"
if hostname == "sylar":
   gpu_rate = 0.45 # CHANGE ME!!!
else:
   gpu_rate = 0.30  # CHANGE ME!!!
theSEED = 1330


# for reproducibility
np.random.seed(theSEED)

from mj_tracksManager import TracksManager
from mj_avagoogleImages import mj_getImagePairSeqFromTracks, mj_getFrameBBsPairFromTracks
from mj_laeoImage import mj_padImageTrack
from ln_tracking_heads import process_video

from keras.models import load_model

# ====================================================================================

modeldir = "./models"
modelfile = os.path.join(modeldir, "model-hmaps-truco.hdf5")

outdirbase = mainsdir+"/../results"

case_wanted = "val"
inputs = "1010"


videoname = "highFive_0016"  # "highFive_0005" #
verbose = 1
save_to_disk = True 

with_other_maps = True

windowLenMap = 1



# Load model
model = load_model(modelfile)
model._make_predict_function()

# ========== MAIN ===========

outdir= os.path.join(outdirbase, videoname+"_laeo")


# Prepare name of results file and check if exists
scoresfile = os.path.join(outdir, "predictions.h5")
if with_other_maps:
    samplesfile = os.path.join(outdir, videoname+"_samples_wom.h5")
else:
    samplesfile = os.path.join(outdir, videoname+"_samples.h5")

if os.path.isfile(samplesfile):
    print("* INFO: samples file already exists. Skipping it!")


submean = True

# Mean head
if submean:
    meanPath = os.path.join(".","models","meanhead.npy")
    meanSample = np.load(meanPath)
else:
    meanSample = [0.0]



# Prepare data
# =================================
videospath = mainsdir+"/../laeonet-head-det-track/data/videos/"
videopath = os.path.join(videospath, videoname+".avi")
framesdir_for_detection = "/tmp/"+videoname+"_frames"


# Given video, detect heads and generate tracks
# ===============================
tracks_live = process_video(videopath, verbose=verbose, framesdir=framesdir_for_detection)
tm2 = TracksManager(filepath="", data=tracks_live)

lTracksInShots = []

lTracksInShots.append(tm2)

# Prepare inputs config
with_heads = inputs[0] == "1"
with_geom = inputs[1] == "1"
with_maps = inputs[2] == "1"
with_fcrops = inputs[3] == "1"

meanSampleFM = [0.0]
if with_maps or with_fcrops:
    meanfile = os.path.join(".", "models", "meanmap.npy")
    mean_map5 = np.load(meanfile)

winlen = 10
windowlenMap = 1
targetsize = (64, 64)

thr_upper_def = 0.6
thr_lower_def = 0.4
max_tracks = 5

lScores = []
lXs= []

nshots = 1 # This demo is for just one shot, extending to more than 1 is straight-forward

# Get tracks from shot
for shix in range(0, nshots):
    tracks = lTracksInShots[shix]

    if verbose > 0:
        print("- Found {:d} tracks in shot #{:d}.".format(tracks.ntracks, shix))

    starting_frame = []
    end_frame = []
    tr_scores = []
    lScoresPairs = []
    lXPairs = []

    tr_limit = min(max_tracks, tracks.ntracks)

    if tracks.ntracks < 2:
        print("\t - WARN: less than 2 tracks! Nothing to do: LAEO-score is 0")
        lXs.append(lXPairs)
        continue

    for trix in range(0, tracks.ntracks):

        starting_frame.append(tracks.start(trix))
        end_frame.append(tracks.end(trix))
        tr_scores.append(tracks.getTrackScore(trix))


    if tracks.ntracks < 3:
        if len(tr_scores) > 0:
            thr_upper = min(tr_scores)
        else:
            thr_upper = 0.1
        thr_lower = thr_upper/2
    else:
        thr_upper = thr_upper_def
        thr_lower = thr_lower_def

    # Selecting two sample tracks
    for trix1 in range(0, tr_limit-1):

        for trix2 in range(trix1+1, tr_limit):

            eval_this_pair = False
            if tr_scores[trix1] >= thr_upper or tr_scores[trix2] >= thr_upper:
                eval_this_pair = True

            if tr_scores[trix1] < thr_lower or tr_scores[trix2] < thr_lower:
                eval_this_pair = False

            # Always eval this two (assuming tracks are sorted by score): 0,1
            if trix1 == 0 and trix2 == 1:
                eval_this_pair = True

            if not eval_this_pair:
                continue

            print("- Focusing on tracks: #{:d} ({:.2f}) and #{:d} ({:.2f}).".format(trix1, tracks.getTrackScore(trix1),
                                                                                    trix2, tracks.getTrackScore(trix2)))

            init_t = max(starting_frame[trix1], starting_frame[trix2])

            end_t = min(end_frame[trix1], end_frame[trix2])

            if end_t < init_t:  # No intersection in time
                continue

            #print(starting_frame)
            if verbose > 0:
                print('\t* Loading data from file...')

            # Get sample
            lsamplesL = []
            lsamplesR = []
            lG = []
            lM = []
            lF = []

            lFrames = []
            lFrix = []
            lBBs = []
            lX = []

            init_frame_loop_limit = end_t - winlen + 1

            if init_frame_loop_limit <= init_t:   # This is to control cases where the shot is shorter than 10 frames (winlen)
                init_frame_loop_limit = init_t+1

            for init_frame in range(init_t, init_frame_loop_limit):
                strict_mode = not ((end_t - init_t) < winlen)
                M = []
                if with_maps:
                    sampleL, sampleR, G, M = mj_getImagePairSeqFromTracks(tracks, (trix1, trix2),
                                                                          init_frame, winlen, framesdir_for_detection,
                                                                targetsize, meanSample, with_maps=with_maps,
                                                                mean_map=mean_map5, strict_mode=strict_mode,
                                                                          with_other_maps=with_other_maps,
                                                                          winlenMap=windowlenMap)
                else:
                    sampleL, sampleR, G = mj_getImagePairSeqFromTracks(tracks, (trix1, trix2),
                                                                       init_frame, winlen, framesdir_for_detection,
                                                                targetsize, meanSample)

                if sampleL.shape[0] < winlen:
                    sampleL = mj_padImageTrack(sampleL, winlen=winlen, mode="shared")
                    sampleR = mj_padImageTrack(sampleR, winlen=winlen, mode="shared")

                    if with_maps and windowlenMap > M.shape[0]:
                        M = mj_padImageTrack(M, winlen=windowlenMap, mode="shared")

                    if with_geom:
                        print("ERROR!!! Not implemented for geometry extension yet!!!")
                        exit(-1)

                if with_heads:
                    lsamplesL.append(sampleL)
                    lsamplesR.append(sampleR)

                if with_geom:
                    lG.append(G)

                if with_maps:
                   lM.append(M)

                if save_to_disk:
                    # Get full frame and BBs for drawing later
                    bb1s, bb2s, frame_t, img = mj_getFrameBBsPairFromTracks(tracks, (trix1, trix2), init_frame, winlen,
                                                                            framesdir_for_detection, strict_mode=strict_mode)
                    lFrames.append(img)
                    lFrix.append(frame_t)
                    lBBs.append((bb1s, bb2s))

            npairs = len(lsamplesL)


            # Run the model on the test samples
            # =================================
            nsamplesEval = 0

            scores = np.zeros((npairs), dtype=np.float32)
            for index in range(0, npairs):
                X = []
                if with_heads:
                    X0 = np.expand_dims(lsamplesL[index], axis=0)
                    X1 = np.expand_dims(lsamplesR[index], axis=0)

                    X.append(X0)
                    X.append(X1)

                if with_geom:
                    G = np.expand_dims(lG[index],axis=0)
                    X.append(G)

                if with_maps:
                    M = np.expand_dims(lM[index], axis=0)
                    X.append(M)


                prediction = model.predict(X)
                if verbose > 1 and not save_to_disk:
                    print(prediction[0,1])
                scores[index] = prediction[0,1]

                lX.append(X)

                nspb = 1

                nsamplesEval = nsamplesEval + nspb

            # Store to save to disk later
            score_dic = {'pairs': (trix1, trix2), 'scores': scores,
                         'lfrix': lFrix, 'lbbs': lBBs}
            sample_dic = {'shix': shix, 'pairs': (trix1, trix2), 'samples': lX,
                         'lfrix': lFrix, 'lbbs': lBBs}
            if verbose > 0 and len(scores) > 0:
                print("\t Max score: {}".format(max(scores)))

            lScoresPairs.append(score_dic)
            lXPairs.append(sample_dic)

            # Export to disk frames with scores?
            if save_to_disk:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (36, int(targetsize[0]+targetsize[0]/2))
                fontScale = 0.8
                fontColor = (255, 255, 255)
                lineType = 2

                if not os.path.exists(outdir):
                    os.makedirs(outdir)

                for i in range(0, npairs):
                    canvas_size = (128, 180, 3)
                    canvas = np.zeros(canvas_size, dtype=np.uint8)

                    dx = int((canvas_size[1] - 2*targetsize[0])/2)

                    canvas[2:2+targetsize[0], dx:dx+targetsize[1], ] = (np.squeeze(lsamplesL[i][5,]) + meanSample)*255
                    canvas[2:2 + targetsize[0], dx + targetsize[1]:dx +2*targetsize[1], ] = (np.squeeze(lsamplesR[i][5,]) + meanSample)*255

                    # Add LAEO score
                    cv2.putText(canvas, "{:1.4f}".format(scores[i]),
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineType)

                    if with_geom:
                        # Draw geometry
                        canvas_geo_size = (canvas_size[0], canvas_size[0], 3)
                        canvas_geo = 255+np.zeros(canvas_geo_size, dtype=np.uint8)
                        center_geo = (int(canvas_geo_size[0]/2), int(canvas_geo_size[0]/2))
                        x = int(center_geo[0]+ lG[i][0]*(canvas_geo_size[0]/2))
                        y = int(center_geo[1]+ lG[i][1]*(canvas_geo_size[0]/2))
                        scale = lG[i][2]
                        cv2.circle(canvas_geo, (x,y), 3, (255, 255, 100), 3)

                        cv2.circle(canvas_geo, center_geo, max(1,int(3*scale)), (255, 100, 255), 3)

                        canvas = np.concatenate((canvas, canvas_geo), axis=1)

                        imagename = os.path.join(outdir,"sh{:03d}_p{:d}_{:d}_{:03d}.jpg".format(shix,trix1, trix2, i))

                        cv2.imwrite(imagename, canvas)

                    # -------------------------------------------
                    # Full frame with BBs
                    # -------------------------------------------
                    borderwidth = 6
                    color = (0, int(255 * scores[i]), 0)
                    img2 = cv2.rectangle(lFrames[i], (lBBs[i][0][0], lBBs[i][0][1]), (lBBs[i][0][2], lBBs[i][0][3]),
                                         color, borderwidth)
                    img2 = cv2.rectangle(img2, (lBBs[i][1][0], lBBs[i][1][1]), (lBBs[i][1][2], lBBs[i][1][3]), color,
                                         borderwidth)

                    # Text config
                    bottomLeftCornerOfTextFr = (int(img2.shape[1]*0.4), 35)
                    fontScaleFr = 1
                    lineThick = 4

                    fontColor = (int(255 * scores[i]), 0, 0)

                    # Add LAEO score
                    cv2.putText(img2, "{:1.3f}".format(scores[i]), bottomLeftCornerOfTextFr,
                                font, fontScaleFr, fontColor, lineThick)



                    imagename = os.path.join(outdir, "fr_sh{:03d}_p{:d}_{:d}_{:03d}.jpg".format(shix, trix1, trix2, i))
                    cv2.imwrite(imagename, img2)

    lScores.append(lScoresPairs)
    lXs.append(lXPairs)

# Save scores to disk
if verbose > 0:
    print("- Saving results: {:s}".format(scoresfile))
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Sanity check
if False:
    cv2.imshow("One-head", np.squeeze(lXs[0][0]['samples'][0][0][0, 4,]) + meanSample)
    cv2.waitKey()


# Want to save data into files?
# dd.io.save(scoresfile, lScores)
# dd.io.save(samplesfile, lXs)


# ========== MAIN ===========

