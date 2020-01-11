import os
import csv
import numpy as np 

def read_csv(in_csv_file, delimiter=','):
    out_csv = []
    with open(in_csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        for row in reader:
            out_csv.append(row)


    return out_csv

phase = 'val' # train
laeo_csv_file = 'ava_laeo_' + phase + '.csv'
# DOWNLOAD the ava csv files 
ava_gt_csv_file = 'ava_' + phase + '_v2.2.csv'
laeo_file = read_csv(laeo_csv_file)

# find the bounding boxes of each human from the ava csv file 
gt_csv = np.array(read_csv(ava_gt_csv_file))
for _, video, clip_id, h_id1, h_id2, laeo) in laeo_file:
    vid_idx = np.where(gt_csv[:,0] == video)[0]
    clip_idx = np.where(gt_csv[vid_idx,1] == clip_id.zfill(4))[0]
    csv_lines_human1 = np.where(gt_csv[vid_idx[clip_idx],7] == h_id1)[0]
    csv_lines_human2 = np.where(gt_csv[vid_idx[clip_idx],7] == h_id2)[0]
    # bounding boxes
    bbox_human1 = gt_csv[vid_idx[clip_idx][csv_lines_human1[0]],2:6]
    bbox_human2 = gt_csv[vid_idx[clip_idx][csv_lines_human2[0]],2:6]
    if int(laeo) == 1:
        print('In {}, humans id1={}-id2={} are looking at each other (lines {} and {} in ava csv file)'.format(
            video + '/' + clip_id, h_id1, h_id2, csv_lines_human1, csv_lines_human2))



