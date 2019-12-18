# LAEO-Net (head track generation demo) 
This repository contains demo code related with the head track generation task for the [LAEO-Net paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Marin-Jimenez_LAEO-Net_Revisiting_People_Looking_at_Each_Other_in_Videos_CVPR_2019_paper.pdf) (CVPR'2019)

### Quick start
The following demo detects and generates head tracks for the videos included in the [videos folder](./data/videos) 
subdirectory. Please take a look at the [software requirements section](#software_reqs), then
run the following commands in a terminal:

```bash
cd laeonet-head-tracking

# Download the detection model that matches the installed Python version
# NOTE: this step is optional, the demo script will try to download the file for you
# curl is required, if not installed, run this command
sudo apt-get install curl
# Check your Python version
python --version
# Replace .Y for ".5" or ".6", depending on the output of the previous command 
chmod +x ./data/models/detector/download_model_py3.Y.sh
./data/models/detector/download_model_py3.Y.sh

# Run the demo script
# Replace <video_file> with 'handShake_0001.avi', 'handShake_0013.avi' or 'highFive_0018.avi'
python ln_tracking_heads.py --video_path data/videos/<video_file> [--verbose 0|1]
```

### Software requirements
<a id='software_reqs'></a>
These are the most relevant dependencies required to run this demo:
- Python packages (Python 2.x is not supported): 
    - [numpy](https://www.scipy.org/install.html#pip-install)
    - [opencv-python](https://pypi.org/project/opencv-python/) (tested on version 3.4.5.20)
    - [h5py](https://pypi.org/project/h5py/)
- [Tensorflow](https://www.tensorflow.org/install/pip) (tested on `tensorflow-gpu` 1.14)
- [Keras](https://keras.io/#installation) (tested on version 2.2.4)
- [Our head detection model](https://github.com/AVAuco/ssd_head_keras) (code and model already provided in this repository)

### References
The videos used in this demo are part of the [TV Human Interaction Dataset](http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html)
which is cited below:

- A. Patron-Perez, M. Marszałek, A. Zisserman, I. D. Reid. "High Five: Recognising human interactions in TV shows.". British Machine Vision Conference, 2010.
 
If you use this repository for your work, you can cite it as:
```
@inproceedings{marin19cvpr,
  author    = {Mar\'in-Jim\'enez, Manuel J. and Kalogeiton, Vicky and Medina-Su\'arez, Pablo and and Zisserman, Andrew},
  title     = {{LAEO-Net}: revisiting people {Looking At Each Other} in videos},
  booktitle = CVPR,
  year      = {2019}
}
```
