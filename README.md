# Requirements
***
**Models**
* Basel Face Model (BFM'09): [click here](https://faces.dmi.unibas.ch/bfm/index.php?nav=1-2&id=downloads) to obtain the Basel Face Model from the University of Basel
* Expression Model: Download the expression model (the Exp_Pca.bin) file from [this link](https://github.com/Juyong/3DFace)

**Software**
* CUDA (tested with v11.0)
* OpenCV (tested with v4.5)
* Python 3

The following python packages are also needed, but these can be installed by following the instructions in Section 2 of Installation below.
* cvxpy (for temporal smoothing via post-processing)
* scikit-learn
* matplotlib
* opencv-python

# Installation
***


### 1) Compile CUDA code
Clone the git repository and compile the CUDA code as below

```
cd build 
chmod +x builder.sh
./builder.sh
```

### 2) Install python packages

Install the necessary packages via pip. It is advised to use a virtual environment by running
```
cd build
python -m venv env
source env/bin/activate

```

The necessary packages can simply be installed by running.

```
pip install -r requirements.txt
```

### 3) Pre-process Morphable Models

Make sure that you downloaded the Basel Face Model (`01_MorphableModel.mat`) and the Expression Model (`Exp_Pca.bin`) as highlighted in the Requirements section above. Then, copy these model files into the `build/models/raw` directory. Specifically, these files should be in the following locations:
```
build/models/raw/01_MorphableModel.mat
build/models/raw/Exp_Pca.bin
```

Then run the following python script to adapt these models to the 3DI code:
```
cd build/models
python prepare_BFM.py
```


# Running the code
***

### Quickstart
Go to the `build` directory and, if you used a virtual environment, activate it by running `source env/bin/activate`. Then, the following is an example command (will produce visuals as well)
```
python process_video.py ./output testdata/elaine.mp4
```
The produced files are in the subdirectory created under the `output` directory. The file with expression parameters has the extension `.expressions_smooth` and the pose parameters have the extension `.poses_smooth`. The script above takes approximately 6 minutes to run, and this includes the production of the visualization parameters as well as temporal smoothing. Some tips to reduce total processing time:
* *Visualization* (i.e., production of rendering videos) can be disabled by passing the parameter `--produce_3Drec_videos=False`.
* *Temporal smoothing* can be disabled by passing the parameter `--smooth_pose_exp=False`
* The reconstruction can be sped up by passing the parameter `--cfgid=2` to use the 2nd configuration file, which is faster although results will include more jitter


More parameters can be seen by running 
```python process_video.py --help```

The `process_video.py` script does a series of pre- and post-processing for reconstruction (details are in the section below). It is important to know that we **first estimate the identity parameters** of the subject in the video, by using a small subset of the video frames, and then we compute pose and expression coefficients at every frame. Thus, the identity parameters are held common throughout the video.


### Details of video processing


The `process_video.py` script does a series of processes on the video. Specifically, it does the following steps in this order:
1. Face detection on the entire video
1. Facial landmark detection on the entire video
1. 3D (neutral) identity parameter estimation via 3DI (using a subset of frames from the videos)
1. Frame-by-frame 3D reconstruction via 3DI (identity parameters are fixed here to the ones produced in the previous step)
1. (Optional) Temporal smoothing
1. (Optional) Production of 3D reconstruction videos (similar to these###)
1. (Optional) Production of video with 2D landmarks estimated by 3DI

The first four steps are visualized below; the blue text indicates the extension of the corresponding files

<img src="docs/process_video1.png"  alt="Drawing" style="width: 800px;"/>

The 2D landmarks estimated by 3DI are also produced optionally based on the files produced above.

<img src="docs/process_video3.png"  alt="Drawing" style="width: 400px;"/>


# Output Formats

Below are the extensions some of the output files provided by 3DI video analysis software:
* `.expressions_smooth`: A text file that contains all the expression coefficients of the video. That is, the file contains a `Tx79` matrix, where the `t`th row contains the 79 expression coefficients of the expression (PCA) model
* `.poses_smooth`: A text file that contains all the poses coefficients of the video. The file contains a `Tx9` matrix, where the **first 3 columns** contain the the 3D translation `(tx, ty, tz)` for all the `T` frames of the video and the **last 3 columns** of the matrix contain the rotation (`yaw, pitch, roll`) for all the `T` frames. 
* `.2Dlandmarks`: A text file with the 51 landmarks corresponding to the inner face (see below), as predicted by 3DI. The file contains a matrix of size `Tx102`, where each row is of the format: `x0 y0 x1 y1 ... x51 y51`.
<img src="docs/landmarks.png"  alt="Landmarks" style="width: 450px;"/>


