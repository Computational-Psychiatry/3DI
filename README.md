# Compiling the code

Follow the steps below to compile the code

1. Clone this repository somewhere in your home directory via `git clone https://github.com/Computational-Psychiatry/3DI.git`
1. Populate the `build/models` directory by copying files (on `WS03`) with the command: `cp -r /offline_data/models/3DI/* <your-local-git-clone>/build/models/.`
1. Go inside the `build` directory (`cd build`) and compile the code by running `./builder.sh`
1. Optionally, clean unnecessary files by running `./cleaner.sh`

***

# Running the code

If the code is compiled successfully with the instructions above, an executable file named `video` must be created in the `build` directory. Run this executable within the `build` directory as below

`./video <path_to_video_file> <path_to_output_dir> <path_to_config_file> <camera_parameter>`

Parameters here are as below

- **parameter #1** *path_to_video_file*: This must be the absolute path to video file to be processed
- **parameter #2** *path_to_output_dir*: This is where all the outputs (i.e., expression & pose signals, video) will be stored
- **parameter #3** *path_to_config_file*: The path to the config file
- **parameter #4** *camera_parameter*: the parameter for the camera to be used during reconstruction. There are two options here:
    - a) The user can put a string, which must be the filepath of the camera model. That is, if the did camera calibration and stored the output to a config file, we can put the path of the config file
    - b) The user can put a number, say, 120.0. In this case, the software will use a camera with a field of view of 120 degrees. This option should be used when we do not have calibration file for the camera---the calibration option above is expected to be better and eliminate the guesswork to find the correct fov

Example command:

`./video /offline_data/face/CAR/DSv2/interested/ML0001_2.mp4 ./output ./configs/fast.txt 50.0`

If the video is collected using the SensorTree v3, you should run the code as below (i.e., using the calibration file named `TreeCam_1041a.txt`):

`./video /offline_data/face/CAR/DSv2/interested/ML0325_1.mp4 ./output ./configs/fast.txt models/cameras/TreeCam_1041a.txt`

***

# Outputs

- **Expressions**: This is a text file that contains a matrix of size `Tx79`, where `T` is the number of processed frames, which is either the number of frames in the video or `MAX_VID_FRAMES_TO_PROCESS` if this parameter (see below) is smaller than the total number of frames in the video. If we set the SKIP_FIRST_N_SECS parameter (see below) to something greater than 0, then the first *K* frames (*K=SKIP_FIRST_N_SECSxFPS of video*) will contain `NaN` values. The k*th* row of this matrix contains the 79 expression coefficients corresponding to the k*th* frame of the video
- **Pose**: This is a text file that contains a matrix of size `Tx6`, where `T` is as above and the first `K` frames may be `NaN` as described above. The first three columns contain the 3D translation parameters. The last three columns contain the pose (head orientation) parameters; **WARNING** these are not the angles, but the Euler-Rodrigues representation of the pose. Post-processing needs to be applied to convert them to angles (TO-DO). 
- **Expression-related landmark movement**: This produces a text file (with the extension `.landmarks_dexp`) of size `Tx153`. The `k`th row contains the expression-related movement of the 51 landmarks at the `k`th frame, in the format `dX0 dY0 dZ0 dX1 dY1 dZ1 ... dX50 dY50 dZ50`. To obtain the absolute coordinates, you must add the mean face. Run script `build/scripts/visualize_landmarks.py` to see an example. The `OUTPUT_LANDMARKS_EXP_VARIATION` parameter must be set to 1 in order to produce this output.

Also, rendered videos that may be produced alongside with expression/pose coefficients and landmarks. See `OUTPUT_VISUALS` parameter below.

*** 

# Examples

## Example 1: Landmarks (i.e., expression-related landmark movements)

1. Compile the code by following instructions in the section Compiling the code
1. Run the command `./video vid_file.mp4 ./output ./config/default_output_landmarks.txt 90`
    - a) Warning 1: the last parameter, `90`, is the camera parameter and it must be modified accordingly particularly if you know and have the model of the camera that recorded the data; see **parameter #4** in section Running the code.
    - b) Warning 2: the configuration file `./config/default_output_landmarks.txt` is for example purposes and will process only a little part of the video to produce results quickly. If you run the code for actual analyses you must change the `MAX_VID_FRAMES_TO_PROCESS` in the config file to a value large enough to process entire videos.
1. The landmark movements are stored in the file `./output/vid_file.landmarks_dexp` (see section Outputs above to interpret this file) 
1. You can visualize the landmarks by following the script `./build/scripts/visualize_landmarks.py`


***

# Config options

As seen in the section above, the code takes a config file as input. Below are config options that are possibly of interest to the end user.

### Ignoring beginning/ending of file

- `SKIP_FIRST_N_SECS`: You can change this number to ignore the first N seconds of the video. Typically, in the CASS videos we ignore about 10-15 seconds of the video
- `MAX_VID_FRAMES_TO_PROCESS: ` You can limit the number of frames to be processed here. Note that, here you put the *frame number*, whereas the setting above is in terms of number of seconds. If you set this parameter, say, to 5000, it will stop processing after the 5000th frame of the video (independently of the parameter above)

### Output files
- `OUTPUT_VISUALS`: Create the videos that visualize the result of 3D face reconstruction. **NOTE:** The first `SKIP_FIRST_N_SECS` seconds of the video will be blank, this is on purpose. Two videos will be produced: 
    - a) a video that shows the rendered pose+expression and frontal expression alongside with the input video
    - b) a video that shows the texture alongside the input video

### Command line output
- `PRINT_EVERY_N_FRAMES`: print the number of processed video frame every N frames. E.g., setting it to 100 will print a line every 100 frames
- `PRINT_WARNINGS`: By default this should be set to 0. It prints some warnings that are helpful for debugging
- `PRINT_DEBUG`: By default this should be set to 0. It prints further notes for debugging

### Parameters that affect processing speed
- `NTOT_RECONSTRS`: This parameter should ideally be set to something like 8, but it can be temporarily reduced (e.g., to 2) to speed up output if we need to get results quickly. This parameter affects the processing speed only at the identity learning phase, which is the preprocessing phase before we actually start processing the video. There is no point in increasing this parameter beyond 10
- `NMULTICOMBS`: This parameter should typically be set to something like 4 or 6. Increasing it will reduce the temporal noise in the expression coefficients (and the jitter in the rendered video -- see `OUTPUT_VISUALS`), but will linearly increase processing time. There is no point in increasing this parameter beyond 12 (which will already be very slow)
- `NRES_COEFS`: Ditto. A good rule of thumb is to make this parameter equal with `NMULTICOMBS`

### The output directory
- `OUTDIR_WITH_PARAMS`: By default this should be set to 0. But if you want to run the code by varying certain parameters (see below) and observe the effect of each parameter, `OUTDIR_WITH_PARAMS` can be set to 1, as it will create a subdirectory (whose name will encode the parameter settings)


