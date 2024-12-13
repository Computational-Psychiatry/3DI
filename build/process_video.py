import numpy as np
import argparse
import warnings
from time import time
import os


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#def save_identity_and_shape(alpha, beta, shp_path, tex_path, morphable_model):
#    sdir = './models/MMs/%s' % morphable_model 
#    IX  = np.loadtxt('%s/IX.dat' % sdir)
#    IY  = np.loadtxt('%s/IY.dat' % sdir)
#    IZ  = np.loadtxt('%s/IZ.dat' % sdir)
#    
#    TEX  = np.loadtxt('%s/TEX.dat' % sdir)
#    
#    tex_mu = np.loadtxt('%s/tex_mu.dat' % sdir)
#    
#    x0 = np.loadtxt('%s/X0_mean.dat' % sdir)
#    y0 = np.loadtxt('%s/Y0_mean.dat' % sdir)
#    z0 = np.loadtxt('%s/Z0_mean.dat' % sdir)
#    
#    x = (x0+(IX @ alpha)).reshape(-1,1)
#    y = (y0+(IY @ alpha)).reshape(-1,1)
#    z = (z0+(IZ @ alpha)).reshape(-1,1)
#    
#    tex = (tex_mu+(TEX @ beta)).reshape(-1, 1)
#    np.savetxt(shp_path, np.concatenate((x,y,z), axis=1))
#    np.savetxt(tex_path, tex)
    

def plot_landmarks(vid_fpath, lmks_path, out_video_path):
    import cv2
    print(out_video_path)
    
    points_all = np.loadtxt(lmks_path)
    cap = cv2.VideoCapture(vid_fpath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'XVID', 'MJPG', etc.
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    
    # Loop through each frame of the video
    frame_idx = -1
    while cap.isOpened():
        frame_idx += 1
        ret, frame = cap.read()
        
        if ret and  frame_idx < points_all.shape[0]:
            points = points_all[frame_idx,:].reshape(-1,2)
            # print(points)
            
            frame_large = cv2.resize(frame, None, fx=4, fy=4)

            # Plot the 2D points on the current frame
            for point in points:
                if point[0] == 0:
                    continue
                cv2.circle(frame_large, tuple((4*point).astype(int)), 10, (0, 0, 255), -1)
                
            frame = cv2.resize(frame_large, None, fx=0.25, fy=0.25)
            
            out.write(frame)
            
        else:
            break
        
    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    
    # Close all windows
    cv2.destroyAllWindows()
        
    
    
    
# def process_single_video(vid_fpath0, landmark_model, out_dir_base, camera_param, 
#                          do_undistort, cfgid, produce_3Drec_videos, produce_landmark_video,
#                          write_2D_landmarks, write_canonical3D_landmarks,
#                          smooth_pose_exp, 
#                          local_exp_basis_version,
#                          compute_local_exp_coeffs,
#                          morphable_model='BFMmm-19830', 
#                          use_smoothed_lmks=False):
    
def process_single_video(args, 
                         morphable_model='BFMmm-19830',
                         use_smoothed_lmks=False):
    vid_fpath0 = args.video_path
    landmark_model = args.landmark_model
    out_dir_base = args.out_dir
    camera_param = args.camera_param
    do_undistort = args.undistort
    cfgid = args.cfgid
    write_2D_landmarks = args.write_2D_landmarks
    write_canonical3D_landmarks = args.write_canonical3D_landmarks
    produce_3Drec_videos = args.produce_3Drec_videos
    produce_landmark_video = args.produce_landmark_video
    smooth_pose_exp = args.smooth_pose_exp
    compute_local_exp_coeffs = args.compute_local_exp_coeffs
    local_exp_basis_version = args.local_exp_basis_version
    
    if compute_local_exp_coeffs:
        write_canonical3D_landmarks = True
        
    if write_canonical3D_landmarks:
        smooth_pose_exp = True

    # args = parser.parse_args()
    
    
    out_dir_pre = out_dir_base + '/preprocessing/' 
    os.makedirs(out_dir_pre, exist_ok=True)
    
    Nsteps = 4

    if produce_3Drec_videos and not smooth_pose_exp:
        warnings.warn(""""Setting smooth_pose_exp to True. 
                      This is required to produce the 3D reconstruction videos
                      that are asked (produce_3Drec_videos=True).""")
        smooth_pose_exp = True
        
    if smooth_pose_exp:
        Nsteps += 1
        
    if produce_3Drec_videos:
        Nsteps += 1

    if produce_landmark_video:
        Nsteps += 1
        
    if write_canonical3D_landmarks:
        Nsteps += 1
        
    if compute_local_exp_coeffs:
        Nsteps += 1
    
    curstep = 0

    if do_undistort:
        bbn0 = '.'.join(os.path.basename(vid_fpath0).split('.')[:-1])
        vid_fpath = '%s/%s_undistorted.mp4' % (out_dir_pre, bbn0)
        cmd = './video_undistort %s %s %s' % (vid_fpath0, camera_param, vid_fpath)
        
        # Command #0: Undistort video (if necessary)
        if not os.path.exists(vid_fpath):
            print('Preprocessing: Undistorting video')
            print(cmd)
            os.system(cmd)
    else:
        vid_fpath = vid_fpath0
    
    bbn = '.'.join(os.path.basename(vid_fpath).split('.')[:-1])
    vid_bn = os.path.basename(vid_fpath).split('.')[0]
    
    cfg_landmark = './configs/%s.cfg%d.%s.txt' % (morphable_model, cfgid, landmark_model)
    cfg_fpath = './configs/%s.cfg%d.%s.txt' % (morphable_model, cfgid, landmark_model)
    cfg_bn = '.'.join(os.path.basename(cfg_fpath).split('.')[:-1])
    
    if not os.path.exists(out_dir_base):
        os.mkdir(out_dir_base)
    
    if not os.path.exists(out_dir_pre):
        os.mkdir(out_dir_pre)
    
    out_dir = '%s/%s' % (out_dir_base, cfg_bn)
    
    if use_smoothed_lmks:
        out_dir += '_sm2'
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    rects_fpath = '%s/%s.face_rects' % (out_dir_pre, vid_bn)
    landmarks_fpath = '%s/%s.landmarks.%s' % (out_dir_pre, vid_bn, landmark_model)
    
    # This is currently not encourged -- landmark smoothing hasn't been tested
    if use_smoothed_lmks:
        landmarks_fpath_raw = landmarks_fpath
        landmarks_fpath = landmarks_fpath + '_sm'
        if not os.path.exists(landmarks_fpath):
            cmd_smoother = 'python ./scripts/total_variance_rec_landmarks.py %s %s' % (landmarks_fpath_raw, landmarks_fpath) 
            os.system(cmd_smoother)
        
    shp_cfpath = '%s/%s.alphas' % (out_dir, vid_bn)
    tex_cfpath = '%s/%s.betas' % (out_dir, vid_bn)
    shp_fpath = '%s/%s.shp' % (out_dir, vid_bn)
    tex_fpath = '%s/%s.tex' % (out_dir, vid_bn)
    shpsm_fpath = '%s/%s.shp_sm' % (out_dir, vid_bn)
    texsm_fpath = '%s/%s.tex_sm' % (out_dir, vid_bn)
    
    curstep += 1
    if not os.path.exists(rects_fpath):
        cmd_rects = './video_detect_face %s %s' % (vid_fpath, rects_fpath)
        
        print('Step %d: Face detection on all frames' % (curstep))
        print('======================================')
        # Command #1: Detect face rectangles
        print(cmd_rects)
        t0 = time()
        os.system(cmd_rects)
        print('\t > Took %.2f secs' % (time()-t0))
    
    curstep += 1
    if not os.path.exists(landmarks_fpath):
        cmd_landmarks = './video_detect_landmarks %s %s %s %s' % (vid_fpath, rects_fpath, landmarks_fpath, cfg_landmark)
        
        print('\n\nStep %d: Landmark detection on all frames' % (curstep))
        print('===========================================')
        # Command #2: Detect landmarks per face rectangle
        print(cmd_landmarks)
        t0 = time()
        os.system(cmd_landmarks)
        print('\t > Took %.2f secs' % (time()-t0))
    
    curstep += 1
    if not os.path.exists(shp_cfpath) or not os.path.exists(tex_cfpath):
        cmd_identity = './video_learn_identity %s %s %s %s %s %s' % (vid_fpath, landmarks_fpath, cfg_landmark, 
                                                                              camera_param, shp_cfpath, tex_cfpath)
        print('\n\nStep %d: Learning subject identity' % (curstep))
        print('===================================')
        # Command #3: Learn identity
        
        print(cmd_identity)
        t0 = time()
        os.system(cmd_identity)
        print('\t > Took %.2f secs' % (time()-t0))

    
    alpha = np.loadtxt(shp_cfpath)
    beta =  0.4*np.loadtxt(tex_cfpath)
    
    alpha_sm = 0.70*np.loadtxt(shp_cfpath)
    beta_sm =  0.70*np.loadtxt(tex_cfpath)
    
    if not os.path.exists(shp_fpath) or not os.path.exists(tex_fpath):
        # save_identity_and_shape(alpha, beta, shp_fpath, tex_fpath, morphable_model)
        cmd_save = 'python ./scripts/save_identity_and_shape.py %s %s %s %s %s' % (alpha, beta, shp_fpath, tex_fpath, morphable_model)
        os.system(cmd_save)
        
    if not os.path.exists(shpsm_fpath) or not os.path.exists(texsm_fpath):
        # save_identity_and_shape(alpha_sm, beta_sm, shpsm_fpath, texsm_fpath, morphable_model)
        cmd_save = 'python ./scripts/save_identity_and_shape.py %s %s %s %s %s' % (alpha_sm, beta_sm, shpsm_fpath, texsm_fpath, morphable_model)
        os.system(cmd_save)
    
    exp_path = '%s/%s.expressions' % (out_dir, bbn)
    exps_path = '%s/%s.expressions_smooth' % (out_dir, bbn)
    pose_path = '%s/%s.poses'  % (out_dir, bbn)
    poses_path = '%s/%s.poses_smooth'  % (out_dir, bbn)
    illum_path = '%s/%s.illums' % (out_dir, bbn)
    # render3d_path = '%s/%s_3D.avi' % (out_dir, bbn)
    # texture_path = '%s/%s_texture.avi' % (out_dir, bbn)
    render3ds_path = '%s/%s_3D_pose-exp-greenbg.avi' % (out_dir, bbn)
    texturefs_path = '%s/%s_texture_sm.avi' % (out_dir, bbn)
    render3d2_path = '%s/%s_3D_comb.mp4' % (out_dir, bbn)
    render3d3_path = '%s/%s_3D_pose-exp.mp4' % (out_dir, bbn)
    lmks_video_path = '%s/%s_lmks.mp4' % (out_dir, bbn)
    
    lmks_path = '%s/%s.2Dlmks' % (out_dir, bbn)
    canonical_lmks_path = '%s/%s.canonical_lmks' % (out_dir, bbn)
    local_exp_coeffs_path = '%s/%s.local_exp_coeffs.v%s' % (out_dir, bbn, local_exp_basis_version)
    
    curstep += 1
    if not os.path.exists(exp_path) or not os.path.exists(pose_path) or not os.path.exists(illum_path):
        cmd_video = './video_from_saved_identity %s %s %s %s %s %s %s %s %s' % (vid_fpath, landmarks_fpath, cfg_fpath, camera_param,
                                                                                shp_fpath, tex_fpath, exp_path, pose_path, illum_path)
        print('\n\nStep %d: Performing 3D reconstruction on the entire video' % (curstep))
        print('==========================================================')
        # Command #4: Compute pose and expression coefficients
        print(cmd_video)
        t0 = time()
        os.system(cmd_video)
        print('\t > Took %.2f secs' % (time()-t0))

    t0 = time()
    curstep += 1
    if smooth_pose_exp and not os.path.exists(exps_path):
        cmd_smooth = 'python ./scripts/total_variance_rec.py %s %s %s' % (exp_path, exps_path, morphable_model)
        print('\n\nStep %d: Smoothing expression and pose coefficients over time' % (curstep))
        print('=========================================================================================')
        print(cmd_smooth)
        os.system(cmd_smooth)

        
    if smooth_pose_exp and not os.path.exists(poses_path):
        cmd_smooth = 'python ./scripts/total_variance_rec_pose.py %s %s %s' % (pose_path, poses_path, morphable_model)
        print(cmd_smooth)
        os.system(cmd_smooth)
        print('\t > Took %.2f secs' % (time()-t0))

    
    """
    if not os.path.exists(render3d_path) or not os.path.exists(texture_path):
        cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (vid_fpath, cfg_fpath, camera_param, shpsm_fpath, tex_fpath,
                                                                           exp_path, pose_path, illum_path, render3d_path, texture_path)
        rm_texture = 'rm %s' % texture_path

        # Command #5: visualize output
        print(cmd_vis)
        os.system(cmd_vis)
        os.system(rm_texture) # delete texture file -- we don't use it typically
    """

    t0 = time()
    curstep += 1
    if produce_3Drec_videos and not os.path.exists(render3ds_path): # or not os.path.exists(texturefs_path):    
        cmd_vis = './visualize_3Doutput %s %s %s %s %s %s %s %s %s %s' % (vid_fpath, cfg_fpath, camera_param, shpsm_fpath, tex_fpath,
                                                                           exps_path, poses_path, illum_path, render3ds_path, texturefs_path)
        rm_texture = 'rm %s' % texturefs_path

        # Command #5: visualize output
        print('\n\nStep %d: Producing reconstructed face videos' % (curstep))
        print('=============================================')
        print(cmd_vis)
        os.system(cmd_vis)
        os.system(rm_texture) # delete texture file -- we don't use it typically
        
    if produce_3Drec_videos and not os.path.exists(render3d2_path):
        cmd_merge_3D2 = 'python ./scripts/apply_background_2part.py %s %s' % (render3ds_path, render3d2_path)
        
        # Command #5: visualize output
        print(cmd_merge_3D2)
        os.system(cmd_merge_3D2)
    
    if produce_3Drec_videos and not os.path.exists(render3d3_path):
        cmd_merge_3D3 = 'python ./scripts/apply_background_3part.py %s %s' % (render3ds_path, render3d3_path)
        
        # Command #5: visualize output
        print(cmd_merge_3D3)
        os.system(cmd_merge_3D3)
        print('\t > Took %.2f secs' % (time()-t0))


    t0 = time()
    curstep += 1
    if write_2D_landmarks and not os.path.exists(lmks_path):
        print('\n\nStep %d: Producing landmarks video' % (curstep))
        print('===================================')
        cmd_lmks = './write_2Dlandmarks_3DI %s %s %s %s %s %s %s' % (vid_fpath, cfg_fpath, camera_param, shp_fpath, 
                                                                    exps_path, poses_path, lmks_path)
        print(cmd_lmks)
        os.system(cmd_lmks)
        
    curstep += 1
    if write_canonical3D_landmarks and not os.path.exists(canonical_lmks_path):
        
        cmd_canonical_lmks = 'python ./scripts/produce_canonicalized_3Dlandmarks.py %s %s %s' % (exps_path, 
                                                                                            canonical_lmks_path,
                                                                                            morphable_model)
        print('\n\nStep %d: Producing canonicalized landmarks' % (curstep))
        print('===================================')

        os.system(cmd_canonical_lmks)
    
    curstep += 1
    if compute_local_exp_coeffs and not os.path.exists(local_exp_coeffs_path):
        
        cmd_local_exp = 'python ./scripts/compute_local_exp_coefficients.py %s %s %s %s' % (canonical_lmks_path, 
                                                                                            local_exp_coeffs_path,
                                                                                            morphable_model,
                                                                                            local_exp_basis_version)
        print('\n\nStep %d: Computing local expression coefficients' % (curstep))

        print(cmd_local_exp)
        os.system(cmd_local_exp)
    
    if produce_landmark_video and not os.path.exists(lmks_video_path):
        plot_landmarks(vid_fpath, lmks_path, lmks_video_path)
        print('\t > Took %.2f secs' % (time()-t0))
        
        
    render3ds_path_mp4 = render3ds_path.replace('.avi', '.mp4')
    
    if not os.path.exists(render3ds_path_mp4) and os.path.exists(render3ds_path):
        os.system('ffmpeg -i %s %s 2> /dev/null' % (render3ds_path, render3ds_path_mp4))
        os.system('rm %s' % render3ds_path)
        print('Wrapping up -- converting avi to mp4')
            
    return (pose_path, exp_path, illum_path)




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                        prog='process_video',
                        description='Perform 3D reconstruction via 3DI',
                        epilog="""The default parameters are tuned to produce little jitter but they lead to slow processing.
                        You can speed up by (1) setting cfgid to 2, (2) turning off video production 
                        (set produce_3Drec_videos and produce_landmark_video to False) and (3) turning off smoothing (set smooth_pose_exp to False.)
                        """)

    parser.add_argument('out_dir', type=str,
                        help="""Directory that will contain all the processing output. Will be 
                        created if it doesn't exist.""")
    parser.add_argument('video_path', type=str, 
                        help="""Path of the video file to be processed.""")
    parser.add_argument('--camera_param', type=str, default='30', required=False, 
                        help="""Camera parameter, which is either a float or a string.
                        If it is a float, it is the field of view (in degrees) of the camera model
                        to be used during reconstruction. If it is a string, it is
                        the path to the (yaml) file that contains OpenCV camera calibration.""")
    parser.add_argument('--landmark_model', default='global4', type=str,
                        help="""The landmark model to be used. Default value is 'global4', 
                        different landmark models may be supported in the future.""")
    parser.add_argument('--cfgid', default=1, type=int, 
                        help="""The id of configuration file to be used by 3DI. Set to 1 for 
                        the default config file and 2 for a configuration file that's faster 
                        but produces more jittery output.""")
    parser.add_argument('--undistort', default=False, type=str2bool,
                        help="""This will determine whether the video will be undistorted prior
                        to 3D reconstruction. This parameter should be False unless you (1) provide 
                        a string as a camera_param (i.e., a calibration parameter) and (2) the calibration 
                        data contains undistortion parameters.""")
    parser.add_argument('--smooth_pose_exp', default=True, type=str2bool, 
                        help="""Smooth the pose and expression coefficients. Processing speed: ~10fps (very approximately). """)
    parser.add_argument('--produce_3Drec_videos', default=True, type=str2bool,
                        help="""Produce 3D reconstruction videos. Requires the video to be processed on an
                        environment with GUI support. The parameter smooth_pose_exp must be set to True.
                        Takes some time to produce.""")
    parser.add_argument('--write_2D_landmarks', default=True, type=str2bool,
                        help="""Write 2D landmarks produced by 3DI. Requires the smooth_pose_exp parameter to be True.""")
    parser.add_argument('--write_canonical3D_landmarks', default=False, type=str2bool,
                        help="""Write canonicalized 3D landmarks produced by 3DI. Requires the smooth_pose_exp parameter to be True.""")
    parser.add_argument('--produce_landmark_video', default=True, type=str2bool, 
                        help="""Produce video with detected landmarks. Takes some times to produce.""")
    parser.add_argument('--compute_local_exp_coeffs', default=False, type=str2bool, 
                        help="""Compute local basis expression coefficients. This option will need an expression basis
                        (see argument --local_exp_basis_version) and the default argument will be used if not provided. 
                        Also, the --write_canonical3D_landmarks needs to be and will be set to True.""")
    parser.add_argument('--local_exp_basis_version', default='0.0.1.4', type=str, 
                        help="""Version of local expression basis.""")
    

    args = parser.parse_args()
    
    process_single_video(args)

