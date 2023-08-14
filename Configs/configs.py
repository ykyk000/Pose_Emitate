from yacs.config import CfgNode as CN
import numpy as np
_C = CN()
#========config of mmaction========
_C.MMACT=CN()
_C.MMACT.DATA_SETTING = "../mmaction2/demo/demo_configs/tsn_r50_1x1x8_video_infer.py"
# the training of pth文件
_C.MMACT.CHECKPOINT = "./Model_Chechpoint/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth"
# label
_C.MMACT.LABEL_FILE = "../mmaction2/tools/data/kinetics/label_map_k400.txt"
# "gpu:0","cpu"
_C.MMACT.DEVICE = "cpu"
# Camera ID 
_C.MMACT.CAMERA_ID=0
# the thresh of detect
_C.MMACT.THRES = 0.3
_C.MMACT.AVERAGE_SIZE = 2
_C.MMACT.DRAW_FPS = 30
_C.MMACT.INFER_FPS = 4

#==========unity==================
_C.UNITY=CN()
_C.UNITY.IS_CONNECT = True
_C.UNITY.PORT = 5053
_C.UNITY.ADDRESS = "127.0.0.1"
#===========mediapipe====================
_C.MEDPIPE=CN()
# cap的时候设置为False
_C.MEDPIPE.SHOW = False
# MODE: video/cap
_C.MEDPIPE.MODE = "video" 
# cap, [holistic, face, face_mesh, hands, pose ]
_C.MEDPIPE.DETECT_MODE = "pose"
# the mode of video,the address of video
_C.MEDPIPE.VIDEO_PATH = "./Pose_Lib/wave.mp4"

# the fov of camera
_C.MEDPIPE.CAMERA_FOV = 45
_C.MEDPIPE.TRACK_HANDS = False
_C.MEDPIPE.SMOOTH_RANGE = 5
_C.MEDPIPE.BARYCENTER_SMOOTH_RANGE= 20

#=========IK SOLVER=============
_C.IK_SOLVER = CN()
_C.IK_SOLVER.SMOOTH_RANGE = 15

def get_cfg_defaults():
    return _C.clone()