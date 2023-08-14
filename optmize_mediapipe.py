import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from Configs.configs import get_cfg_defaults
import cv2
from Pose_with_Skeleton.connection_with_unity import Connection_With_Unity,Pose_lib
from skeleton_optimize.keypoint_track import BodyKeypointTrack,show_annotation
from tqdm import tqdm
import numpy as np
from Pose_with_Skeleton.body_keypoint_track import Body_Keypoint_Track
import time


def mediapipe_with_optimize(cfg):
    if cfg.UNITY.IS_CONNECT == True:
        connection_with_unity = Connection_With_Unity(
            ip = cfg.UNITY.ADDRESS,
            port = cfg.UNITY.PORT
        )
    if cfg.MEDPIPE.MODE=="cap":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_rate = 30
    else:
        cap = cv2.VideoCapture(cfg.MEDPIPE.VIDEO_PATH)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    body_keypoint_track = BodyKeypointTrack(
        im_width=frame_width,
        im_height=frame_height,
        fov=cfg.MEDPIPE.CAMERA_FOV*np.pi/180,
        frame_rate=frame_rate,
        track_hands=cfg.MEDPIPE.TRACK_HANDS,
        smooth_range=cfg.MEDPIPE.SMOOTH_RANGE * (1 / frame_rate),
        smooth_range_barycenter=cfg.MEDPIPE.BARYCENTER_SMOOTH_RANGE * (1 / frame_rate)
    )
    frame_t = 0.0
    #bone_euler_sequence, scale_sequence, location_sequence = [], [], []
    bar = tqdm(total=total_frames, desc='Running...')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        body_keypoint_track.track(frame, frame_t)
        kpts3d,valid,unity_data= body_keypoint_track.smooth_3d_keypoints(frame_t)
        show_annotation(frame, kpts3d, valid, body_keypoint_track.K)
        if cv2.waitKey(1) == 27:
            break
        frame_t += 1.0 / frame_rate
        bar.update(1)
        # try:
        #     connection_with_unity.send_dict_to_unity(unity_data)
        # except:
        #     continue

def mediapipe_detect(cfg):
    if cfg.UNITY.IS_CONNECT == True:
        connection_with_unity = Connection_With_Unity(
            ip = cfg.UNITY.ADDRESS,
            port = cfg.UNITY.PORT
        )
    if cfg.MEDPIPE.MODE=="cap":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cfg.MEDPIPE.VIDEO_PATH)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    body_keypoint_track = Body_Keypoint_Track(
    width=frame_width,
    height=frame_height,
    frame_rate=frame_rate,
    track_hands=cfg.MEDPIPE.TRACK_HANDS,  
    track_pose =True,
    smooth_range = 5
    )
    while(cap.isOpened()):
        start_time = time.time()
        success, frame = cap.read()
        # if resizee:
        #     frame = cv2.resize(frame, (frame_width, frame_height))
        if not success:
            break
        body_keypoint_track.Track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if body_keypoint_track.img is None:
            continue
        end_time = time.time()
        cv2.putText(body_keypoint_track.img, str(end_time - start_time), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        cv2.imshow('gg', body_keypoint_track.img[:,:,::-1])
        #start_time = end_time


        if cv2.waitKey(1) & 0xFF == 27:
            break


        try:
            connection_with_unity.send_info_to_unity(body_keypoint_track)
        except:
            continue



if __name__ =="__main__":
    cfg = get_cfg_defaults()
    #mediapipe_with_optimize(cfg)
    mediapipe_detect(cfg)