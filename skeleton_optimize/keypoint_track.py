import mediapipe as mp
import numpy as np
import cv2
from typing import List, Tuple
from skeleton_optimize.optimize import intrinsic_from_fov,mls_smooth_numpy,optimize_bones_fit
from Configs.Skleleton_config import MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS,WEIGHTS,MEDIAPIPE_KEYPOINTS_HANDS,MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS_CONNECTIONS,MEDIAPIPE_ALL_KEYPOINTS_CONNECTIONS

class BodyKeypointTrack:
    def __init__(self, im_width: int, im_height: int, fov: float, frame_rate: float, *, track_hands: bool = True, model_complexity=1, smooth_range: float = 0.3, smooth_range_barycenter: float = 1.0):
        self.K = intrinsic_from_fov(fov, im_width, im_height)
        self.im_width, self.im_height = im_width, im_height
        self.frame_delta = 1. / frame_rate

        self.mp_pose_model = mp.solutions.pose.Pose(
            model_complexity=model_complexity, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            static_image_mode=False
            #smooth_landmarks=True,
        )
        self.pose_rvec, self.pose_tvec = None, None
        self.pose_kpts2d = self.pose_kpts3d = None

        self.track_hands = track_hands

        self.mp_hands_model = mp.solutions.hands.Hands(
            max_num_hands=2,
            #model_complexity=model_complexity, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            static_image_mode=False
        )

        self.left_hand_rvec, self.left_hand_tvec = None, None
        self.left_hand_kpts2d = self.left_hand_kpts3d = None
        self.right_hand_rvec, self.right_hand_tvec = None, None
        self.right_hand_kpts2d = self.right_hand_kpts3d = None

        self.barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS])
        self.smooth_range = smooth_range

        self.smooth_range_barycenter = smooth_range_barycenter
        self.origin_pos = None
        self.barycenter_history: List[Tuple[np.ndarray, float]] = []
        self.pose_history: List[Tuple[np.ndarray, float]] = []
        self.left_hand_history: List[Tuple[np.ndarray, float]] = []
        self.right_hand_history: List[Tuple[np.ndarray, float]] = []

    def track(self,image,frame_t):
        self.track_posepoint(image,frame_t)
        if self.track_hands and self.pose_kpts3d is not None:
            self.track_handpoint(image,frame_t)
    def track_posepoint(self,image,frame_t):
        self.pose_kpts2d = self.pose_kpts3d = self.barycenter = None
        results = self.mp_pose_model.process(image)
        if results.pose_landmarks is None:
            return

        image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in results.pose_landmarks.landmark])

        world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark])

        visible = np.array([lm.visibility > 0.2 for lm in results.pose_landmarks.landmark])
        if visible.sum() < 6:
            return 

        kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.pose_rvec, self.pose_tvec)
        if tvec[2] < 0:
            return
        self.pose_kpts2d = image_landmarks

        self.barycenter = np.average(kpts3d, axis=0, weights=self.barycenter_weight)
        self.pose_kpts3d = kpts3d - self.barycenter

        self.pose_rvec, self.pose_tvec = rvec, tvec

        self.barycenter_history.append((self.barycenter, frame_t))

        self.pose_history.append((kpts3d, frame_t))
    def track_handpoint(self,image,frame_t):
        self.left_hand_kpts2d = None
        self.left_hand_kpts3d = None
        self.right_hand_kpts2d = None
        self.right_hand_kpts3d = None
        # run mediapipe hand estimation,
        results = self.mp_hands_model.process(image)
        # get left hand keypoints
        if results.multi_handedness is None:
            return
        print("手部的信息:{}".format(results.multi_handedness))

        num_hands_detected = len(results.multi_handedness)

        left_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Right', range(num_hands_detected)))
        if len(left_hand_id) > 0:
            left_hand_id = left_hand_id[0]

            image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in results.multi_hand_landmarks[left_hand_id].landmark])
            world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[left_hand_id].landmark])
            visible = np.array([True]*len(results.multi_hand_landmarks[left_hand_id].landmark))
            if visible.sum() >= 6:
                kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.left_hand_rvec, self.left_hand_tvec)
                if tvec[2] > 0:
                    self.left_hand_kpts2d = image_landmarks
             
                    self.left_hand_kpts3d = kpts3d + (self.pose_kpts3d[MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index('left_wrist')] - kpts3d[MEDIAPIPE_KEYPOINTS_HANDS.index('wrist_dummy')]).reshape(1, 3)
          
                    self.left_hand_rvec, self.left_hand_tvec = rvec, tvec
                    self.left_hand_history.append((self.left_hand_kpts3d, frame_t))

        right_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Left', range(num_hands_detected)))
        if len(right_hand_id) > 0:
            right_hand_id = right_hand_id[0]

            image_landmarks = np.array([[lm.x * self.im_width, lm.y * self.im_height] for lm in results.multi_hand_landmarks[right_hand_id].landmark])
            world_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[right_hand_id].landmark])
            visible = np.array([True]*len(results.multi_hand_landmarks[right_hand_id].landmark))

            if visible.sum() >= 6:
                kpts3d, rvec, tvec = self._get_camera_space_landmarks(image_landmarks, world_landmarks, visible, self.right_hand_rvec, self.right_hand_tvec)
                if tvec[2] > 0:
                    self.right_hand_kpts2d = image_landmarks

                    self.right_hand_kpts3d = kpts3d + (self.pose_kpts3d[MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index('right_wrist')] - kpts3d[MEDIAPIPE_KEYPOINTS_HANDS.index('wrist_dummy')]).reshape(1, 3)
                    self.right_hand_rvec, self.right_hand_tvec = rvec, tvec
                    self.right_hand_history.append((self.right_hand_kpts3d, frame_t))

    def _get_camera_space_landmarks(self, image_landmarks, world_landmarks, visible, rvec, tvec):

        _, rvec, tvec = cv2.solvePnP(world_landmarks[visible], image_landmarks[visible], self.K, np.zeros(5), rvec=rvec, tvec=tvec, useExtrinsicGuess=rvec is not None)

        rmat, _ = cv2.Rodrigues(rvec)

        kpts3d_cam = world_landmarks @ rmat.T + tvec.T

        kpts3d_cam_z = kpts3d_cam[:, 2].reshape(-1, 1)
        kpts3d_cam[:, :2] =  (np.concatenate([image_landmarks, np.ones((image_landmarks.shape[0], 1))], axis=1) @ np.linalg.inv(self.K).T * kpts3d_cam_z)[:, :2]
        return kpts3d_cam, rvec, tvec
    

    def smooth_3d_keypoints(self,frame_t):

        barycenter_list = [barycenter for barycenter, t in self.barycenter_history if abs(t - frame_t) < self.smooth_range_barycenter]

        barycenter_t = [t for _, t in self.barycenter_history if abs(t - frame_t) < self.smooth_range_barycenter]

        if len(barycenter_t) == 0:
            barycenter = np.zeros(3)
        else:

            barycenter = mls_smooth_numpy(barycenter_t, barycenter_list, frame_t, self.smooth_range_barycenter)

        # Get smoothed pose keypoints

        pose_kpts3d_list = [kpts3d for kpts3d, t in self.pose_history if abs(t - frame_t) < self.smooth_range]

        pose_t = [t for _, t in self.pose_history if abs(t - frame_t) < self.smooth_range]

        pose_kpts3d = None if not any(abs(t - frame_t) < self.frame_delta * 0.6  for t in pose_t) else mls_smooth_numpy(pose_t, pose_kpts3d_list, frame_t, self.smooth_range)
        all_kpts3d = pose_kpts3d if pose_kpts3d is not None else np.zeros((len(MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS), 3))
        all_valid = np.full(len(MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS), pose_kpts3d is not None)

        if self.track_hands:
            # Get smoothed left hand keypoints
            left_hand_kpts3d_list = [kpts3d for kpts3d, t in self.left_hand_history if abs(t - frame_t) < self.smooth_range]
            left_hand_t = [t for _, t in self.left_hand_history if abs(t - frame_t) < self.smooth_range]
            if any(abs(t - frame_t) < self.frame_delta * 0.6 for t in left_hand_t):
                left_hand_kpts3d = barycenter[None, :] + mls_smooth_numpy(left_hand_t, left_hand_kpts3d_list, frame_t, self.smooth_range)
            else:
                left_hand_kpts3d = None
                
            # Get smoothed right hand keypoints
            right_hand_kpts3d_list = [kpts3d for kpts3d, t in self.right_hand_history if abs(t - frame_t) < self.smooth_range]
            right_hand_t = [t for kpts3d, t in self.right_hand_history if abs(t - frame_t) < self.smooth_range]
            if any(abs(t - frame_t) < self.frame_delta * 0.6 for t in right_hand_t):
                right_hand_kpts3d = barycenter[None, :] + mls_smooth_numpy(right_hand_t, right_hand_kpts3d_list, frame_t, self.smooth_range)
            else:
                right_hand_kpts3d = None
            
            all_kpts3d = np.concatenate([
                all_kpts3d,
                left_hand_kpts3d if left_hand_kpts3d is not None else np.zeros((len(MEDIAPIPE_KEYPOINTS_HANDS), 3)),
                right_hand_kpts3d if right_hand_kpts3d is not None else np.zeros((len(MEDIAPIPE_KEYPOINTS_HANDS), 3))
            ], axis=0)

            all_valid = np.concatenate([
                all_valid,
                np.full(len(MEDIAPIPE_KEYPOINTS_HANDS), left_hand_kpts3d is not None),
                np.full(len(MEDIAPIPE_KEYPOINTS_HANDS), right_hand_kpts3d is not None)
            ], axis=0)
        unity_landmark,now_pos = optimize_bones_fit(all_kpts3d,all_valid,self.track_hands,self.origin_pos)
        self.origin_pos = now_pos
        return all_kpts3d, all_valid,unity_landmark




def show_annotation(image, kpts3d, valid, intrinsic):
    annotate_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    kpts3d_homo = kpts3d @ intrinsic.T
    # print(kpts3d[70],valid)
    kpts2d = kpts3d_homo[:, :2] / kpts3d_homo[:,2:]
    #MEDIAPIPE_ALL_KEYPOINTS_CONNECTIONS
    # MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS_CONNECTIONS
    for a, b in MEDIAPIPE_ALL_KEYPOINTS_CONNECTIONS:
        if valid[a] == 0 or valid[b] == 0:
            continue
        cv2.line(annotate_image, (int(kpts2d[a, 0]), int(kpts2d[a, 1])), (int(kpts2d[b, 0]), int(kpts2d[b, 1])), (0, 255, 0), 1)
    for i in range(kpts2d.shape[0]):
        if valid[i] == 0:
            continue
        cv2.circle(annotate_image, (int(kpts2d[i, 0]), int(kpts2d[i, 1])), 2, (0, 0, 255), -1)
    cv2.imshow('Keypoint annotation', annotate_image)