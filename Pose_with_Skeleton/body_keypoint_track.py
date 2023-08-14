import mediapipe as mp
import numpy as np
import cv2
from collections import deque
from skeleton_optimize.optimize import mls_smooth_numpy,KalmanFilterWrapper
from Configs.Skleleton_config import MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS,WEIGHTS,MEDIAPIPE_KEYPOINTS_HANDS
import cv2
class Body_Keypoint_Track:
    def __init__(self, width:int, height:int, frame_rate: int, track_hands = True, track_pose = True,smooth_range = 5):
        self.width = width
        self.height = height
        self.track_hands = track_hands
        self.track_pose = track_pose
        self.pose_2d, self.pose_3d, self.pose_vis = None, None, None
        self.mp_pose = mp.solutions.pose.Pose(
            model_complexity=1, #default to 1, val from 0 to 2,
            static_image_mode=False, ##video stream, only detect the first very prominent image and track the landmarks until lose
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True,
        )
        self.left_hands_2d, self.right_hands_2d, self.left_hands_3d, self.right_hands_3d, self.left_hands_vis, self.right_hands_vis = None, None, None, None, None, None
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,        
            min_tracking_confidence=0.5,
            min_detection_confidence=0.5,
            max_num_hands = 2,
        )
        self.world_landmark = None
        self.origin_pos = None
        self.temp_world_landmark = None
        self.unity_landmark = None
        self.img = None
        self.smooth_range = smooth_range
        #---------use stack-----------------
        self.queue = []
        # init the calman=========
        #--init body--------
        self.kf_body =KalmanFilterWrapper(33*3,0.05,0.03,0.05)
        # init left hand------
        self.kf_lefthand =KalmanFilterWrapper(21*3,0.1,0.1,0.05)
        # init right hand------
        self.kf_righthand =KalmanFilterWrapper(21*3,0.1,0.1,0.05)
        # use queue
        # self.queue = deque(maxlen=self.smooth_range)
        # self.barycenter_weight = np.array([WEIGHTS.get(kp, 0.) for kp in MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS])
        # self.barycenter_history = []
        
    def Track_pose(self, image):
        self.pose_2d, self.pose_3d = None, None
        results = self.mp_pose.process(image)
        if results.pose_landmarks is None:
            return
        # 2d
        tmp_2d_ldmk = np.array(
            [[a.x * self.width, a.y * self.height, a.z * self.width] for a in results.pose_landmarks.landmark])
        # 3d
        tmp_3d_ldmk = np.array([[a.x , a.y , a.z] for a in results.pose_world_landmarks.landmark])
        visiable = np.array([[a.visibility] for a in results.pose_landmarks.landmark])          

        self.img = image
        mp.solutions.drawing_utils.draw_landmarks(
            image=self.img,
            landmark_list=results.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                           circle_radius=2)
        )
        #self.barycenter = np.average(tmp_2d_ldmk, axis=0, weights=self.barycenter_weight)
        #self.barycenter_history.append(self.barycenter)
        self.pose_2d, self.pose_3d, self.pose_vis = tmp_2d_ldmk, tmp_3d_ldmk, visiable
        #--init the KalmanFilter Initial state-----------------
        if not hasattr(self.kf_body.kf, 'statePost') and self.pose_2d:
            self.kf_body.statapose=np.array(self.pose_2d,dtype=np.float32).flatten()
        #--use KalmanFilter to predict
        self.kf_body.predict()
        #--fixed of input coordinate
        self.kf_body.filter(np.array(self.pose_2d,dtype=np.float32).flatten())
        #--the result of prediction
        self.pose_2d = self.kf_body.kf.statePost.reshape(-1,3)

    def Track_hands(self, image):
        self.left_hands_2d, self.right_hands_2d, self.left_hands_3d, self.right_hands_3d, self.left_hands_vis, self.right_hands_vis = None, None, None, None, None, None
        results = self.mp_hands.process(image)

        if results.multi_handedness is None:      
            return

        numsOfhands = len(results.multi_handedness)
        left_hand_id = list(filter(lambda i: results.multi_handedness[i].classification[0].label == 'Right', range(numsOfhands)))
        right_hand_id = list(filter(lambda i : results.multi_handedness[i].classification[0].label == "Left", range(numsOfhands)))
        if len(left_hand_id) > 0:
            left_hand_id = left_hand_id[0]
            tmp_2d_left_ldmk = np.array([[a.x * self.width, a.y * self.height, a.z * self.width] for a in results.multi_hand_landmarks[left_hand_id].landmark])
            tmp_3d_left_ldmk = np.array([[a.x, a.y, a.z] for a in results.multi_hand_world_landmarks[left_hand_id].landmark])
            visiable = np.array([[a.visibility] for a in results.multi_hand_landmarks[left_hand_id].landmark])
            #print("左手的关键点：{}".format(len(results.multi_hand_landmarks[left_hand_id].landmark)))
            self.left_hands_2d, self.left_hands_3d, self.right_hands_vis = tmp_2d_left_ldmk, tmp_3d_left_ldmk, visiable

            if not hasattr(self.kf_lefthand.kf, 'statePost') and self.left_hands_2d:
                self.kf_lefthand.statapose=np.array(self.left_hands_2d,dtype=np.float32).flatten()

            self.kf_lefthand.predict()

            self.kf_lefthand.filter(np.array(self.left_hands_2d,dtype=np.float32).flatten())

            self.left_hands_2d = self.kf_lefthand.kf.statePost.reshape(-1,3)

            mp.solutions.drawing_utils.draw_landmarks(
            image=self.img,
            landmark_list=results.multi_hand_landmarks[left_hand_id],
            connections=mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0,255), thickness=2,
                                                                            circle_radius=2)
            )
        if len(right_hand_id) > 0:
            right_hand_id = right_hand_id[0]
            tmp_2d_right_ldmk = np.array(
                [[a.x * self.width, a.y * self.height, a.z * self.width] for a in results.multi_hand_landmarks[right_hand_id].landmark])
            tmp_3d_right_ldmk = np.array(
                [[a.x, a.y, a.z] for a in results.multi_hand_world_landmarks[right_hand_id].landmark])
            visiable = np.array([[a.visibility] for a in results.multi_hand_world_landmarks[right_hand_id].landmark])
            self.right_hands_2d, self.right_hands_3d, self.right_hands_vis = tmp_2d_right_ldmk, tmp_3d_right_ldmk, visiable

            if not hasattr(self.kf_righthand.kf, 'statePost') and self.right_hands_2d:
                self.kf_righthand.statapose=np.array(self.right_hands_2d,dtype=np.float32).flatten()

            self.kf_righthand.predict()

            self.kf_righthand.filter(np.array(self.right_hands_2d,dtype=np.float32).flatten())

            self.right_hands_2d = self.kf_righthand.kf.statePost.reshape(-1,3)

            mp.solutions.drawing_utils.draw_landmarks(
            image=self.img,
            landmark_list=results.multi_hand_landmarks[right_hand_id],
            connections=mp.solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0,255), thickness=2,
                                                                            circle_radius=2)
            )

    def optimizer(self):
        if self.pose_2d is None:
            return
        x = np.array([x > 0.2 for x in self.pose_vis])
        if x.sum() < 6:
            self.unity_landmark = None
            return
        #===use slide window filter to optimize bones====================
        #slide window filter
        # if len(self.queue) > 3:
        #     del self.queue[0]
        # self.queue.append(self.unity_landmark)
        # self.unity_landmark = (sum(self.queue)) / len(self.queue)
        #===use MLS to optimize bones=======================
        # mls_smooth_numpy(map(float,list(range(len(self.queue))),list(self.queue),)
        # self.queue.append(self.unity_landmark)

    def merge(self):
        if self.pose_3d is None:
            self.img = None
            return
        self.unity_landmark = np.zeros((62, 3), dtype = 'float')

        # 0 hips:     24+23 / 2
        # 1 LeftUpperLeg 23
        # 2 RightUpperLeg 24
        # 3 LeftLowerLeg 25
        # 4 RightLowerLeg 26
        # 5 LeftFoot 27
        # 6 RightFoot 28

        self.unity_landmark[0] = (self.pose_2d[24] + self.pose_2d[23]) / 2
        self.unity_landmark[1] = self.pose_2d[23]
        self.unity_landmark[2] = self.pose_2d[24]
        self.unity_landmark[3] = self.pose_2d[25]
        self.unity_landmark[4] = self.pose_2d[26]
        self.unity_landmark[5] = self.pose_2d[27]
        self.unity_landmark[6] = self.pose_2d[28]



        # 7 Spine
        # 8 Chest
        # 9 UpperChest
        # 10 Neck
        upper_part_ = self.unity_landmark[0] - (self.pose_2d[12] + self.pose_2d[11]) / 2
        #print(upper_part_)
        up_part = upper_part_ / 1000

        self.unity_landmark[0] -= up_part * 145#167   
        self.unity_landmark[7] = self.unity_landmark[0] - up_part * 219#164
        self.unity_landmark[8] = self.unity_landmark[7] - up_part * 213#173
        self.unity_landmark[9] = self.unity_landmark[8] - up_part * 198#156
        self.unity_landmark[10] = (self.pose_2d[12] + self.pose_2d[11]) / 2  
        self.unity_landmark[10, 1] += (self.unity_landmark[10, 1] - self.unity_landmark[9, 1] ) / 1.67#3.061

        self.unity_landmark[11] = (self.pose_2d[9] + self.pose_2d[10]) / 2
        self.unity_landmark[11, 2] = (self.unity_landmark[11, 2]) - (
        self.unity_landmark[11, 2] - self.unity_landmark[10, 2])  / 5 * 4
        self.unity_landmark[11, 1] = self.unity_landmark[10, 1] - (self.unity_landmark[10, 1] - (self.pose_2d[7, 1] + self.pose_2d[8, 1]) / 2) / 2




        b = self.pose_2d[0] - self.unity_landmark[10]
        a = (self.pose_2d[7] + self.pose_2d[8]) / 2 - self.unity_landmark[10]
        b_norm = np.linalg.norm(b)
        b_unit = b / b_norm
        p = np.dot(a, b_unit)
        self.unity_landmark[11] = (self.unity_landmark[10]) + p * b_unit



        self.unity_landmark[11] = self.unity_landmark[10]
        self.unity_landmark[11, 1] = (self.pose_2d[12, 1] + self.pose_2d[11, 1]) / 2 + (self.unity_landmark[10, 1] - (self.pose_2d[12, 1] + self.pose_2d[11, 1]) / 2) / 0.372


        # 12 LeftShoulder
        # 13 RightShoulder
        # 14 LeftUpperArm
        # 15 RightUpperArm
        # 16 LeftLowerArm
        # 17 RightLowerArm
        # 18 LeftHand
        # 19 RightHand

        _ = (self.pose_2d[24] + self.pose_2d[23]) / 2 + upper_part_
        self.unity_landmark[12] = _
        self.unity_landmark[13] = self.pose_2d[12]
        self.unity_landmark[14] = self.pose_2d[11]
        self.unity_landmark[15] = self.pose_2d[12]
        self.unity_landmark[16] = self.pose_2d[13]
        self.unity_landmark[17] = self.pose_2d[14]
        self.unity_landmark[18] = self.pose_2d[15]
        self.unity_landmark[19] = self.pose_2d[16]

        # 20 LeftToes
        # 21 RightToes 
        # 22 LeftEye   
        # 23 RightEye  
        # 24 Jaw
        self.unity_landmark[20] = self.pose_2d[31]
        self.unity_landmark[21] = self.pose_2d[32]
        self.unity_landmark[22] = self.pose_2d[2]
        self.unity_landmark[23] = self.pose_2d[5]
        self.unity_landmark[24] = (self.pose_2d[10] + self.pose_2d[9]) / 2
        if self.left_hands_2d is not None:
            #self.unity_landmark[18] = self.left_hands_2d[0]
            self.unity_landmark[25] = self.left_hands_2d[1]
            self.unity_landmark[26] = self.left_hands_2d[2]
            self.unity_landmark[27] = self.left_hands_2d[3]
            self.unity_landmark[28] = self.left_hands_2d[5]
            self.unity_landmark[29] = self.left_hands_2d[6]
            self.unity_landmark[30] = self.left_hands_2d[7]
            self.unity_landmark[31] = self.left_hands_2d[9]
            self.unity_landmark[32] = self.left_hands_2d[10]
            self.unity_landmark[33] = self.left_hands_2d[11]
            self.unity_landmark[34] = self.left_hands_2d[13]
            self.unity_landmark[35] = self.left_hands_2d[14]
            self.unity_landmark[36] = self.left_hands_2d[15]
            self.unity_landmark[37] = self.left_hands_2d[17]
            self.unity_landmark[38] = self.left_hands_2d[18]
            self.unity_landmark[39] = self.left_hands_2d[19]
            self.unity_landmark[55] = self.left_hands_2d[0]

        if self.right_hands_2d is not None:
            self.unity_landmark[40] = self.right_hands_2d[1]
            self.unity_landmark[41] = self.right_hands_2d[2]
            self.unity_landmark[42] = self.right_hands_2d[3]
            self.unity_landmark[43] = self.right_hands_2d[5]
            self.unity_landmark[44] = self.right_hands_2d[6]
            self.unity_landmark[45] = self.right_hands_2d[7]
            self.unity_landmark[46] = self.right_hands_2d[9]
            self.unity_landmark[47] = self.right_hands_2d[10]
            self.unity_landmark[48] = self.right_hands_2d[11]
            self.unity_landmark[49] = self.right_hands_2d[13]
            self.unity_landmark[50] = self.right_hands_2d[14]
            self.unity_landmark[51] = self.right_hands_2d[15]
            self.unity_landmark[52] = self.right_hands_2d[17]
            self.unity_landmark[53] = self.right_hands_2d[18]
            self.unity_landmark[54] = self.right_hands_2d[19]
            self.unity_landmark[56] = self.right_hands_2d[0]

        #self.unity_landmark[0] = (self.pose_2d[24] + self.pose_2d[23]) / 2
        pos = self.unity_landmark[0]
        if self.origin_pos is None:
            self.origin_pos = pos

        shift = pos - self.origin_pos
        self.origin_pos = pos
        # hips的偏移
        self.unity_landmark[57] = shift
        #-left-ear
        self.unity_landmark[58] = self.pose_2d[7]
        #-right_ear
        self.unity_landmark[59] = self.pose_2d[8]
        #-nose
        self.unity_landmark[60] = self.pose_2d[0]
        #-mouth
        self.unity_landmark[61] = (self.pose_2d[9] + self.pose_2d[10]) / 2



    def Track(self, image):
        if self.track_pose == True:
            self.Track_pose(image)
        if self.track_hands == True and self.pose_3d is not None:
            self.Track_hands(image)
        self.merge()
        self.optimizer()
