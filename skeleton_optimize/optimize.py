import numpy as np
from typing import List
import cv2
# Return the parameters of camera 
#------Camera's internal parameter matrix------Convert camera coordinates to pixel coordinates
def intrinsic_from_fov(fov: float, width: int, height: int) -> np.ndarray:
    normed_int =  np.array([
        [0.5 / (np.tan(fov / 2) * (width / max(width, height))), 0., 0.5],
        [0., 0.5 / (np.tan(fov / 2) * (height / max(width, height))), 0.5],
        [0., 0., 1.],
    ], dtype=np.float32)
    return normed_int * np.array([width, height, 1], dtype=np.float32).reshape(3, 1)

def mls_smooth_numpy(input_t: List[float], input_y: List[np.ndarray], query_t: float, smooth_range: float):
    # input_t:gravity of frame
    # input_y:the coordinate of center of gravity[ [x1,y1,z1],[x2,y2,z2],[]........,[]]
    # query_t:current frame
    # smooth_range:the range of optimize bones
    # 1-D MLS: input_t: (N), input_y: (..., N), query_t: scalar
    if len(input_y) == 1:
        return input_y[0]
    input_t = np.array(input_t) - query_t
    # input_y:
    input_y = np.stack(input_y, axis=-1)
    broadcaster = (None,)*(len(input_y.shape) - 1)
    w = np.maximum(smooth_range - np.abs(input_t), 0)
    # input_t[broadcaster]:(1,len(input_t))--2dim
    # w[broadcaster]:(1,len(input_t))---2dim
    coef = moving_least_square_numpy(input_t[broadcaster], input_y, w[broadcaster])
    return coef[..., 0]

def moving_least_square_numpy(x: np.ndarray, y: np.ndarray, w: np.ndarray):
    # 1-D MLS: x: (..., N), y: (..., N), w: (..., N)
    # p:(1,2,len(input_t))
    p = np.stack([np.ones_like(x), x], axis=-2)             # (..., 2, N)
    M = p @ (w[..., :, None] * p.swapaxes(-2, -1))
    a = np.linalg.solve(M, (p @ (w * y)[..., :, None]))
    a = a.squeeze(-1)
    return a



# def get_optimization_target(bone_parents: Dict[str, str], skeleton_remap: Dict[str, str], track_hand: bool):
#     optimizable_bones = [skeleton_remap[b] for b in OPTIMIZABLE_BONES if b in skeleton_remap]

#     # target pairs
#     if track_hand:
#         kpt_pairs = [(a, b) for a, b in TARGET_KEYPOINT_PAIRS_WITH_HANDS if a in skeleton_remap and b in skeleton_remap]
#     else:
#         kpt_pairs = [(a, b) for a, b in TARGET_KEYPOINT_PAIRS_WITHOUT_HANDS if a in skeleton_remap and b in skeleton_remap]
#     joint_pairs = [(skeleton_remap[a], skeleton_remap[b]) for a, b in kpt_pairs]
#     # 
#     # Find bones that has target bones as children
#     # bone_subset = [parent_bone,child_1_bone,child_2_bone,child_3_bone.........]
#     bone_subset = []
#     for t in itertools.chain(*joint_pairs):
#         bone_chain = [t]
#         while bone_parents[t] is not None:
#             t = bone_parents[t]
#             # bone_chain:[child_bone,parent_bone]
#             bone_chain.append(t)
#         for b in reversed(bone_chain):
#             if b not in bone_subset:
#                 bone_subset.append(b)
#     if track_hand:
#         kpt_pairs_id = torch.tensor([(MEDIAPIPE_KEYPOINTS_WITH_HANDS.index(a), MEDIAPIPE_KEYPOINTS_WITH_HANDS.index(b)) for a, b in kpt_pairs], dtype=torch.long)
#     else:
#         kpt_pairs_id = torch.tensor([(MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(a), MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS.index(b)) for a, b in kpt_pairs], dtype=torch.long)
#     joint_pairs_id = torch.tensor([(bone_subset.index(a), bone_subset.index(b)) for a, b in joint_pairs], dtype=torch.long)
    
#     return bone_subset, optimizable_bones, kpt_pairs_id, joint_pairs_id


# the keypoint of mediapipe, fit with the model========================
def optimize_bones_fit(kpt3d,valid,track_hands,ori_pos):
    if kpt3d is None:
        return
    unity_landmark = np.zeros((60, 3), dtype = 'float')
    #-hip:0
    unity_landmark[0] = (kpt3d[24] + kpt3d[23]) / 2
    #-left-upperleg:1
    unity_landmark[1] = kpt3d[23]
    #-right-upperleg:2
    unity_landmark[2] = kpt3d[24]
    #-left-knee:3
    unity_landmark[3] = kpt3d[25]
    #-right-knee:4
    unity_landmark[4] = kpt3d[26]
    #-left-foot:5
    unity_landmark[5] = kpt3d[27]
    #-right-foot:6
    unity_landmark[6] = kpt3d[28]
    #-left-toe:7
    unity_landmark[7] = kpt3d[31]
    #-right-toe:8
    unity_landmark[8] = kpt3d[32]

    # original center
    upper_center = unity_landmark[0] - (kpt3d[12] + kpt3d[11]) / 2
    up_center = upper_center / 1000

    unity_landmark[0] -= up_center * 145
    # Spine:9
    unity_landmark[9] = unity_landmark[0] - up_center * 219
    # Chest:10
    unity_landmark[10] = unity_landmark[9] - up_center * 213
    # UpperChest:11
    unity_landmark[11] = unity_landmark[10] - up_center * 198
    unity_landmark[12] = (kpt3d[12] + kpt3d[11]) / 2 
    # Neck的y轴进行微调
    unity_landmark[12, 1] += (unity_landmark[12, 1] - unity_landmark[11, 1] ) / 1.67
    
    #===============Proportionality method to calculate of head coordinate=========================
    #-Head:13
    unity_landmark[13] = unity_landmark[12]
    unity_landmark[13, 1] = (kpt3d[12, 1] + kpt3d[11, 1]) / 2 + (unity_landmark[12, 1] - (kpt3d[12, 1] + kpt3d[11, 1]) / 2) / 0.372
    #-left-eye:14
    unity_landmark[14]= kpt3d[2]
    #-right-eye:15
    unity_landmark[25]= kpt3d[5]
    #jaw-16
    unity_landmark[16]= (kpt3d[10] + kpt3d[9]) / 2
    #-LeftShoulder:17
    unity_landmark[17] = (kpt3d[24] + kpt3d[23]) / 2 + upper_center
    #-RightShoulder:18
    unity_landmark[18] = unity_landmark[17]
    #-LeftUpperArm:19
    unity_landmark[19] = kpt3d[11]
    #-RightUpperArm:20
    unity_landmark[20] = kpt3d[12]
    #-LeftLowerArm:21
    unity_landmark[21] = kpt3d[13]
    #-RightLowerArm: 22
    unity_landmark[22] = kpt3d[14]
    #-LeftHand: 23
    unity_landmark[23]= kpt3d[15]
    #-RightHand: 24
    unity_landmark[24]= kpt3d[16]
    #==================================
    # left_thumb1:25
    unity_landmark[25]= kpt3d[21]
    # left_index1:28
    unity_landmark[28]= kpt3d[19]
    # left_pinky1:37
    unity_landmark[37]= kpt3d[17]
    # right_thumb1:40
    unity_landmark[40]= kpt3d[22]
    # right_index1:43
    unity_landmark[43]= kpt3d[20]
    # right_pinky1:51
    unity_landmark[52]= kpt3d[18]
    if track_hands:
        if valid[34]:
            # left_thumb1:25
            unity_landmark[25]= kpt3d[34]
            # left_thumb2:26
            unity_landmark[26]= kpt3d[35]
            # left_thumb3:27
            unity_landmark[27]= kpt3d[36]
            # left_index1:28
            unity_landmark[28]= kpt3d[38]
            # left_index2:29
            unity_landmark[29]= kpt3d[39]
            # left_index3:30
            unity_landmark[30]= kpt3d[40]
            # left_middle1:31
            unity_landmark[31]= kpt3d[42]
            # left_middle2:32
            unity_landmark[32]= kpt3d[43]
            # left_middle3:33
            unity_landmark[33]= kpt3d[44]
            # left_ring1:34
            unity_landmark[34]= kpt3d[46]
            # left_ring2:35
            unity_landmark[35]= kpt3d[47]
            # left_ring3:36
            unity_landmark[36]= kpt3d[48]
            # left_pinky1:37
            unity_landmark[37]= kpt3d[50]
            # left_pinky2:38
            unity_landmark[38]= kpt3d[51]
            # left_pinky3:39
            unity_landmark[39]= kpt3d[52]
        if valid[55]:
            # right_thumb1:40
            unity_landmark[40]= kpt3d[55]
            # right_thumb2:41
            unity_landmark[41]= kpt3d[56]
            # right_thumb3:42
            unity_landmark[42]= kpt3d[57]
            # right_index1:43
            unity_landmark[43]= kpt3d[59]
            # right_index2:44
            unity_landmark[44]= kpt3d[60]
            # right_index3:45
            unity_landmark[45]= kpt3d[61]
            # right_middle1:46
            unity_landmark[46]= kpt3d[63]
            # right_middle2:47
            unity_landmark[47]= kpt3d[64]
            # right_middle3:48
            unity_landmark[48]= kpt3d[65]
            # right_ring1:49
            unity_landmark[49]= kpt3d[67]
            # right_ring2:50
            unity_landmark[50]= kpt3d[68]
            # right_ring3:51
            unity_landmark[51]= kpt3d[69]
            # right_pinky1:52
            unity_landmark[52]= kpt3d[71]
            # right_pinky2:53
            unity_landmark[53]= kpt3d[72]
            # right_pinky3:54
            unity_landmark[54]= kpt3d[73]
    # nose:55
    unity_landmark[55]= kpt3d[0]
    # lear:56
    unity_landmark[56]= kpt3d[7]
    # rear:57
    unity_landmark[57]= kpt3d[8]
    # mouth:58
    unity_landmark[58]= (kpt3d[9]+kpt3d[10])/2
    ori_pos =ori_pos if ori_pos is not None else unity_landmark[0]

    shift = unity_landmark[0]-ori_pos
    #the shift of hips:59
    unity_landmark[59] = shift
    return unity_landmark,unity_landmark[0]


class KalmanFilterWrapper:
    def __init__(self, input_dim, init_error, init_process_var, init_measure_var):
        self.input_dim = input_dim
        self.init_error = init_error
        self.init_process_var = init_process_var
        self.init_measure_var = init_measure_var
        self.kf,self.statapose= self._create_kalman_filter()

    def _create_kalman_filter(self):
        kf = cv2.KalmanFilter(self.input_dim, self.input_dim)
        kf.transitionMatrix = np.eye(self.input_dim,dtype=np.float32)
        kf.measurementMatrix = np.eye(self.input_dim,dtype=np.float32)
        kf.processNoiseCov = self.init_process_var * np.eye(self.input_dim,dtype=np.float32)
        kf.measurementNoiseCov = self.init_measure_var * np.eye(self.input_dim,dtype=np.float32)
        kf.errorCovPost = self.init_error * np.eye(self.input_dim,dtype=np.float32)
        return kf,kf.statePost

    def filter(self, observation):
        return self.kf.correct(observation)

    def predict(self):
        return self.kf.predict()
        
        
if __name__ == "__main__":
    # a = intrinsic_from_fov(np.pi/3,360,240)
    a = [1,2,3,4,5,6,7,8]
    y = [np.array([[1,3,4],[3,4,6],[5,4,6]]),np.array([[2,3,5],[3,3,2],[3,4,5]]),np.array([[1,2,4],[3,1,2],[2,3,6]]),np.array([[1,3,5],[2,4,6],[5,1,2]]),np.array([[2,1,4],[3,4,6],[5,4,6]]),np.array([[1,3,5],[3,4,6],[5,4,6]]),np.array([[1,2,2],[2,2,6],[5,4,6]]),np.array([[4,1,2],[1,2,1],[4,4,1]])]
    query_t = 9
    smooth_range = 10
    out = mls_smooth_numpy(a,y,query_t,smooth_range)
    print(out)