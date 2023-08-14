#=================unity中的骨骼对应=================================
UNITY_SKELETONS_WITH_HANDS= [
    #头部
    "Jaw","LeftEye","RightEye","Head",
    #身体上半部分
    "Hips","Spine","Chest","UpperChest","Neck","LeftShoulder","RightShoulder",
    "LeftUpperArm","RightUpperArm","LeftLowerArm","RightLowerArm","LeftHand","RightHand",
    #腿部
    "LeftUpperLeg","RightUpperLeg","LeftLowerLeg","RightLowerLeg","LeftFoot","RightFoot","LeftToes","RightToes",
    #左手---出现了两次手腕
    "LeftThumbProximal","LeftThumbIntermediate","LeftThumbDistal",
    "LeftIndexProximal","LeftIndexIntermediate","LeftIndexDistal",
    "LeftMiddleProximal","LeftMiddleIntermediate","LeftMiddleDistal",
    "LeftRingProximal","LeftRingIntermediate"," LeftRingDistal",
    "LeftLittleProximal","LeftLittleIntermediate","LeftLittleDistal",
    #右手
    "RightThumbProximal","RightThumbIntermediate","RightThumbDistal",
    "RightIndexProximal","RightIndexIntermediate","RightIndexDistal",
    "RightMiddleProximal","RightMiddleIntermediate","RightMiddleDistal",
    "RightRingProximal","RightRingIntermediate","RightRingDistal",
    "RightLittleProximal","RightLittleIntermediate", "RightLittleDistal"
]
#===============子骨骼和父骨骼一一对应=================================
UNITY_SKELETONS_PARENTS = {
    #头部
    "LeftEye":"Head",
    "RightEye":"Head",
    "Head":"Neck",
    #身体上半部分
    "Hips": None,
    "Spine": "Hips",
    "Chest":"Spine",
    "UpperChest":"Chest",
    "Neck":"UpperChest",
    "LeftShoulder":"UpperChest",
    "RightShoulder":"UpperChest",
    "LeftUpperArm":"LeftShoulder",
    "RightUpperArm":"RightShoulder",
    "LeftLowerArm":"LeftUpperArm",
    "RightLowerArm":"RightUpperArm",
    "LeftHand":"LeftLowerArm",
    "RightHand":"RightLowerArm",
    #腿部
    "LeftUpperLeg":"Hips",
    "RightUpperLeg":"Hips",
    "LeftLowerLeg":"LeftUpperLeg",
    "RightLowerLeg":"RightUpperLeg",
    "LeftFoot":"LeftLowerLeg",
    "RightFoot":"RightLowerLeg",
    "LeftToes":"LeftFoot",
    "RightToes":"RightFoot",
    #左手---出现了两次手腕
    "LeftThumbProximal":"LeftHand",
    "LeftThumbIntermediate":"LeftThumbProximal",
    "LeftThumbDistal":"LeftThumbIntermediate",
    "LeftIndexProximal":"LeftHand",
    "LeftIndexIntermediate":"LeftIndexProximal",
    "LeftIndexDistal":"LeftIndexIntermediate",

    "LeftMiddleProximal":"LeftHand",
    "LeftMiddleIntermediate":"LeftMiddleProximal",
    "LeftMiddleDistal":"LeftMiddleIntermediate",
    "LeftRingProximal":"LeftHand",
    "LeftRingIntermediate":"LeftRingProximal",
    "LeftRingDistal":"LeftRingIntermediate",
    "LeftLittleProximal":"LeftHand",
    "LeftLittleIntermediate":"LeftLittleProximal",
    "LeftLittleDistal":"LeftLittleIntermediate",
    #右手
    "RightThumbProximal":"RightHand",
    "RightThumbIntermediate":"RightThumbProximal",
    "RightThumbDistal":"RightThumbIntermediate",
    "RightIndexProximal":"RightHand",
    "RightIndexIntermediate":"RightIndexProximal",
    "RightIndexDistal":"RightIndexIntermediate",
    "RightMiddleProximal":"RightHand",
    "RightMiddleIntermediate":"RightMiddleProximal",
    "RightMiddleDistal":"RightMiddleIntermediate",
    "RightRingProximal":"RightHand",
    "RightRingIntermediate":"RightRingProximal",
    "RightRingDistal":"RightRingIntermediate",
    "RightLittleProximal":"RightHand",
    "RightLittleIntermediate":"RightLittleProximal", 
    "RightLittleDistal":"RightLittleIntermediate"
}
#=================mediapipe中的骨骼对应=========================================
# 带上了手部的关键点
MEDIAPIPE_KEYPOINTS_WITH_HANDS = [
    #头部
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 
    'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', #16
    'left_pinky_dummy', 'right_pinky_dummy', 'left_index_dummy', 'right_index_dummy', 'left_thumb_dummy', 'right_thumb_dummy',#22
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_toe', 'right_toe',#32

    "left_wrist_dummy", "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
    "left_index1", "left_index2", "left_index3", "left_index4",
    "left_middle1", "left_middle2", "left_middle3", "left_middle4",
    "left_ring1", "left_ring2", "left_ring3", "left_ring4",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky4",

    "right_wrist_dummy", "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
    "right_index1", "right_index2", "right_index3", "right_index4",
    "right_middle1", "right_middle2", "right_middle3", "right_middle4",
    "right_ring1", "right_ring2", "right_ring3", "right_ring4",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky4",
]
#没有带上手部的关键点
MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky1', 'right_pinky1', 'left_index1', 'right_index1', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_toe', 'right_toe'
]
#============UNITY和MEDIAPIP共有的================================
MEDIAPIPE_UNITY_KEYPOINTS_UNION= ['nose','left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_pinky1', 'right_pinky1', 'left_index1', 'right_index1', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_toe', 'right_toe']

MEDIAPIPE_UNITY_KEYPOINTS_UNION_WITH_HANDS =['nose','left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle', 'left_heel', 'right_heel', 'left_toe', 'right_toe',
    "left_thumb1", "left_thumb2", "left_thumb3", "left_thumb4",
    "left_index1", "left_index2", "left_index3", "left_index4",
    "left_middle1", "left_middle2", "left_middle3", "left_middle4",
    "left_ring1", "left_ring2", "left_ring3", "left_ring4",
    "left_pinky1", "left_pinky2", "left_pinky3", "left_pinky4",
    "right_thumb1", "right_thumb2", "right_thumb3", "right_thumb4",
    "right_index1", "right_index2", "right_index3", "right_index4",
    "right_middle1", "right_middle2", "right_middle3", "right_middle4",
    "right_ring1", "right_ring2", "right_ring3", "right_ring4",
    "right_pinky1", "right_pinky2", "right_pinky3", "right_pinky4"
    ]
# 左右手的手部信息共用
# 33
MEDIAPIPE_KEYPOINTS_HANDS = [
    "wrist_dummy", "thumb1", "thumb2", "thumb3", "thumb4",
    "index1", "index2", "index3", "index4",
    "middle1", "middle2", "middle3", "middle4",
    "ring1", "ring2", "ring3", "ring4",
    "pinky1", "pinky2", "pinky3", "pinky4"
]
#关键点的权重
WEIGHTS = {
    'left_ear': 0.04,
    'right_ear': 0.04,
    'left_shoulder': 0.18,
    'right_shoulder': 0.18,
    'left_elbow': 0.02,
    'right_elbow': 0.02,
    'left_wrist': 0.01,
    'right_wrist': 0.01,
    'left_hip': 0.2,
    'right_hip': 0.2,
    'left_knee': 0.03,
    'right_knee': 0.03,
    'left_ankle': 0.02,
    'right_ankle': 0.02,
}

MEDIAPIPE_KEYPOINTS_WITHOUT_HANDS_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                                                (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                                                (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                                                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                                                (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
                                                (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                                                (29, 31), (30, 32), (27, 31), (28, 32)]

MEDIAPIPE_ALL_KEYPOINTS_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                                        (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
                                        (13, 15),
                                        (15,34), (34,35),(35,36),(36,37),
                                        (15,38), (38,39),(39,40),(40,41),
                                        (15,42), (42,43),(43,44),(44,45),
                                        (15,46), (46,47),(47,48),(48,49),
                                        (15,50), (50,51),(51,52),(52,53),
                                        (12, 14), (14, 16), 
                                        (16,55), (55,56),(56,57),(57,58),
                                        (16,59), (59,60),(60,61),(61,62),
                                        (16,63), (63,64),(64,65),(65,66),
                                        (16,67), (67,68),(68,69),(69,70),
                                        (16,71), (71,72),(72,73),(73,74),
                                        (11, 23), (12, 24), (23, 24), (23, 25),
                                        (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
                                        (29, 31), (30, 32), (27, 31), (28, 32)]


if __name__ =="__main__":
    print(len(MEDIAPIPE_KEYPOINTS_WITH_HANDS))