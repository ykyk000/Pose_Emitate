import socket
import sys
import os 
import time
from collections import deque

class Connection_With_Unity:
    def __init__(self, ip = None, port = None, pose = True, hands = True):
        self.ip = ip
        self.port = port
        self.address = (ip, port)
        self.socket = self.connection()
        self.pose = pose
        self.hands = hands

    def connection(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)#
            s.connect(self.address)
            print("Connected to address:", socket.gethostbyname(socket.gethostname()) + ":" + str(self.port))
            print(s)
            return s
        except OSError as e:
            print("Error while connecting :: %s" % e)
            sys.exit()

    def send_info_to_unity(self, body_keypoint_track):
        if body_keypoint_track.unity_landmark is not None:
            b = body_keypoint_track.unity_landmark.flatten() * -1
            a = ' '.join(str(i) for i in b)
            self.socket.send(str.encode(a))
    def send_fixdata_to_unity(self, content):
        self.socket.send(str.encode(content))


    def send_dict_to_unity(self,key_points):
        if key_points is not None:
            b = key_points.flatten() * -1
            a = ' '.join(str(i) for i in b)
            self.socket.send(str.encode(a))


class Pose_lib():
    def __init__(self,lib_path,cfg):
        self.root_path = lib_path
        self.txtname = cfg.MEDPIPE.VIDEO_PATH.split('/')[-1].split('.')[0]+'.txt'
        self.file_path = os.path.join(self.root_path,self.txtname).replace("\\","/")
    def save_data(self,body_keypoint_track):
        f = open(self.file_path, 'a', encoding='utf-8')
        if body_keypoint_track.unity_landmark is not None:
            b = body_keypoint_track.unity_landmark.flatten() * -1
            a = ' '.join(str(i) for i in b)
            f.writelines(a)
            f.write('\n')
    def read_data(self,name):
        file_path = os.path.join(self.root_path,name+".txt").replace("\\","/")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()
        return content
        
