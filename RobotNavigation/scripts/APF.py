#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import copy
import rospy
from std_msgs.msg import String
import message_filters
import geometry_msgs.msg
import threading
import open3d as o3d


class RFIDRobot(object):

    def __init__(self):
        '''
        定义ROS相关: 节点、订阅者
        '''
        rospy.init_node('APFNavigation')
        rospy.Subscriber('/PointCloud',, queue_size=5)
        rospy.Subscriber('/RFIDTarget',, queue_size=5)
        
        '''
        程序控制参数
        '''
        self.naviFlag = True
        
        '''
        定义机器人与导航相关参数
        '''
        # 初始化参数
        self.secureRange = 0.5  #安全范围
        self.robotW = 1  #机器人宽度
        self.robotL = 1  #机器人长度
        
        # 定义当前坐标、终点坐标
        self.P0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # 机器人当前坐标及速度（相对于起始点，x,y,w，vx,vy,vw）
        self.Pg = np.array([0.0, 0.0, 0.0]) # 目标位置

        # 障碍物位置
        self.Pobs = o3d.geometry.PointCloud()

        # 定义人工势场法相关参数
        self.Eta_att = 5  # 引力的增益系数
        self.Eta_rep_ob = 15  # 斥力的增益系数
        self.d0 = 0.5  # 障碍影响的最大距离
        self.num = 0 #障碍与目标总计个数
        self.len_step = 0.05 # 步长
        self.n=1

        # 历史记录
        self.path = []  # 保存机器人走过的每个点的坐标
        self.unite_vec = np.zeros((self.num,2)) #  保存机器人当前位置与障碍物的单位方向向量，方向指向机器人；以及保存机器人当前位置与目标点的单位方向向量，方向指向目标点


    '''
    定义ROS相关函数: 多线程、订阅者回调函数、rosspin
    '''
    def callbackRFID(data1, data2):
        rospy.loginfo( "I heard %s %s", data1.data,data2.data)
    
    def callbackCamera(data1, data2):
        rospy.loginfo( "I heard %s %s", data1.data,data2.data)
    
    def listening(self):
        rospy.spin()

    def startROS(self):
        #args是关键字参数，需要加上名字，写成args=(self,)
        th1 = threading.Thread(target=RFIDRobot.listening, args=(self,))
        th2 = threading.Thread(target=RFIDRobot.navigate, args=(self,))
        th1.start()
        th2.start()
        th1.join()
        th2.join()

    '''
    定义人工势场导航核心函数
    '''
    def navigate(self):
        while self.naviFlag:
            # 导入需要的数据：机器人位姿，障碍物坐标，目标坐标
            Pi = self.P0[0:3]
            obstacles = np.asarray(self.Pobs.points)
            L = len(obstacles)
            dists = np.zeros((L+1,2)) # 保存机器人当前位置与障碍物的距离以及机器人当前位置与目标点的距离 
            delta = np.zeros((L+1,2)) 
            F_rep_ob = np.zeros((L+1,2))  # 存储每一个障碍到机器人的斥力,带方向
            v=np.linalg.norm(self.P0[4:6]) # 设机器人速度为常值
            unite_vec = np.zeros((L+1,2)) # 引力和斥力的单位方向

            # 判断是否到达目标位置
            if ((Pi[0] - self.Pg[0]) ** 2 + (Pi[1] - self.Pg[1]) ** 2) ** 0.5 < 1:
                    break
            self.path.append(Pi)  
                
            #计算机器人当前位置与障碍物的单位方向向量
            for j in range(L):
                delta[j]=Pi[0:2] - obstacles[j, 0:2]
                dists.append(np.linalg.norm(delta[j]))
                unite_vec[j]=delta[j]/dists[j]
                
            #计算机器人当前位置与目标的单位方向向量
            delta[L]=self.Pg[0:2] - Pi[0:2]
            dists.append(np.linalg.norm(delta[L]))
            unite_vec[L] = delta[L]/dists[L]
                
            # 计算引力
            F_att = self.Eta_att*dists[L]*unite_vec[L]
                
            # 计算斥力
            # 在原斥力势场函数增加目标调节因子（即机器人至目标距离），以使机器人到达目标点后斥力也为0
            for j in  range(L):
                if dists[j] >= self.d0:
                    F_rep_ob[j] = np.array([0, 0])
                else:
                    # 障碍物的斥力1，方向由障碍物指向车辆
                    F_rep_ob1_abs = self.Eta_rep_ob * (1 / dists[j] - 1 / self.d0) * (dists[L])**self.n / dists[j] ** 2  # 斥力大小
                    F_rep_ob1 = F_rep_ob1_abs*unite_vec[j]  # 斥力向量
                    # 障碍物的斥力2，方向由车辆指向目标点
                    F_rep_ob2_abs = self.n/2 * self.Eta_rep_ob * (1 / dists[j] - 1 / self.d0) **2 *(dists[L])**(self.n-1) # 斥力大小
                    F_rep_ob2 = F_rep_ob2_abs * unite_vec[L]  # 斥力向量
                    # 改进后的障碍物合斥力计算
                    F_rep_ob[j] = F_rep_ob1 + F_rep_ob2
                
                
            # 计算合力和方向
            F_rep = np.sum(F_rep_ob, axis=0)
            F_sum = F_att+F_rep
            UnitVec_Fsum = 1 / np.linalg.norm(F_sum) * F_sum
            #计算机器人的下一步位置
            Pi = copy.deepcopy(Pi+ self.len_step * UnitVec_Fsum)


if __name__ == '__main__':
    R1 = RFIDRobot
    R1.startROS()

    


    



