import numpy as np
import sys
import numpy as np
import matplotlib as plt
import random
import math

def phaseFunction(x,y,z,a,b,c,phase_offset):
    c0 = 3e8
    f = 920.625e6
    Lambda = c0/f
    result = (4 * math.pi * math.sqrt(pow(x-a,2) + pow(y-b,2) + pow(z-c,2)) / Lambda) - phase_offset
    return result

def unwrappingPhase(phaseVec):
    unwrappingPhase = []
    length = phaseVec.shape[0]
    print(length)
    for i in range(length):
        phaseVec[i] = 2 * math.pi - phaseVec[i] * math.pi / 180
    phase_lb = 1.5
    phase_ub = 0.5
    unwrappingPhase.append(phaseVec[0])
    for i in range(length-1):
        diff = phaseVec[i+1] - phaseVec[i]
        if diff > phase_lb * math.pi:
            unwrappingPhase.append(unwrappingPhase[i] + diff - 2 * math.pi)
        elif diff < (-phase_lb * math.pi):
            unwrappingPhase.append(unwrappingPhase[i] + diff + 2 * math.pi)
        else:
            unwrappingPhase.append(unwrappingPhase[i] + diff)
    phaseVec = np.array(unwrappingPhase)
    unwrappingPhase = []
    unwrappingPhase.append(phaseVec[0])
    for i in range(length-1):
        diff = phaseVec[i+1] - phaseVec[i]
        if diff > phase_ub * math.pi:
            unwrappingPhase.append(unwrappingPhase[i] + diff - math.pi)
        elif diff < (-phase_ub * math.pi):
            unwrappingPhase.append(unwrappingPhase[i] + diff + math.pi)
        else:
            unwrappingPhase.append(unwrappingPhase[i] + diff)

    return unwrappingPhase

def LMConduct(phase,antPos,x0,y0,z0):
    # 初始化参数
    e_min = 0.0005       #步长最小值
    time_max = 500      #最大迭代次数
    u = 0.01            #阻尼因子
    I = np.eye(4)

    c = 3e8             #光速
    f = 920.625e6       #频率
    Lambda = c / f      #波长

    # 迭代初始值
    length = len(phase)
    phaseOffset0 = 4 * math.pi * math.sqrt(pow(antPos[0,0] - x0,2) + pow(antPos[0,1] - y0,2) + pow(antPos[0,2] - z0,2)) / Lambda - phase[0]

    # 当前迭代值
    xThis = x0
    yThis = y0
    zThis = z0
    phaseOffsetThis = phaseOffset0

    # 计算雅克比行列式
    J = np.zeros((length,4))
    for i in range(length):
        J[i,0] = -4 * math.pi * (antPos[i,0] - xThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
        J[i,1] = -4 * math.pi * (antPos[i,1] - yThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
        J[i,2] = -4 * math.pi * (antPos[i,2] - zThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
        J[i,3] = -1
        
    # 计算残差
    phasePredict = []
    for i in range(length):
        res = phaseFunction(antPos[i,0],antPos[i,1],antPos[i,2],xThis,yThis,zThis,phaseOffsetThis)
        phasePredict.append(res)

    phasePredict = np.array(phasePredict)
    residual = phase - phasePredict
    eThis = np.dot(residual,residual.T)

    # print(eThis,xThis,yThis,zThis,phaseOffsetThis)

    # 初始化下一次迭代值
    # 计算Hessian矩阵 并 更新步长
    H = J.transpose() @ J
    step = np.linalg.inv((H + u * I)) @ J.transpose() @ residual

    # 更新下一次迭代参数
    xNext = xThis + step[0]
    yNext = yThis + step[1]
    zNext = zThis + step[2]
    phaseOffsetNext = phaseOffsetThis + step[3]

    xThis = xNext
    yThis = yNext
    zThis = zNext
    phaseOffsetThis = phaseOffsetNext

    # print(eThis,xThis,yThis,zThis,phaseOffsetThis)
    
    # 开始迭代
    for t in range(time_max):
        # print(t,eThis,xThis,yThis,zThis,phaseOffsetThis)
        # 计算雅克比行列式
        for i in range(length):
            J[i,0] = -4 * math.pi * (antPos[i,0] - xThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
            J[i,1] = -4 * math.pi * (antPos[i,1] - yThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
            J[i,2] = -4 * math.pi * (antPos[i,2] - zThis) / (Lambda * math.sqrt(pow(antPos[i,0] - xThis,2) + pow(antPos[i,1] - yThis,2) + pow(antPos[i,2] - zThis,2)))
            J[i,3] = -1
        
        # 计算残差
        phasePredict = []
        for i in range(length):
            res = phaseFunction(antPos[i,0],antPos[i,1],antPos[i,2],xThis,yThis,zThis,phaseOffsetThis)
            phasePredict.append(res)

        phasePredict = np.array(phasePredict)
        residual = phase - phasePredict

        eNext = np.dot(residual,residual.transpose())
        # print(eNext)

        # 计算Hessian矩阵 并 更新步长
        H = np.matmul(J.transpose(),J)
        step = np.matmul(np.matmul(np.linalg.inv((H + u * I)),J.transpose()),residual)

        # 更新下一次迭代参数
        xNext = xThis + step[0]
        yNext = yThis + step[1]
        zNext = zThis + step[2]
        phaseOffsetNext = phaseOffsetThis + step[3]

        if (eNext < eThis):
            if (eThis - eNext < e_min):
                break
            u = u / 10
            xThis = xNext
            yThis = yNext
            zThis = zNext
            phaseOffsetThis = phaseOffsetNext
            eThis = eNext
        else:
            u = u * 10
    
    location = [xThis, yThis, zThis]

    return location,eThis



'''
def antennaCoordinate(x,y,z,antennaID):
    x1 = 0
    y1 = 0.165
    y2 = 0.145
    z1 = 0.8
    z2 = 0.5
    ones = np.ones((x.shape[0],1))
    p = newCoordinate()
 
class newCoordinate(object):
    def __init__(self):
        self.x_new = np.array([0.0,0.0,0.0])
        self.y_new = np.array([0.0,0.0,0.0])
        self.z_new = np.array([0.0,0.0,0.0])

class myOdomData(object):
    def __init__(self):
        self.x = []
        self.y = []
        self.w = []
        self.timestamp_ros = []

class EPC(object):
    def __init__(self):
        self.epc = ""
        self.phaseVec = []
        self.timestamp = []
        self.rssi = []
        self.phaseLM = []
        self.rssiLM = []
        self.xLM = []
        self.yLM = []

class myReaderData(object):
    def __init__(self):
        self.EPC = []
    
    #def downloadData(self,s,id):
'''
