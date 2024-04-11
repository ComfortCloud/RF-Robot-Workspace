import pandas as pd
import numpy as np
import math

class antepos:
    def __init__(self):
        self.x = []
        self.y = []
        self.z = []
        self.timestamp = []

class epc_valid:
    def __init__(self):
        self.id = ""
        self.phase = []
        self.timestamp = []
        self.rssi = []
        self.antepos = antepos()
        self.startpoint = antepos()

class localization_data:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.error = 0
        self.id = ""

class LMcalculate:
    def __init__(self, odomfile, epcfile0, epcfile1, epcfile2, initpos):
        self.odom = pd.read_table(odomfile, sep='\t', header=None, names = ['x', 'y', 'w', 'timestamp'])
        self.epc0 = pd.read_table(epcfile0, sep='\t', header=None, names = ['id', 'phase', 'timestamp', 'rssi'])
        self.epc1 = pd.read_table(epcfile1, sep='\t', header=None, names = ['id', 'phase', 'timestamp', 'rssi'])
        self.epc2 = pd.read_table(epcfile2, sep='\t', header=None, names = ['id', 'phase', 'timestamp', 'rssi'])
        self.initpos = initpos
        tag = []
        tag.append(self.epc0['id'][0])
        for i in range(1, len(self.epc0['id'])):
            if self.epc0['id'][i] != tag[-1]:
                tag.append(self.epc0['id'][i])
        for i in range(0, len(self.epc1['id'])):
            if self.epc1['id'][i] != tag[-1]:
                tag.append(self.epc1['id'][i])
        for i in range(0, len(self.epc2['id'])):
            if self.epc2['id'][i] != tag[-1]:
                tag.append(self.epc2['id'][i])
        self.tag = tag
        print("Get tag id done!")
        # 使用epc_valid类，根据tag的信息，将epc的信息分开
        self.epc_init0 = []
        print("epc_init0 init")
        for i in range(len(tag)):
            self.epc_init0.append(epc_valid())
            self.epc_init0[i].id = tag[i]
            self.epc_init0[i].phase = self.epc0['phase'][self.epc0['id'] == tag[i]]
            self.epc_init0[i].timestamp = self.epc0['timestamp'][self.epc0['id'] == tag[i]]
            self.epc_init0[i].rssi = self.epc0['rssi'][self.epc0['id'] == tag[i]]
        self.epc_init1 = []
        print("1")
        for i in range(len(tag)):
            self.epc_init1.append(epc_valid())
            self.epc_init1[i].id = tag[i]
            self.epc_init1[i].phase = self.epc1['phase'][self.epc1['id'] == tag[i]]
            self.epc_init1[i].timestamp = self.epc1['timestamp'][self.epc1['id'] == tag[i]]
            self.epc_init1[i].rssi = self.epc1['rssi'][self.epc1['id'] == tag[i]]
        self.epc_init2 = []
        print("2")
        for i in range(len(tag)):
            self.epc_init2.append(epc_valid())
            self.epc_init2[i].id = tag[i]
            self.epc_init2[i].phase = self.epc2['phase'][self.epc2['id'] == tag[i]]
            self.epc_init2[i].timestamp = self.epc2['timestamp'][self.epc2['id'] == tag[i]]
            self.epc_init2[i].rssi = self.epc2['rssi'][self.epc2['id'] == tag[i]]
        print("LMcalculate init finished")
    
    def unwrappingPhase(self, epc_init_):
        epc_unwrapped = []
        phase_lb = 1.5
        phase_ub = 0.5
        for tag_init in epc_init_:
            phaseVec = tag_init.phase
            unwrappingPhase = []
            for i in range(len(phaseVec)):
                phaseVec[i] = 2 * math.pi - phaseVec[i] * math.pi / 180
            unwrappingPhase.append(phaseVec[0])
            for i in range(1, len(phaseVec)):
                diff = phaseVec[i] - phaseVec[i-1]
                if diff > phase_lb * math.pi:
                    unwrappingPhase.append(phaseVec[i] - 2 * math.pi)
                elif diff < (-phase_lb * math.pi):
                    unwrappingPhase.append(phaseVec[i] + 2 * math.pi)
                else:
                    unwrappingPhase.append(phaseVec[i])
            phaseVec = np.array(unwrappingPhase)
            unwrappingPhase = []
            unwrappingPhase.append(phaseVec[0])
            for i in range(1, len(phaseVec)):
                diff = phaseVec[i+1] - phaseVec[i]
                if diff > phase_ub * math.pi:
                    unwrappingPhase.append(phaseVec[i] - math.pi)
                elif diff < (-phase_ub * math.pi):
                    unwrappingPhase.append(phaseVec[i] + math.pi)
                else:
                    unwrappingPhase.append(phaseVec[i])
            tag_init.phase = unwrappingPhase
            epc_unwrapped.append(tag_init)
        print("unwrappingPhase done!")
        return epc_unwrapped
    
    def processPhase(self, epc_unwrapped_):
        epc_processed = []
        for tag_unwrapped in epc_unwrapped_:
            phase = tag_unwrapped.phase
            lineDist = []
            timestamp = []
            for i in range(1, len(phase)):
                diff = math.sqrt((phase[i] - phase[i-1])**2 + (tag_unwrapped.timestamp[i] - tag_unwrapped.timestamp[i-1])**2)
                lineDist.append(diff)
            lineDist = np.array(lineDist)
            lineDistAvg = np.mean(lineDist)
            gapIndex = []
            gapIndex.append(0)
            for i in range(1, len(lineDist)):
                if lineDist[i] > lineDistAvg * 3:
                    gapIndex.append(i)
            gapIndex.append(len(phase) - 1)
            if len(gapIndex) == 2:
                tag_unwrapped.phase = phase
                tag_unwrapped.timestamp = self.epc['timestamp']
                tag_unwrapped.rssi = self.epc['rssi']
            else:
                gapMax = gapIndex[1] - gapIndex[0]
                for i in range(2, len(gapIndex)):
                    gapDiff = gapIndex[i] - gapIndex[i-1]
                    if gapDiff > gapMax:
                        gapMax = gapDiff
                        gapMaxIndex = i
                tag_unwrapped.phase = phase[gapIndex[gapMaxIndex-1]:gapIndex[gapMaxIndex]]
                tag_unwrapped.timestamp = tag_unwrapped.timestamp[gapIndex[gapMaxIndex-1]:gapIndex[gapMaxIndex]]
                tag_unwrapped.rssi = self.rssi[gapIndex[gapMaxIndex-1]:gapIndex[gapMaxIndex]]
            if len(tag_unwrapped.phase) >= 40:
                epc_processed.append(tag_unwrapped)
        print("processPhase done!")
        return epc_processed
    
    def getAntePos(self, epc_processed0_, epc_processed1_, epc_processed2_):
        epc_estimated0 = []
        for tag_processed in epc_processed0_:
            antepos_ = antepos()
            temp_antepos = antepos()
            k = 0
            for i in range(len(tag_processed.timestamp)):
                for j in range(k, len(self.odom['timestamp']) - 1):
                    if tag_processed.timestamp[i] >= self.odom['timestamp'][j] and tag_processed.timestamp[i] < self.odom['timestamp'][j+1]:
                        k = j
                        temp_antepos.timestamp.append(tag_processed.timestamp[i])
                        temp_antepos.x.append(self.odom['x'][j] + (self.odom['x'][j+1] - self.odom['x'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.y.append(self.odom['y'][j] + (self.odom['y'][j+1] - self.odom['y'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.z.append(0)
                        break
            for i in range(len(temp_antepos.timestamp)):
                rotate_matrix = np.array([math.cos(self.odom['w'][i]), -math.sin(self.odom['w'][i]), 0, temp_antepos.x[i], 
                                        math.sin(self.odom['w'][i]), math.cos(self.odom['w'][i]), 0, temp_antepos.y[i], 
                                        0, 0, 1, 0, 
                                        0, 0, 0, 1]).reshape(4,4)
                initpos_matrix = np.array([self.initpos[0][0], self.initpos[0][1], self.initpos[0][2], 1]).reshape(4,1)
                antepos_matrix = np.dot(rotate_matrix, initpos_matrix)
                antepos_.x.append(antepos_matrix[0][0])
                antepos_.y.append(antepos_matrix[1][0])
                antepos_.z.append(antepos_matrix[2][0])
                antepos_.timestamp.append(temp_antepos.timestamp[i])
            tag_processed.antepos = antepos_
            epc_estimated0.append(tag_processed)
        
        epc_estimated1 = []
        for tag_processed in epc_processed1_:
            antepos_ = antepos()
            temp_antepos = antepos()
            k = 0
            for i in range(len(tag_processed.timestamp)):
                for j in range(k, len(self.odom['timestamp']) - 1):
                    if tag_processed.timestamp[i] >= self.odom['timestamp'][j] and tag_processed.timestamp[i] < self.odom['timestamp'][j+1]:
                        k = j
                        temp_antepos.timestamp.append(tag_processed.timestamp[i])
                        temp_antepos.x.append(self.odom['x'][j] + (self.odom['x'][j+1] - self.odom['x'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.y.append(self.odom['y'][j] + (self.odom['y'][j+1] - self.odom['y'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.z.append(0)
                        break
            for i in range(len(temp_antepos.timestamp)):
                rotate_matrix = np.array([math.cos(self.odom['w'][i]), -math.sin(self.odom['w'][i]), 0, temp_antepos.x[i], 
                                        math.sin(self.odom['w'][i]), math.cos(self.odom['w'][i]), 0, temp_antepos.y[i], 
                                        0, 0, 1, 0, 
                                        0, 0, 0, 1]).reshape(4,4)
                initpos_matrix = np.array([self.initpos[1][0], self.initpos[1][1], self.initpos[1][2], 1]).reshape(4,1)
                antepos_matrix = np.dot(rotate_matrix, initpos_matrix)
                antepos_.x.append(antepos_matrix[0][0])
                antepos_.y.append(antepos_matrix[1][0])
                antepos_.z.append(antepos_matrix[2][0])
                antepos_.timestamp.append(temp_antepos.timestamp[i])
            tag_processed.antepos = antepos_
            epc_estimated1.append(tag_processed)

        epc_estimated2 = []
        for tag_processed in epc_processed2_:
            antepos_ = antepos()
            temp_antepos = antepos()
            k = 0
            for i in range(len(tag_processed.timestamp)):
                for j in range(k, len(self.odom['timestamp']) - 1):
                    if tag_processed.timestamp[i] >= self.odom['timestamp'][j] and tag_processed.timestamp[i] < self.odom['timestamp'][j+1]:
                        k = j
                        temp_antepos.timestamp.append(tag_processed.timestamp[i])
                        temp_antepos.x.append(self.odom['x'][j] + (self.odom['x'][j+1] - self.odom['x'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.y.append(self.odom['y'][j] + (self.odom['y'][j+1] - self.odom['y'][j]) / (self.odom['timestamp'][j+1] - self.odom['timestamp'][j]) * (tag_processed.timestamp[i] - self.odom['timestamp'][j]))
                        temp_antepos.z.append(0)
                        break
            for i in range(len(temp_antepos.timestamp)):
                rotate_matrix = np.array([math.cos(self.odom['w'][i]), -math.sin(self.odom['w'][i]), 0, temp_antepos.x[i], 
                                        math.sin(self.odom['w'][i]), math.cos(self.odom['w'][i]), 0, temp_antepos.y[i], 
                                        0, 0, 1, 0, 
                                        0, 0, 0, 1]).reshape(4,4)
                initpos_matrix = np.array([self.initpos[2][0], self.initpos[2][1], self.initpos[2][2], 1]).reshape(4,1)
                antepos_matrix = np.dot(rotate_matrix, initpos_matrix)
                antepos_.x.append(antepos_matrix[0][0])
                antepos_.y.append(antepos_matrix[1][0])
                antepos_.z.append(antepos_matrix[2][0])
                antepos_.timestamp.append(temp_antepos.timestamp[i])
            tag_processed.antepos = antepos_
            epc_estimated2.append(tag_processed)
        print("Done")
        return epc_estimated0, epc_estimated1, epc_estimated2
    
    def phaseFunction(self,x,y,z,a,b,c,phase_offset):
        c0 = 3e8
        f = 920.625e6
        Lambda = c0/f
        result = (4 * math.pi * math.sqrt(pow(x-a,2) + pow(y-b,2) + pow(z-c,2)) / Lambda) - phase_offset
        return result
    
    def initStartpoint(self, epc_estimated_):
        epc_init = []
        for tag_estimated in epc_estimated_:
            phase = tag_estimated.phase
            phase = np.array(phase)
            phaseMinIndex = np.argmin(phase)
            tag_estimated.startpoint.x = tag_estimated.antepos.x[phaseMinIndex]
            tag_estimated.startpoint.y = tag_estimated.antepos.y[phaseMinIndex]
            tag_estimated.startpoint.z = tag_estimated.antepos.z[phaseMinIndex]
            epc_init.append(tag_estimated)
        print("Done")
        return epc_init
    
    def LMcore(self, epc0, epc1, epc2):
        localresult = []
        e_min = 0.0005      #步长最小值
        time_max = 500      #最大迭代次数
        u = 0.01            #阻尼因子
        c = 3e8             #光速
        f = 920.625e6       #频率
        I = np.eye(4)       #单位矩阵
        Lambda = c / f      #波长
        for tag_id in self.tag:
            n_epc0 = epc_valid()
            n_epc1 = epc_valid()
            n_epc2 = epc_valid()
            for tag in epc0:
                if tag.id == tag_id:
                    n_epc0 = tag
            for tag in epc1:
                if tag.id == tag_id:
                    n_epc1 = tag
            for tag in epc2:
                if tag.id == tag_id:
                    n_epc2 = tag
            length0 = len(n_epc0.phase)
            length1 = len(n_epc1.phase)
            length2 = len(n_epc2.phase)
            a_iter = n_epc0.startpoint.x
            b_iter = n_epc0.startpoint.y
            c_iter = n_epc0.startpoint.z
            phase_offset0_iter = 4 * math.pi * math.sqrt(pow(n_epc0.antepos.x[0]-a_iter,2) + pow(n_epc0.antepos.y[0]-b_iter,2) + pow(n_epc0.antepos.z[0]-c_iter,2)) / Lambda - n_epc0.phase[0]
            phase_offset1_iter = 4 * math.pi * math.sqrt(pow(n_epc1.antepos.x[0]-a_iter,2) + pow(n_epc1.antepos.y[0]-b_iter,2) + pow(n_epc1.antepos.z[0]-c_iter,2)) / Lambda - n_epc1.phase[0]
            phase_offset2_iter = 4 * math.pi * math.sqrt(pow(n_epc2.antepos.x[0]-a_iter,2) + pow(n_epc2.antepos.y[0]-b_iter,2) + pow(n_epc2.antepos.z[0]-c_iter,2)) / Lambda - n_epc2.phase[0]
            # Calculate the Jacobian matrix
            J = np.zeros((length0+length1+length2, 6))
            for i in range(length0):
                J[i][0] = (n_epc0.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                J[i][1] = (n_epc0.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                J[i][2] = (n_epc0.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                J[i][3] = -1
                J[i][4] = 0
                J[i][5] = 0
            for i in range(length1):
                J[i+length0][0] = (n_epc1.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                J[i+length0][1] = (n_epc1.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                J[i+length0][2] = (n_epc1.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                J[i+length0][3] = 0
                J[i+length0][4] = -1
                J[i+length0][5] = 0
            for i in range(length2):
                J[i+length0+length1][0] = (n_epc2.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                J[i+length0+length1][1] = (n_epc2.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                J[i+length0+length1][2] = (n_epc2.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                J[i+length0+length1][3] = 0
                J[i+length0+length1][4] = 0
                J[i+length0+length1][5] = -1
                     
            phasePredict = []
            for i in range(length0):
                res = self.phaseFunction(n_epc0.antepos.x[i], n_epc0.antepos.y[i], n_epc0.antepos.z[i], a_iter, b_iter, c_iter, phase_offset0_iter)
                phasePredict.append(res)
            for i in range(length1):
                res = self.phaseFunction(n_epc1.antepos.x[i], n_epc1.antepos.y[i], n_epc1.antepos.z[i], a_iter, b_iter, c_iter, phase_offset1_iter)
                phasePredict.append(res)
            for i in range(length2):
                res = self.phaseFunction(n_epc2.antepos.x[i], n_epc2.antepos.y[i], n_epc2.antepos.z[i], a_iter, b_iter, c_iter, phase_offset2_iter)
                phasePredict.append(res)
            phasePredict = np.array(phasePredict)
            phase_origin = np.concatenate((n_epc0.phase, n_epc1.phase, n_epc2.phase))
            phase_origin = np.array(phase_origin)
            # Calculate the residual
            residual = phasePredict - phase_origin
            error_this = np.dot(residual, residual.T)

            # Calculate the Hessian matrix
            H = np.dot(J.T, J)
            # Calculate the next step
            step = np.dot(np.linalg.inv(H + u * I), np.dot(J.T, residual))
            # Update the parameters
            a_iter = a_iter + step[0]
            b_iter = b_iter + step[1]
            c_iter = c_iter + step[2]
            phase_offset0_iter = phase_offset0_iter + step[3]
            phase_offset1_iter = phase_offset1_iter + step[4]
            phase_offset2_iter = phase_offset2_iter + step[5]

            location = []

            for t in range(time_max):
                for i in range(length0):
                    J[i][0] = (n_epc0.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                    J[i][1] = (n_epc0.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                    J[i][2] = (n_epc0.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc0.antepos.x[i]-a_iter,2) + pow(n_epc0.antepos.y[i]-b_iter,2) + pow(n_epc0.antepos.z[i]-c_iter,2)))
                    J[i][3] = -1
                    J[i][4] = 0
                    J[i][5] = 0
                for i in range(length1):
                    J[i+length0][0] = (n_epc1.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                    J[i+length0][1] = (n_epc1.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                    J[i+length0][2] = (n_epc1.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc1.antepos.x[i]-a_iter,2) + pow(n_epc1.antepos.y[i]-b_iter,2) + pow(n_epc1.antepos.z[i]-c_iter,2)))
                    J[i+length0][3] = 0
                    J[i+length0][4] = -1
                    J[i+length0][5] = 0
                for i in range(length2):
                    J[i+length0+length1][0] = (n_epc2.antepos.x[i] - a_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                    J[i+length0+length1][1] = (n_epc2.antepos.y[i] - b_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                    J[i+length0+length1][2] = (n_epc2.antepos.z[i] - c_iter) / (Lambda * math.sqrt(pow(n_epc2.antepos.x[i]-a_iter,2) + pow(n_epc2.antepos.y[i]-b_iter,2) + pow(n_epc2.antepos.z[i]-c_iter,2)))
                    J[i+length0+length1][3] = 0
                    J[i+length0+length1][4] = 0
                    J[i+length0+length1][5] = -1

                # Calculate the predicted phase
                phasePredict = []
                for i in range(length0):
                    res = self.phaseFunction(n_epc0.antepos.x[i], n_epc0.antepos.y[i], n_epc0.antepos.z[i], a_iter, b_iter, c_iter, phase_offset0_iter)
                    phasePredict.append(res)
                for i in range(length1):
                    res = self.phaseFunction(n_epc1.antepos.x[i], n_epc1.antepos.y[i], n_epc1.antepos.z[i], a_iter, b_iter, c_iter, phase_offset1_iter)
                    phasePredict.append(res)
                for i in range(length2):
                    res = self.phaseFunction(n_epc2.antepos.x[i], n_epc2.antepos.y[i], n_epc2.antepos.z[i], a_iter, b_iter, c_iter, phase_offset2_iter)
                    phasePredict.append(res)
                
                # Calculate the residual
                residual = []
                residual = phasePredict - phase_origin

                error_last = error_this
                error_this = np.dot(residual.T, residual)

                # Calculate the Hessian matrix
                H = np.dot(J.T, J)
                # Calculate the next step
                step = np.dot(np.linalg.inv(H + u * I), np.dot(J.T, residual))
                # Update the parameters
                a_iter = a_iter + step[0]
                b_iter = b_iter + step[1]
                c_iter = c_iter + step[2]
                phase_offset0_iter = phase_offset0_iter + step[3]
                phase_offset1_iter = phase_offset1_iter + step[4]
                phase_offset2_iter = phase_offset2_iter + step[5]

                if(error_this < error_last):
                    if(error_last - error_this < e_min):
                        location = [a_iter, b_iter, c_iter]
                        break
                    u = u / 10
                    # Update the parameter
                    a_iter = a_iter + step[0]
                    b_iter = b_iter + step[1]
                    c_iter = c_iter + step[2]
                    phase_offset0_iter = phase_offset0_iter + step[3]
                    phase_offset1_iter = phase_offset1_iter + step[4]
                    phase_offset2_iter = phase_offset2_iter + step[5]
                else:
                    u = u * 10

            print("Tag is:", tag_id)
            print("Location is:", location)
            print("Error is:", error_this)
            tag_local = localization_data()
            tag_local.tag_id = tag_id
            tag_local.x = location[0]
            tag_local.y = location[1]
            tag_local.z = location[2]
            tag_local.error = error_this
            self.local_data.append(tag_local)
        return self.local_data
        
    def run(self):
        unwrappedEpc0 = self.unwrappingPhase(self.epc_init0)
        unwrappedEpc1 = self.unwrappingPhase(self.epc_init1)
        unwrappedEpc2 = self.unwrappingPhase(self.epc_init2)
        processedEpc0 = self.processPhase(unwrappedEpc0)
        processedEpc1 = self.processPhase(unwrappedEpc1)
        processedEpc2 = self.processPhase(unwrappedEpc2)
        estimatedEpc0, estimatedEpc1, estimatedEpc2 = self.getAntePos(processedEpc0, processedEpc1, processedEpc2)
        initEpc0 = self.initStartpoint(estimatedEpc0)
        initEpc1 = self.initStartpoint(estimatedEpc1)
        initEpc2 = self.initStartpoint(estimatedEpc2)
        result = self.LMcore(initEpc0, initEpc1, initEpc2)
        return result