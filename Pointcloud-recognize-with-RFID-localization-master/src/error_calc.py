import numpy as np
import math 
import matplotlib.pyplot as plt

realTag_path='../build/real.txt'
getTag_path='../build/EPCxyz_11_new.txt'

class EpcVec:
    def __init__(self):
        self.epc = ' '
        self.phase = []
        self.x=[]
        self.y=[]
        self.th=[]
        self.timestamp=[]
        self.rssi = []

def unwrappingPhase(phaseVec):
    phaseVec=2*math.pi-np.array(phaseVec)*math.pi/180
    unwrappingphase = []

    phase_lb = 1.5
    phase_ub = 0.5
    diff = 0
    unwrappingphase.append(phaseVec[0])

    for i in range(1, len(phaseVec)):
        diff = phaseVec[i] - phaseVec[i - 1]
        if diff > phase_lb * math.pi:
            unwrappingphase.append(unwrappingphase[i - 1] + diff - 2 * math.pi)
        elif diff < -phase_lb * math.pi:
            unwrappingphase.append(unwrappingphase[i - 1] + diff + 2 * math.pi)
        else:
            unwrappingphase.append(unwrappingphase[i - 1] + diff)
    phaseVec = unwrappingphase
    unwrappingphase=[]
    unwrappingphase.append(phaseVec[0])
    for i in range(1, len(phaseVec)):
        diff = phaseVec[i] - phaseVec[i - 1]
        if diff > phase_ub * math.pi:
            unwrappingphase.append(unwrappingphase[i - 1] + diff - math.pi)
        elif diff < -phase_ub * math.pi:
            unwrappingphase.append(unwrappingphase[i - 1] + diff + math.pi)
        else:
            unwrappingphase.append(unwrappingphase[i - 1] + diff)
    
    return unwrappingphase

def Data_load(firstAntenna,secondAntenna,thirdAntenna):
    # 读取数据
    epc_data_first=[]
    myDataResult_first=[]
    for i in range(len(firstAntenna)):
        epc_data_first.append(firstAntenna[i][1])
        myDataResult_first_i=[]
        for j in range(2,len(firstAntenna[i])):
            myDataResult_first_i.append(float(firstAntenna[i][j]))
        myDataResult_first.append(myDataResult_first_i)
    myDataResult_first=np.array(myDataResult_first)

    epc_data_second=[]
    myDataResult_second=[]
    for i in range(len(secondAntenna)):
        epc_data_second.append(secondAntenna[i][1])
        myDataResult_second_i=[]
        for j in range(2,len(secondAntenna[i])):
            myDataResult_second_i.append(float(secondAntenna[i][j]))
        myDataResult_second.append(myDataResult_second_i)
    myDataResult_second=np.array(myDataResult_second)

    epc_data_third=[]
    myDataResult_third=[]
    for i in range(len(thirdAntenna)):
        epc_data_third.append(thirdAntenna[i][1])
        myDataResult_third_i=[]
        for j in range(2,len(thirdAntenna[i])):
            myDataResult_third_i.append(float(thirdAntenna[i][j]))
        myDataResult_third.append(myDataResult_third_i)
    myDataResult_third=np.array(myDataResult_third)

    EPC="E200-001A-0411-0133-1040-6B0E"
    filter_flag=1
    gap=5
    a=2
    epc_data_first_list=list(set(epc_data_first))
    for i in range(len(epc_data_first_list)):
        epcvec=EpcVec() 
        epcvec.epc=epc_data_first_list[i]
        for j in range(len(myDataResult_first)):
            if epc_data_first_list[i]==epc_data_first[j]:
                epcvec.phase.append(myDataResult_first[j,2])
                epcvec.timestamp.append(myDataResult_first[j,4])
        epcvec.phase=unwrappingPhase(epcvec.phase)
 
        # #选取合适的数据段
        if filter_flag==1:
            phase= epcvec.phase
            minindex = phase.index(min(phase))

            if len(phase)-minindex < 20:
                continue

            
            # 向右查找
            rightindex=len(phase)
            for j in range(minindex,len(phase)-gap):
                if phase[j+gap]-phase[j]<0:
                    rightindex=j
                    break
            # 向左查找
            leftindex=0        
            for j in range(0,minindex-gap):
                if phase[minindex-(j+gap)]-phase[minindex-j]<0:
                    leftindex=minindex-j
                    break

            epcvec.phase=phase[leftindex:rightindex]
            epcvec.timestamp = epcvec.timestamp[leftindex:rightindex]
            epcvec.rssi = epcvec.rssi[leftindex:rightindex]
            epcvec.x = epcvec.x[leftindex:rightindex]
            epcvec.y = epcvec.y[leftindex:rightindex]
            epcvec.th = epcvec.th[leftindex:rightindex]
            
            # 去除断点
            e = []
            e_avg = 0
            e_sum = 0

            n = len(epcvec.phase)
            temp = []
            left_temp = 0
            right_temp = 0

            for j in range(1, len(epcvec.phase)):
                e.append(math.sqrt((epcvec.phase[j] - epcvec.phase[j - 1]) ** 2 + (epcvec.timestamp[j] - epcvec.timestamp[j - 1]) ** 2))
                e_sum += e[j-1]

            e_avg = e_sum / len(e)

            temp.append(0)
            for m in range(len(e)):
                if e[m] > a*e_avg:
                    temp.append(m)

            temp.append(n-1)

            if len(temp) == 2:
                pass
            elif len(temp) == 3:
                if temp[0] - 0 > n-1 - (temp[1]+1):
                    left_temp = 0
                    right_temp = temp[1]
                else:
                    left_temp = temp[1]+1
                    right_temp = n-1
            else:
                temp0 = temp[1] - temp[0]
                left_temp = temp[0]
                right_temp = temp[1]
                for k in range(1, len(temp)):
                    if (temp[k] - temp[k - 1]) > temp0:
                        temp0 = temp[k] - temp[k - 1]
                        left_temp = temp[k - 1]+1
                        right_temp = temp[k]

            epcvec.phase = epcvec.phase[left_temp:right_temp+1]
            epcvec.timestamp = epcvec.timestamp[left_temp:right_temp+1]
            epcvec.rssi = epcvec.rssi[left_temp:right_temp+1]
            epcvec.x = epcvec.x[left_temp:right_temp+1]
            epcvec.y = epcvec.y[left_temp:right_temp+1]
            epcvec.th = epcvec.th[left_temp:right_temp+1]
        # if epcvec.epc=='EPC74':
        #     plt.plot(epcvec.timestamp,epcvec.phase,'r.')
        # if epcvec.epc=='EPC83':
        #     plt.plot(epcvec.timestamp,epcvec.phase,'b.')
        # if epcvec.epc=='EPC34':
        #     plt.plot(epcvec.timestamp,epcvec.phase,'g.')
        # if epcvec.epc=='EPC02':
        #     plt.plot(epcvec.timestamp,epcvec.phase,'y.')
        if epcvec.epc==EPC:
            plt.plot(epcvec.timestamp,epcvec.phase,'k.')

    epc_data_second_list=list(set(epc_data_second))
    for i in range(len(epc_data_second_list)):
        epcvec=EpcVec() 
        epcvec.epc=epc_data_second_list[i]
        for j in range(len(myDataResult_second)):
            if epc_data_second_list[i]==epc_data_second[j]:
                epcvec.phase.append(myDataResult_second[j,2])
                epcvec.timestamp.append(myDataResult_second[j,4])
        epcvec.phase=unwrappingPhase(epcvec.phase)
        if filter_flag==1:
            phase=epcvec.phase
            #选取合适的数据段
            minindex = phase.index(min(phase))

            if len(phase)-minindex < 20:
                continue

            # 向右查找
            rightindex=len(phase)
            for j in range(minindex,len(phase)-gap):
                if phase[j+gap]-phase[j]<0:
                    rightindex=j
                    break
            # 向左查找
            leftindex=0        
            for j in range(0,minindex-gap):
                if phase[minindex-(j+gap)]-phase[minindex-j]<0:
                    leftindex=minindex-j
                    break

            epcvec.phase=phase[leftindex:rightindex]
            epcvec.timestamp = epcvec.timestamp[leftindex:rightindex]
            epcvec.rssi = epcvec.rssi[leftindex:rightindex]
            epcvec.x = epcvec.x[leftindex:rightindex]
            epcvec.y = epcvec.y[leftindex:rightindex]
            epcvec.th = epcvec.th[leftindex:rightindex]

            # if epcvec.epc=='EPC74':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'r.')
            # if epcvec.epc=='EPC83':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'b.')
            # if epcvec.epc=='EPC34':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'g.')
            # if epcvec.epc=='EPC02':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'y.')
            # if epcvec.epc=='EPC10':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'k.')
            
            e = []
            e_avg = 0
            e_sum = 0

            n = len(epcvec.phase)
            temp = []
            left_temp = 0
            right_temp = 0

            for j in range(1, len(epcvec.phase)):
                e.append(math.sqrt((epcvec.phase[j] - epcvec.phase[j - 1]) ** 2 + (epcvec.timestamp[j] - epcvec.timestamp[j - 1]) ** 2))
                e_sum += e[j-1]

            e_avg = e_sum / len(e)

            temp.append(0)
            for m in range(len(e)):
                if e[m] > a*e_avg:
                    temp.append(m)

            temp.append(n-1)

            if len(temp) == 2:
                pass
            elif len(temp) == 3:
                if temp[0] - 0 > n-1 - (temp[1]+1):
                    left_temp = 0
                    right_temp = temp[1]
                else:
                    left_temp = temp[1]+1
                    right_temp = n-1
            else:
                temp0 = temp[1] - temp[0]
                left_temp = temp[0]
                right_temp = temp[1]
                for k in range(1, len(temp)):
                    if (temp[k] - temp[k - 1]) > temp0:
                        temp0 = temp[k] - temp[k - 1]
                        left_temp = temp[k - 1]+1
                        right_temp = temp[k]

            epcvec.phase = epcvec.phase[left_temp:right_temp+1]
            epcvec.timestamp = epcvec.timestamp[left_temp:right_temp+1]
            epcvec.rssi = epcvec.rssi[left_temp:right_temp+1]
            epcvec.x = epcvec.x[left_temp:right_temp+1]
            epcvec.y = epcvec.y[left_temp:right_temp+1]
            epcvec.th = epcvec.th[left_temp:right_temp+1]
        if epcvec.epc==EPC:
            plt.plot(epcvec.timestamp,epcvec.phase,'r.')

    epc_data_third_list=list(set(epc_data_third))
    for i in range(len(epc_data_third_list)):
        epcvec=EpcVec() 
        epcvec.epc=epc_data_third_list[i]
        for j in range(len(myDataResult_third)):
            if epc_data_third_list[i]==epc_data_third[j]:
                epcvec.phase.append(myDataResult_third[j,2])
                epcvec.timestamp.append(myDataResult_third[j,4])
        epcvec.phase=unwrappingPhase(epcvec.phase)
        if filter_flag==1:
            phase=epcvec.phase
            #选取合适的数据段
            minindex = phase.index(min(phase))

            if len(phase)-minindex < 20:
                continue

            # 向右查找
            rightindex=len(phase)
            for j in range(minindex,len(phase)-gap):
                if phase[j+gap]-phase[j]<0:
                    rightindex=j
                    break
            # 向左查找
            leftindex=0        
            for j in range(0,minindex-gap):
                if phase[minindex-(j+gap)]-phase[minindex-j]<0:
                    leftindex=minindex-j
                    break

            epcvec.phase=phase[leftindex:rightindex]
            epcvec.timestamp = epcvec.timestamp[leftindex:rightindex]
            epcvec.rssi = epcvec.rssi[leftindex:rightindex]
            epcvec.x = epcvec.x[leftindex:rightindex]
            epcvec.y = epcvec.y[leftindex:rightindex]
            epcvec.th = epcvec.th[leftindex:rightindex]

            # if epcvec.epc=='EPC74':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'r.')
            # if epcvec.epc=='EPC83':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'b.')
            # if epcvec.epc=='EPC34':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'g.')
            # if epcvec.epc=='EPC02':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'y.')
            # if epcvec.epc=='EPC10':
            #     plt.plot(epcvec.timestamp,epcvec.phase,'k.')
            
            e = []
            e_avg = 0
            e_sum = 0

            n = len(epcvec.phase)
            temp = []
            left_temp = 0
            right_temp = 0

            for j in range(1, len(epcvec.phase)):
                e.append(math.sqrt((epcvec.phase[j] - epcvec.phase[j - 1]) ** 2 + (epcvec.timestamp[j] - epcvec.timestamp[j - 1]) ** 2))
                e_sum += e[j-1]

            e_avg = e_sum / len(e)

            temp.append(0)
            for m in range(len(e)):
                if e[m] > a*e_avg:
                    temp.append(m)

            temp.append(n-1)

            if len(temp) == 2:
                pass
            elif len(temp) == 3:
                if temp[0] - 0 > n-1 - (temp[1]+1):
                    left_temp = 0
                    right_temp = temp[1]
                else:
                    left_temp = temp[1]+1
                    right_temp = n-1
            else:
                temp0 = temp[1] - temp[0]
                left_temp = temp[0]
                right_temp = temp[1]
                for k in range(1, len(temp)):
                    if (temp[k] - temp[k - 1]) > temp0:
                        temp0 = temp[k] - temp[k - 1]
                        left_temp = temp[k - 1]+1
                        right_temp = temp[k]

            epcvec.phase = epcvec.phase[left_temp:right_temp+1]
            epcvec.timestamp = epcvec.timestamp[left_temp:right_temp+1]
            epcvec.rssi = epcvec.rssi[left_temp:right_temp+1]
            epcvec.x = epcvec.x[left_temp:right_temp+1]
            epcvec.y = epcvec.y[left_temp:right_temp+1]
            epcvec.th = epcvec.th[left_temp:right_temp+1]
        if epcvec.epc==EPC:
            plt.plot(epcvec.timestamp,epcvec.phase,'b.')

# plot
plot_flag=1
if plot_flag==1:
    firstAntenna=[]
    secondAntenna=[]
    thirdAntenna=[]
    with open('../experiment/17-1/first.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.split()
            if line!=[]:
                firstAntenna.append(line)
    with open('../experiment/17-1/second.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.split()
            if line!=[]:
                secondAntenna.append(line)
    with open('../experiment/17-1/third.txt','r') as f:
        lines=f.readlines()
        for line in lines:
            line=line.split()
            if line!=[]:
                thirdAntenna.append(line)
    Data_load(firstAntenna,secondAntenna,thirdAntenna)
    plt.show()


realTag_epc=[]
realTag_position=[]
with open(realTag_path,'r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.split()
        if line!=[]:
            realTag_epc.append(line[1])
            realTag_position.append([float(line[2]),float(line[3]),float(line[4])])
realTag_epc=np.array(realTag_epc)
realTag_position=np.array( realTag_position)

getTag_epc=[]
getTag_position=[]
with open(getTag_path,'r') as f:
    lines=f.readlines()
    for line in lines:
        line=line.split()
        if line!=[]:
            getTag_epc.append(line[1])
            getTag_position.append([float(line[2]),float(line[3]),float(line[4])])
getTag_epc=np.array(getTag_epc)
getTag_position=np.array( getTag_position)

error_x_avg=0
error_y_avg=0
error_z_avg=0
error_avg=0
count=0
for i in range(len(getTag_epc)):
    for j in range(len(realTag_epc)):
        if getTag_epc[i]==realTag_epc[j]:
            count=count+1
            error_x= realTag_position[j,0]-round(getTag_position[i,0],3)
            error_y=-realTag_position[j,1]-round(getTag_position[i,1],3)
            error_z=realTag_position[j,2]-round(getTag_position[i,2],3)
            error=np.sqrt(error_x**2+error_y**2+error_z**2)
            if np.abs(error_y)<0.3 and np.abs(error_z)<0.3 :
                print("---------",getTag_epc[i],"---------")
                print(round(getTag_position[i,0],3),"    ",round(getTag_position[i,1],3),"    ",round(getTag_position[i,2],3))
                print(round(error_x,3),"    ",round(error_y,3),"    ",round(error_z,3))      
                error_x_avg += np.abs(error_x)
            error_y_avg += np.abs(error_y)
            error_z_avg += np.abs(error_z)
            error_avg += error
print(count)
error_x_avg=error_x_avg/count
error_y_avg=error_y_avg/count
error_z_avg=error_z_avg/count
error_avg=error_avg/count      
print("error_x_avg : ", error_x_avg)
print("error_y_avg : ", error_y_avg)
print("error_z_avg : ", error_z_avg)
print("error_avg : ", error_avg)
