import LMcalculate as LM

epc0 = "/home/haoran/GitHub/Pointcloud-recognize-with-RFID-localization/data/first.txt"
epc1 = "/home/haoran/GitHub/Pointcloud-recognize-with-RFID-localization/data/second.txt"
epc2 = "/home/haoran/GitHub/Pointcloud-recognize-with-RFID-localization/data/third.txt"
odom = "/home/haoran/GitHub/Pointcloud-recognize-with-RFID-localization/data/odom.txt"

pos0 = [-0.017, 0.281, 1.558]
pos1 = [-0.017, 0.282, 1.061]
pos2 = [-0.017, 0.28, 0.59]

initpose = [pos0, pos1, pos2]

LMcalculater = LM.LMcalculate(odom, epc0, epc1, epc2, initpose)
LMcalculater.run()