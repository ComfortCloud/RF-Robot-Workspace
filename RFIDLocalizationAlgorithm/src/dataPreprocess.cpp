#include "dataPreprocess.h"
using namespace Eigen;
using namespace std;

// 载入RFID数据
void MyReaderData::DownloadDataRFID(string s, int id)
{
	// 初始化变量
	vector<string> epc;
	vector<int> rID;
	vector<int> aID;
	vector<double> phase;
	vector<double> timestamp_ros;
	vector<double> unwrappingphase;
	vector<double> rssi;
	fstream fin;
	int id0;
	string epc0;
	int readerID;
	double p;
	double phase0;
	double timestamp_ros0;
	double t;
	double rssi0;
	fin.open(s, ios::in);
	while (1)
	{
		// 读取顺序：序号id0，epc0，读写器ID，readerID，p，相位phase0，rssi0，时间t，时间timestamp_ros0
		fin >> id0;
		if (fin.eof())
			break;
		fin >> epc0;
		epc.push_back(epc0);
		fin >> readerID;
		rID.push_back(readerID);
		fin >> p;
		fin >> phase0;
		phase.push_back(phase0);
		fin >> rssi0;
		rssi.push_back(rssi0);
		fin >> t;
		fin >> timestamp_ros0;
		timestamp_ros.push_back(timestamp_ros0 / 1000000);
	}
	fin.close();

	// 分类数据
	for (int i = 0; i < epc.size(); i++)
	{
		int flag = 0;
		int n = EPCVec.size();
		for (int j = 0; j < n; j++)
		{
			EPCVec[j].reader[0].readerID = 1;
			EPCVec[j].reader[0].ant[id - 1].antennaId = id;
			if (EPCVec[j].epc == epc[i])
			{
				EPCVec[j].reader[0].ant[id - 1].phase.push_back(phase[i]);
				EPCVec[j].reader[0].ant[id - 1].timestamp.push_back(timestamp_ros[i]);
				EPCVec[j].reader[0].ant[id - 1].rssi.push_back(rssi[i]);
				flag = 1;
			}
		}
		if (flag == 0)
		{
			EPCVec.resize(++m);
			EPCVec[n].reader[0].readerID = 1;
			EPCVec[n].reader[0].ant[id - 1].antennaId = id;
			EPCVec[n].epc = epc[i];

			EPCVec[n].reader[0].ant[id - 1].phase.push_back(phase[i]);
			EPCVec[n].reader[0].ant[id - 1].timestamp.push_back(timestamp_ros[i]);
			EPCVec[n].reader[0].ant[id - 1].rssi.push_back(rssi[i]);
		}
	}

	// 相位解缠
	int size0 = 40;
	for (int i = 0; i < EPCVec.size(); i++)
	{
		if (EPCVec[i].reader[0].ant[id - 1].phase.size() > 0)
		{
			unwrappingPhase(i, 0, id);
		}
	}
}


// 载入机器人里程计数据
void MyReaderData::DownloadDataOdom(string s)
{
	fstream fin;
	double x0;
	double y0;
	double yaw0;
	double timestamp_ros0;
	fin.open(s, ios::in);
	while (1)
	{
		fin >> x0;
		if (fin.eof())
			break;
		odom.x.push_back(x0);
		fin >> y0;
		odom.y.push_back(y0);
		fin >> yaw0;
		odom.yaw.push_back(yaw0);
		fin >> timestamp_ros0;
		odom.timestamp.push_back(timestamp_ros0);
	}
	fin.close();

	cout << "Finish reading data!" << endl;
}


// 定义预处理函数
// 包含操作：
void MyReaderData::preprocess(void)
{
	vector<int> id;
	for (int epcNum = 0; epcNum < EPCVec.size(); epcNum++)
	{
		// 数据过滤dataFilter
		for (int i = 0; i < 3; i++)
		{
			if (EPCVec[epcNum].reader[0].ant[i].phase.empty())
			{
				std::cout << "ant " << i + 1 << " has no data." << endl;
			}
			else if (EPCVec[epcNum].reader[0].ant[i].phase.size() > size0)
			{
				// filter_flag.push_back(dataFilter(&reader.ant[i]));
				if (dataFilter(epcNum, i) != 0)
					id.push_back(i);
			}
		}
		cout << "id大小为:" << id.size() << endl;

		// 天线优选
		if (id.size() <= 1)
		{
			EPCVec[epcNum].calcability = 0;
		}
		else if (id.size() == 2)
		{
			EPCVec[epcNum].calcability = 1;
			EPCVec[epcNum].reader[0].antChosen = id;
		}
		else if (id.size() == 3)
		{
			vector<int>().swap(id);
			int phaseSize[] = { EPCVec[epcNum].reader[0].ant[0].phase.size(),
							   EPCVec[epcNum].reader[0].ant[1].phase.size(),
							   EPCVec[epcNum].reader[0].ant[2].phase.size() };
			int size_min = min_element(phaseSize, phaseSize + 3) - phaseSize;

			for (int j = 0; j < 3; j++)
			{
				if (size_min != j)
				{
					EPCVec[epcNum].reader[0].antChosen.push_back(j);
				}
			}
			EPCVec[epcNum].calcability = 1;
		}
		id.clear();


		// 时间戳匹配
		for (int antID = 0; antID < 3; antID++)
		{
			linear1(epcNum, antID);
		}

		// 坐标转换
		for (int antID = 0; antID < 3; antID++)
		{
			AntennaCoordinate(0, antID);
		}
	}
}


// 线性插值
double MyReaderData::linearFitting(double x1, double x2, double y1, double y2, double x)
{
	double y = (y2 - y1) / (x2 - x1) * (x - x1) + y1;

	return y;
}


// 两种时间戳匹配方法，LM中只选了一种用
// 根据机器人里程计时间戳对相位进行插值
void MyReaderData::linear(int epcNum, int antID1, int antID2)
{
	vector<double> LMphase1_get;
	vector<double> LMphase2_get;
	vector<double> LMx_get;
	vector<double> LMy_get;
	vector<double> LMw_get;
	vector<double> LMrssi1_get;
	vector<double> LMrssi2_get;
	double ph1_get, ph2_get, rssi1_get, rssi2_get;

	Antenna* ant1 = &EPCVec[epcNum].reader[0].ant[antID1];
	Antenna* ant2 = &EPCVec[epcNum].reader[0].ant[antID2];

	for (auto k = 0; k < odom.timestamp.size(); k = k + 1)
	{
		if (ant1->timestamp.front() < odom.timestamp[k] &&
			odom.timestamp[k] < ant1->timestamp.back() &&
			ant2->timestamp.front() < odom.timestamp[k] &&
			odom.timestamp[k] < ant2->timestamp.back())
		{
			for (auto i = 0; i < ant1->phase.size(); i++)
			{
				if (ant1->timestamp[i] >= odom.timestamp[k])
				{
					ph1_get = linearFitting(ant1->timestamp[i - 1], ant1->timestamp[i], ant1->phase[i - 1], ant1->phase[i], odom.timestamp[k]);
					rssi1_get = linearFitting(ant1->timestamp[i - 1], ant1->timestamp[i], ant1->rssi[i - 1], ant1->rssi[i], odom.timestamp[k]);
					break;
				}
			}

			for (auto i = 0; i < ant2->phase.size(); i++)
			{
				if (ant2->timestamp[i] >= odom.timestamp[k])
				{
					ph2_get = linearFitting(ant2->timestamp[i - 1], ant2->timestamp[i], ant2->phase[i - 1], ant2->phase[i], odom.timestamp[k]);
					rssi2_get = linearFitting(ant2->timestamp[i - 1], ant2->timestamp[i], ant2->rssi[i - 1], ant2->rssi[i], odom.timestamp[k]);
					break;
				}
			}

			LMphase1_get.push_back(ph1_get);
			LMrssi1_get.push_back(rssi1_get);
			LMphase2_get.push_back(ph2_get);
			LMrssi2_get.push_back(rssi2_get);
			LMx_get.push_back(odom.x[k]);
			LMy_get.push_back(odom.y[k]);
			LMw_get.push_back(odom.yaw[k]);
		}
	}

	// 插值后的数据存入dataset
	EPCVec[epcNum].reader[0].ant[antID1].phase = LMphase1_get;
	EPCVec[epcNum].reader[0].ant[antID1].rssi = LMrssi1_get;
	EPCVec[epcNum].reader[0].ant[antID2].phase = LMphase2_get;
	EPCVec[epcNum].reader[0].ant[antID2].phase = LMrssi2_get;
	odom.x = LMx_get;
	odom.y = LMy_get;
	odom.yaw = LMw_get;
}

void MyReaderData::linear1(int epcNum, int antID)
{
	vector<double> LMphase_get;
	vector<double> LMx_get;
	vector<double> LMy_get;
	vector<double> LMw_get;
	vector<double> LMrssi_get;
	double x_get, y_get, w_get;

	Antenna ant = EPCVec[epcNum].reader[0].ant[antID];

	int n = odom.timestamp.size();
	for (int m = 0; m < ant.timestamp.size(); m = m + 1)
	{
		if (odom.timestamp.front() < ant.timestamp[m] && ant.timestamp[m] < odom.timestamp.back())
		{
			int j = 0;
			for (j = 0; j < n; j++)
			{
				if (ant.timestamp[m] <= odom.timestamp[j])
				{
					j--;
					break;
				}
			}
			x_get = odom.x[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.x[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
			y_get = odom.y[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.y[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
			w_get = odom.yaw[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.yaw[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
			LMx_get.push_back(x_get);
			LMy_get.push_back(y_get);
			LMw_get.push_back(w_get);
			LMphase_get.push_back(ant.phase[m]);
			LMrssi_get.push_back(ant.rssi[m]);
		}
	}

	// 插值后的数据存入dataset
	EPCVec[epcNum].reader[0].ant[antID].phase = LMphase_get;
	EPCVec[epcNum].reader[0].ant[antID].rssi = LMrssi_get;
	odom.x = LMx_get;
	odom.y = LMy_get;
	odom.yaw = LMw_get;
}


// 确定天线的全局坐标
void MyReaderData::AntennaCoordinate(int readerID, int antennaId)
{
	NewCoordinate p;
	vector<double> x = odom.x;
	vector<double> y = odom.y;
	vector<double> w = odom.yaw;

	// 武翀平台数据 ：雷达距离补偿20cm
	double x0[3] = { -0.017 - 0.2, -0.017 - 0.2, -0.017 - 0.2 };
	double y0[3] = { 0.281, 0.282, 0.28 };
	double z0[3] = { 1.558, 1.061, 0.59 };

	Vector4d pVec;
	Matrix4d TRobot;
	Vector4d Ant2Robot;

	if (readerID == 1)
	{
		y0[0] = -y0[0];
		y0[1] = -y0[1];
		y0[2] = -y0[2];
	}

	for (auto i = 0; i < odom.x.size(); i++)
	{
		double w0 = M_PI / 2 + atan(abs(x0[antennaId - 1]) / abs(y0[antennaId - 1]));
		double H = sqrt(pow(x0[antennaId - 1], 2) + pow(y0[antennaId - 1], 2));
		p.x_new.push_back(x[i] + H * cos(-w0 + w[i]));
		p.y_new.push_back(y[i] + H * sin(-w0 + w[i]));
		p.z_new = z0[antennaId - 1];
	}
}