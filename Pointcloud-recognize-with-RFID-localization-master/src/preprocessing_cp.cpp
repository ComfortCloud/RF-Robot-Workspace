#include "../include/preprocessing_cp.h"

double phaseFunction(double x, double y, double z, double a, double b, double c, double phase_offset)
{
	double c0 = 3e8;
	double f = 920.625e6;
	double lameda = c0 / f;
	double result = (4 * M_PI * sqrt(pow(x - a, 2) + pow(y - b, 2) + pow(z - c, 2)) / lameda)-phase_offset;
	return result;
}

vector<double> unwrappingPhase(vector<double> phaseVec)
{
	vector<double> unwrappingphase;
	for (int i = 0; i < phaseVec.size(); i++)
	{
		phaseVec[i] = 2 * M_PI - phaseVec[i] * M_PI / 180;
	}
	double phase_lb = 1.5;
	double phase_ub = 0.5;
	double diff;
	unwrappingphase.push_back(phaseVec[0]);
	for (int i = 1; i < phaseVec.size(); i++)
	{
		diff = phaseVec[i] - phaseVec[i - 1];
		if (diff > phase_lb * M_PI)
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff - 2 * M_PI);
		}
		else if (diff < (-phase_lb * M_PI))
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff + 2 * M_PI);
		}
		else
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff);
		}
	}
	unwrappingphase.swap(phaseVec);
	vector<double>().swap(unwrappingphase);
	unwrappingphase.push_back(phaseVec[0]);
	for (int i = 1; i < phaseVec.size(); i++)
	{
		diff = phaseVec[i] - phaseVec[i - 1];
		if (diff > phase_ub * M_PI)
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff - M_PI);
		}
		else if (diff < (-phase_ub * M_PI))
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff + M_PI);
		}
		else
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff);
		}
	}
	return unwrappingphase;
}

NewCoordinate AntennaCoordinate(vector<double> x, vector<double> y, double z, vector<double> w, int readerID, int antennaId)
{
	NewCoordinate p;
	// 挑战杯平台数据
	// double x0[3] = { 0.020, 0.020,0.020 };
	// double y0[3] = { 0.340, 0.340,0.340 };
	// double z0[3] = { 1.4775,1.0175, 0.5775 };

	// 师兄平台数据  雷达距离补偿20cm
	double x0[3] = {-0.017-0.2, -0.017-0.2, -0.017-0.2};
	double y0[3] = {0.281, 0.282, 0.28};
	double z0[3] = {1.558, 1.061, 0.59};
	// std::cout << "--6--" << std::endl;

	Vector4d pVec;
	Matrix4d TRobot;
	Vector4d Ant2Robot;

	// std::cout << "--7--" << std::endl;
	if (readerID == 1)
	{
		y0[0] = -y0[0];
		y0[1] = -y0[1];
		y0[2] = -y0[2];
	}

	for (unsigned int i = 0; i < x.size(); i++)
	{
		// TRobot << cos(w[i]), -sin(w[i]), 0, x[i],
		// 	sin(w[i]), cos(w[i]), 0, y[i],
		// 	0, 0, 1, z,
		// 	0, 0, 0, 1;
		// Ant2Robot << x0[antennaId - 1], y0[antennaId - 1], z0[antennaId - 1], 1;
		// pVec = TRobot * Ant2Robot;
		// p.x_new.push_back(pVec[0]);
		// p.y_new.push_back(pVec[1]);
		// p.z_new = pVec[2];

		double w0=M_PI/2+atan(abs(x0[antennaId - 1])/abs(y0[antennaId - 1]));
		double H=sqrt(pow(x0[antennaId - 1],2)+pow(y0[antennaId - 1],2));
		p.x_new.push_back(x[i]+H*cos(-w0+w[i]));
		p.y_new.push_back(y[i]+H*sin(-w0+w[i]));
		p.z_new = z0[antennaId - 1];
	}

	return p;
}

// 两种数据过滤方法dataFilter（最小值段+断裂点查找）和dataFilter1（斜率变化），LM中只选了一种用，目前用的是dataFilter
int dataFilter1(Antenna *ant)
{
	cout << "***begin to filter ....***" << endl;
	int flag = 0;
	vector<double> phase = ant->phase;
	vector<double> time = ant->timestamp;
	vector<double> rssi = ant->rssi;

	std::vector<double> diff;
	diff.push_back(0);
	for (int i = 0; i < phase.size() - 1; i++)
	{
		diff.push_back(sqrt(pow(phase[i + 1] - phase[i], 2) + pow(time[i + 1] - time[i], 2)));
	}
	int phase_min_index = min_element(phase.begin(), phase.end()) - phase.begin();
	int left_index = 0;
	int right_index = phase.size() - 1;
	for (int i = 0; i < diff.size(); i++)
	{
		if (diff[i] > 1.8)
		{
			if (i < phase_min_index)
			{
				left_index = i;
			}
			else
			{
				right_index = i;
				break;
			}
		}
	}

	std::vector<double> phaseMid;
	std::vector<double> timeMid;
	std::vector<double> rssiMid;
	for (int i = left_index; i <= right_index; i++)
	{
		phaseMid.push_back(phase[i]);
		timeMid.push_back(time[i]);
		rssiMid.push_back(rssi[i]);
	}

	std::vector<double> rate;
	if (phaseMid.size() > 10)
	{
		for (int i = 0; i < phaseMid.size() - 2; i++)
		{
			rate.push_back((phaseMid[i + 2] - phaseMid[i]) / (timeMid[i + 2] - timeMid[i]));
		}
		int phaseMid_min_index = min_element(phaseMid.begin(), phaseMid.end()) - phaseMid.begin();
		left_index = 0;
		right_index = phaseMid.size() - 1;

		for (int i = 0; i < rate.size() - 1; i++)
		{
			if (rate[i + 1] * rate[i] < 0)
			{
				if (i < phaseMid_min_index - 2)
				{
					left_index = i + 1;
				}
				if (i > phaseMid_min_index + 2)
				{
					right_index = i;
					break;
				}
			}
		}

		std::vector<double> phaseNew;
		std::vector<double> timeNew;
		std::vector<double> rssiNew;
		for (int i = left_index; i <= right_index; i++)
		{
			phaseNew.push_back(phaseMid[i]);
			timeNew.push_back(timeMid[i]);
			rssiNew.push_back(rssiMid[i]);
		}
		cout << "phaseNew size is " << phaseNew.size() << endl;
		cout << "left_index is " << left_index << endl;
		cout << "right_index is " << right_index << endl;
		cout << "phaseMid_min_index is " << phaseMid_min_index << endl;
		if (right_index - phaseMid_min_index > 1 && phaseMid_min_index - left_index > 1)
		{
			ant->phase = phaseNew;
			ant->timestamp = timeNew;
			ant->rssi = rssiNew;
			flag = 1;
			cout << "The filter phase size is" << ant->phase.size() << endl;
		}
	}
	cout<<"flag="<<flag<<endl;
	return flag;
}

int dataFilter(Antenna *ant)
{
	cout << "***begin to filter ....***" << endl;
	int flag = 0;
	vector<double> phase = ant->phase;
	vector<double> time = ant->timestamp;
	vector<double> rssi = ant->rssi;

	   // 选取合适的数据段
    int minindex = std::distance(phase.begin(), std::min_element(phase.begin(), phase.end()));

    // if (phase.size() - minindex < 10)
    //     return 0;

    int gap = 5;

    // 向右查找
    int rightindex = phase.size();
    for (int j = minindex; j < phase.size() - gap; ++j) {
        if (phase[j + gap] - phase[j] < 0) {
            rightindex = j;
            break;
        }
    }

    // 向左查找
    int leftindex = 0;
    for (int j = 0; j < minindex - gap; ++j) {
        if (phase[minindex - (j + gap)] - phase[minindex - j] < 0) {
            leftindex = minindex - j;
            break;
        }
    }

    phase = std::vector<double>(phase.begin() + leftindex, phase.begin() + rightindex);
    time = std::vector<double>(time.begin() + leftindex, time.begin() + rightindex);
    rssi = std::vector<double>(rssi.begin() + leftindex, rssi.begin() + rightindex);

    // 去除断点
    std::vector<double> e;
    double e_avg = 0;
    double e_sum = 0;

    int n = phase.size();

    std::vector<int> temp;
    int left_temp = 0;
    int right_temp = 0;

    for (int j = 1; j < phase.size(); ++j) {
        e.push_back(sqrt(pow(phase[j] - phase[j - 1], 2) + pow(time[j] - time[j - 1], 2)));
        e_sum += e[j - 1];
    }

    e_avg = e_sum / e.size();

    temp.push_back(0);
    for (int m = 0; m < e.size(); ++m) {
        if (e[m] >  2* e_avg) {
            temp.push_back(m);
        }
    }
    temp.push_back(n - 1);

    if (temp.size() == 2) {
        // pass
		left_temp =temp[0];
		right_temp=temp[1];
    } 
	else if (temp.size() == 3) 
	{
        if (temp[0] - 0 > n - 1 - (temp[1] + 1)) {
            left_temp = 0;
            right_temp = temp[1];
        } else {
            left_temp = temp[1] + 1;
            right_temp = n - 1;
        }
    } 
	else 
	{
        int temp0 = temp[1] - temp[0];
        left_temp = temp[0];
        right_temp = temp[1];
        for (int k = 1; k < temp.size(); ++k) {
            if (temp[k] - temp[k - 1] > temp0) {
                temp0 = temp[k] - temp[k - 1];
                left_temp = temp[k - 1] + 1;
                right_temp = temp[k];
            }
        }
    }

    phase = std::vector<double>(phase.begin() + left_temp, phase.begin() + right_temp + 1);
    time = std::vector<double>(time.begin() + left_temp, time.begin() + right_temp + 1);
    rssi = std::vector<double>(rssi.begin() + left_temp, rssi.begin() + right_temp + 1);

	if (phase.size() > 20)
	{
		ant->phase = phase;
		ant->timestamp = time;
		ant->rssi = rssi;
		flag = 1;
		cout << "The filter phase size is" << ant->phase.size() << endl;
	}
	
	cout<<"flag="<<flag<<endl;
	return flag;
}

void MyReaderData::DownloadData(string s, int id)
{
	vector<string> epc;
	vector<int> rID;
	vector<int> aID;
	vector<double> phase;
	vector<double> timestamp_ros;
	vector<double> unwrappingphase;
	vector<double> rssi;
	fstream fin;
	fin.open(s, ios::in);
	while (1)
	{
		int id0;
		string epc0;
		int readerID;
		int antennaID;
		double p;
		double phase0;
		double timestamp_ros0;
		double t;
		double rssi0;
		fin >> id0;
		if (fin.eof())
			break;
		fin >> epc0;
		epc.push_back(epc0);
		fin >> readerID;
		rID.push_back(readerID);
		// fin >> antennaID;
		// aID.push_back(antennaID);
		fin >> p;
		fin >> phase0;
		phase.push_back(phase0);
		// fin >> timestamp_ros0;
		// timestamp_ros.push_back(timestamp_ros0);
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

	// data initial choose step
	int size0 = 40;
	for (int i = 0; i < EPCVec.size(); i++)
	{
		if (EPCVec[i].reader[0].ant[id - 1].phase.size() > 0)
		{
			EPCVec[i].reader[0].ant[id - 1].phase = unwrappingPhase(EPCVec[i].reader[0].ant[id - 1].phase);
		}
		// cout<<"EPCVec[i].reader[0].ant[id - 1].phase size is----- "<<id<<"-----"<<EPCVec[i].reader[0].ant[id - 1].phase.size()<<endl;
	}
}
// 线性插值
double fitting(double x1, double x2, double y1, double y2, double x)
{
	double y;
	y = (y2 - y1) / (x2 - x1) * (x - x1) + y1;

	return y;
}

// 两种时间戳匹配方法，LM中只选了一种用
tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> linear(Antenna ant1, Antenna ant2, OdomData odom)
{
	vector<double> LMphase1_get;
	vector<double> LMphase2_get;
	vector<double> LMx_get;
	vector<double> LMy_get;
	vector<double> LMw_get;
	vector<double> LMrssi1_get;
	vector<double> LMrssi2_get;
	double ph1_get, ph2_get, rssi1_get, rssi2_get;

	for (int k = 0; k < odom.timestamp.size(); k = k + 1)
	{
		if (ant1.timestamp.front() < odom.timestamp[k] &&
			odom.timestamp[k] < ant1.timestamp.back() &&
			ant2.timestamp.front() < odom.timestamp[k] &&
			odom.timestamp[k] < ant2.timestamp.back())
		{

			for (int i = 0; i < ant1.phase.size(); i++)
			{
				if (ant1.timestamp[i] >= odom.timestamp[k])
				{
					ph1_get = fitting(ant1.timestamp[i - 1], ant1.timestamp[i], ant1.phase[i - 1], ant1.phase[i], odom.timestamp[k]);
					rssi1_get = fitting(ant1.timestamp[i - 1], ant1.timestamp[i], ant1.rssi[i - 1], ant1.rssi[i], odom.timestamp[k]);
					break;
				}
			}

			for (int i = 0; i < ant2.phase.size(); i++)
			{
				if (ant2.timestamp[i] >= odom.timestamp[k])
				{
					ph2_get = fitting(ant2.timestamp[i - 1], ant2.timestamp[i], ant2.phase[i - 1], ant2.phase[i], odom.timestamp[k]);
					rssi2_get = fitting(ant2.timestamp[i - 1], ant2.timestamp[i], ant2.rssi[i - 1], ant2.rssi[i], odom.timestamp[k]);
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

	return make_tuple(LMphase1_get, LMphase2_get, LMx_get, LMy_get, LMw_get, LMrssi1_get, LMrssi2_get);
}

tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> linear1(Antenna ant, OdomData odom)
{
	vector<double> LMphase_get;
	vector<double> LMx_get;
	vector<double> LMy_get;
	vector<double> LMw_get;
	vector<double> LMrssi_get;
	double x_get, y_get, w_get;

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

	return make_tuple(LMphase_get, LMx_get, LMy_get, LMw_get, LMrssi_get);
}

void LM::MakeHessian(Reader reader, OdomData odom)
{
	vector<int> id;
	std::vector<int> filter_flag;
	for (int i = 0; i < 3; i++)
	{
		
		if (reader.ant[i].phase.empty())
		{
			std::cout << "ant " << i + 1 << " has no data." << endl;
		}
		else if (reader.ant[i].phase.size() > size0)
		{
			// filter_flag.push_back(dataFilter(&reader.ant[i]));
			if (dataFilter(&reader.ant[i])!=0)
				id.push_back(i);
		}
	}
	cout << "id大小为:" << id.size() << endl;

	if (id.size() <= 1)
	{
		flag = 0;
	}
	else if (id.size() == 2)
	{
		flag = 1;
	}
	else if (id.size() == 3)
	{
		vector<int>().swap(id);
		int phaseSize[] = {reader.ant[0].phase.size(),
						   reader.ant[1].phase.size(),
						   reader.ant[2].phase.size()};
		int size_min = min_element(phaseSize, phaseSize + 3) - phaseSize;

		for (int j = 0; j < 3; j++)
		{
			if (size_min != j)
			{
				id.push_back(j);
			}
		}
		flag = 1;
	}

	if (flag)
	{
		int flag1 = 1;
		Antenna ant1 = reader.ant[id[0]];
		Antenna ant2 = reader.ant[id[1]];

		// linear fitting
		// tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata1_get = linear(ant1, ant2, odom);
		// vector<double> LMphase1 = get<0>(LMdata1_get);
		// vector<double> LMphase2 = get<1>(LMdata1_get);
		// vector<double> LMx = get<2>(LMdata1_get);
		// vector<double> LMy = get<3>(LMdata1_get);
		// vector<double> LMw = get<4>(LMdata1_get);
		// vector<double> LMrssi1 = get<5>(LMdata1_get);
		// vector<double> LMrssi2 = get<6>(LMdata1_get);

		// linear fitting
		cout<<"Linear fitting"<<endl;
		tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata1_get = linear1(ant1, odom);
		vector<double> LMphase1 = get<0>(LMdata1_get);
		vector<double> LMx1 = get<1>(LMdata1_get);
		vector<double> LMy1 = get<2>(LMdata1_get);
		vector<double> LMw1 = get<3>(LMdata1_get);
		vector<double> LMrssi1 = get<4>(LMdata1_get);
		tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata2_get = linear1(ant2, odom);
		vector<double> LMphase2 = get<0>(LMdata2_get);
		vector<double> LMx2 = get<1>(LMdata2_get);
		vector<double> LMy2 = get<2>(LMdata2_get);
		vector<double> LMw2 = get<3>(LMdata2_get);
		vector<double> LMrssi2 = get<4>(LMdata2_get);

		// std::cout << "--1--" << std::endl;
		// cout << LMphase1.size() << endl;
		if (LMphase1.size() == 0 || LMphase2.size() == 0)
		{
			flag1 = 0;
			flag = 0;
		}
		cout<<"flag1="<<flag1<<endl;

		if (flag1 == 1)
		{
			double phase_diff1 = LMphase1[0];
			double phase_diff2 = LMphase2[0];

			// vector->eigen
			Eigen::VectorXd phase1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(LMphase1.data(), LMphase1.size());
			Eigen::VectorXd phase2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(LMphase2.data(), LMphase2.size());
			// std::cout << "--2--" << std::endl;
			cout<<"phase1 size="<<LMphase1.size()<<endl;
			NewCoordinate position1 = AntennaCoordinate(LMx1, LMy1, z, LMw1, reader.readerID, ant1.antennaId);
			vector<double> x1 = position1.x_new;
			vector<double> y1 = position1.y_new;
			double z1 = position1.z_new;
			
			NewCoordinate position2 = AntennaCoordinate(LMx2, LMy2, z, LMw2, reader.readerID, ant2.antennaId);
			vector<double> x2 = position2.x_new;
			vector<double> y2 = position2.y_new;
			double z2 = position2.z_new;
			// cout<<"**************************x1***************"<<endl;
			// for(int i=0;i<x1.size();i++)
			// {
			// 	cout<<x1[i]<<endl;
			// }
			// cout<<"**************************x2***************"<<endl;
			// for(int i=0;i<x2.size();i++)
			// {
			// 	cout<<x2[i]<<endl;
			// }
			int minPosition1 = min_element(LMphase1.begin(), LMphase1.end()) - LMphase1.begin();
			int minPosition2 = min_element(LMphase2.begin(), LMphase2.end()) - LMphase2.begin();
			double a0, b0, c0;
			cout<<"minPosition1"<<minPosition1<<endl;
			cout<<"minPosition2"<<minPosition2<<endl;
			cout << "x1min"<<x1[minPosition1]<<endl;
			cout<<"x2min"<<x2[minPosition2]<<endl;
			// cout<<LMy[minPosition1]<<endl;
			a0 = (x1[minPosition1] + x2[minPosition2]) / 2;
			
			// 这里的y_delta是手动修正一下初始点 需要的话加上 注意正负号表示不同方向的迁移
			if (reader.readerID == 2)
				b0 = (y1[minPosition1] + y2[minPosition2]) / 2 -0*y_delta;
			if (reader.readerID == 1)
				b0 = (y1[minPosition1] + y2[minPosition2]) / 2 -0*y_delta;
			// if(b0<0)
			// 	b0=b0-y_delta;
			// else if(b0>0)
			// 	b0=b0+y_delta;
			// if (LMw1[minPosition1]>3)
			// 	b0=b0+y_delta;
			// else
			// 	b0=b0-y_delta;

			double LMrssi1_avg = 0, LMrssi2_avg = 0;
			for (int i = 0; i < LMrssi1.size(); i++)
			{
				LMrssi1_avg += LMrssi1[i];
				LMrssi2_avg += LMrssi2[i];
			}
			LMrssi1_avg = LMrssi1_avg / LMrssi1.size();
			LMrssi2_avg = LMrssi2_avg / LMrssi2.size();
			// cout << "LMrssi1_avg:" << LMrssi1_avg << endl;
			// cout << "LMrssi2_avg:" << LMrssi2_avg << endl;
			if (LMrssi1_avg > LMrssi2_avg)
			//if (LMrssi1[minPosition1] > LMrssi2[minPosition2])
				c0 = z1;
			else
				c0 = z2;
			//c0=(z1+z2)/2;
			// std::cout << "--4--" << std::endl;

			double phase_offset1_0 = 4 * M_PI * sqrt(pow(x1[0] - a0, 2) + pow(y1[0] - b0, 2) + pow(z1 - c0, 2)) / lameda - phase1(0);
			double phase_offset2_0 = 4 * M_PI * sqrt(pow(x2[0] - a0, 2) + pow(y2[0] - b0, 2) + pow(z2 - c0, 2)) / lameda - phase2(0);
			a_it = a0;
			b_it = b0;
			c_it = c0;
			phase_offset1_it = phase_offset1_0;
			phase_offset2_it = phase_offset2_0;

			int n1 = x1.size();
			int n2 = x2.size();
			MatrixXd J = MatrixXd::Zero(n1 + n2, 5);
			double e, e_new;

			VectorXd phase(n1 + n2);

			// std::cout << "--5--" << std::endl;

			for (int i = 0; i < n1; i++)
			{
				phase(i) = phase1(i);
			}
			for (int i = 0; i < n2; i++)
			{
				phase(i + n1) = phase2(i);
			}

			VectorXd step;
			MatrixXd H;
			VectorXd d_new;
			VectorXd phase_new(n1 + n2);
			VectorXd d;
			int updateJ = 1;
			for (int k = 1; k <= k_max; k++)
			{
				if (updateJ == 1)
				{
					for (int i = 0; i < n1; i++)
					{
						J(i, 0) = -4 * M_PI * (x1[i] - a_it) / (lameda * sqrt(pow(x1[i] - a_it, 2) + pow(y1[i] - b_it, 2) + pow(z1 - c_it, 2)));
						J(i, 1) = -4 * M_PI * (y1[i] - b_it) / (lameda * sqrt(pow(x1[i] - a_it, 2) + pow(y1[i] - b_it, 2) + pow(z1 - c_it, 2)));
						J(i, 2) = -4 * M_PI * (z1 - c_it) / (lameda * sqrt(pow(x1[i] - a_it, 2) + pow(y1[i] - b_it, 2) + pow(z1 - c_it, 2)));
						J(i,3)=-1;
						J(i,4)=0;
					}
					for (int i = 0; i < n2; i++)
					{
						J(i + n1, 0) = -4 * M_PI * (x2[i] - a_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 1) = -4 * M_PI * (y2[i] - b_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 2) = -4 * M_PI * (z2 - c_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i,3)=0;
						J(i,4)=-1;
					}
				}

				// std::cout << "--9--" << std::endl;

				if (k == 1)
				{
					VectorXd phase_res(n1 + n2);
					for (int i = 0; i < n1; i++)
					{
						phase_res(i) = phaseFunction(x1[i], y1[i], z1, a0, b0, c0,phase_offset1_0);
					}
					for (int i = 0; i < n2; i++)
					{
						phase_res(i + n1) = phaseFunction(x2[i], y2[i], z2, a0, b0, c0,phase_offset2_0);
					}
					// phase_diff1 = phase_res(0);
					// phase_diff2 = phase_res(n1);
					// for (int i = 0; i < n1; i++)
					// {
					// 	phase_res(i) = phase_res(i) - phase_diff1;
					// }
					// for (int i = 0; i < n2; i++)
					// {
					// 	phase_res(i + n1) = phase_res(i + n1) - phase_diff2;
					// }

					d = phase - phase_res;
					H = J.transpose() * J;
					step = (H + v * I).inverse() * J.transpose() * d;

					e = d.dot(d);
					continue;
				}

				H = J.transpose() * J;

				a_new = a_it + step(0);
				b_new = b_it + step(1);
				c_new = c_it + step(2);
				phase_offset1_new = phase_offset1_it + step(3);
				phase_offset2_new = phase_offset2_it + step(4);

				for (int i = 0; i < n1; i++)
				{
					phase_new(i) = phaseFunction(x1[i], y1[i], z1, a_new, b_new, c_new, phase_offset1_new);
				}
				for (int i = 0; i < n2; i++)
				{
					phase_new(i + n1) = phaseFunction(x2[i], y2[i], z2, a_new, b_new, c_new, phase_offset2_new);
				}

				// phase_diff1 = phase_new(0);
				// phase_diff2 = phase_new(n1);
				// for (int i = 0; i < n1; i++)
				// {
				// 	phase_new(i) = phase_new(i) - phase_diff1;
				// }
				// for (int i = 0; i < n2; i++)
				// {
				// 	phase_new(i + n1) = phase_new(i + n1) - phase_diff2;
				// }
				// std::cout << "--12--" << std::endl;

				d_new = phase - phase_new;
				e_new = d_new.dot(d_new);
				step = (H + v * I).inverse() * J.transpose() * d_new;

				if (e_new <= e)
				{
					if (step.dot(step) < e_min)
						break;
					updateJ = 1;
					v = v / 10;
					a_it = a_new;
					b_it = b_new;
					c_it = c_new;
					phase_offset1_it = phase_offset1_new;
					phase_offset2_it = phase_offset2_new;
					e = e_new;
					std::cout << "Next loop at point:" << std::fixed << a_new << "  " << std::fixed << b_new << "  " << std::fixed << c_new << std::endl;
					std::cout << "e_new=" << e_new << std::endl;
				}
				else
				{
					updateJ = 0;
					v = v * 10;
					// std::cout << v << std::endl;
				}
			}
			cout << "Iteration starts at point:" << std::fixed << a0+0.2 << "  " << std::fixed << b0 << "  " << std::fixed << c0 << endl;
			cout << "Iteration ends at point:" << std::fixed << a_new +0.2<< "  " << std::fixed << b_new << "  " << std::fixed << c_new << endl;
		}
	}
	else
		cout << "Can't calc!" << endl;
}
