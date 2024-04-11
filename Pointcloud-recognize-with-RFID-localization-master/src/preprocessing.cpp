#include "../include/preprocessing.h"

double phaseFunction(double x, double y, double z, double a, double b, double c, double phase_offset)
{
	double c0 = 3e8;
	double f = 920.625e6;
	double lameda = c0 / f;
	double result = (4 * M_PI * sqrt(pow(x - a, 2) + pow(y - b, 2) + pow(z - c, 2)) / lameda) - phase_offset;
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
	double x0[3] = {-0.017+0.2, -0.017+0.2 , -0.017+0.2 };
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
		TRobot << cos(w[i]), -sin(w[i]), 0, x0[antennaId - 1],
			sin(w[i]), cos(w[i]), 0, y0[antennaId - 1],
			0, 0, 1, z0[antennaId - 1],
			0, 0, 0, 1;
		Ant2Robot << x[i], y[i], z, 1;
		pVec = TRobot * Ant2Robot;
		p.x_new.push_back(pVec[0]);
		p.y_new.push_back(pVec[1]);
		p.z_new = pVec[2];
	}

	return p;
}

// 寻找波峰波谷
Antenna dataFilter(Antenna ant)
{
	cout << "begin to filter ...." << endl;
	vector<double> phase = ant.phase;
	vector<double> time = ant.timestamp;

	// 相位差/时间变化率
	vector<double> rate;
	vector<double> rate_change;
	rate.push_back(0);
	for (int i = 1; i < phase.size(); i++)
	{
		rate.push_back((phase[i] - phase[i - 1]) / (time[i] - time[i - 1]));
	}
	// cout<<"1"<<endl;
	//  相位差/时间变化率 的变化
	rate_change.push_back(0);
	rate_change.push_back(0);
	for (int i = 2; i < rate.size(); i++)
	{
		// cout<<rate[i]-rate[i-1]<<endl;
		rate_change.push_back(rate[i] - rate[i - 1]);
	}
	double rate_change_avg = 0;
	for (int i = 2; i < rate_change.size(); i++)
	{
		rate_change_avg += abs(rate_change[i]);
	}
	rate_change_avg = rate_change_avg / (rate_change.size() - 2);
	cout << "rate_change_avg" << rate_change_avg << endl;

	// 区分数据区间
	double rate_changeThreshold = 1.5; // 1.8
	vector<int> sign;
	sign.push_back(1);
	sign.push_back(1);
	double rate_change_difference;
	for (int i = 2; i < phase.size(); i++)
	{
		rate_change_difference = abs(rate_change[i]);
		if (rate_change_difference > rate_changeThreshold)
			sign.push_back(sign[i - 1] + 1);
		else
			sign.push_back(sign[i - 1]);
	}
	// cout<<"3"<<endl;
	// 根据最大的RSSI值找到最大的数段
	vector<int> max_num;
	double rssiMax = *std::max_element(std::begin(ant.rssi), std::end(ant.rssi));
	cout << "rssiMax"
		 << " " << rssiMax << endl;
	for (int i = 0; i < ant.rssi.size(); i++)
	{
		if (ant.rssi[i] == rssiMax)
			max_num.push_back(i);
	}
	// cout<<"max_num"<<max_num.size()<<endl;
	int max_num_mean = 0;
	for (int i = 0; i < max_num.size(); i++)
	{
		max_num_mean += max_num[i];
	}
	max_num_mean = max_num_mean / max_num.size();
	int max_list = sign[floor(max_num_mean)];
	// cout<<"max_list"<<max_list<<endl;
	vector<int> maxInd;
	int available_num[2];
	for (int i = 0; i < sign.size(); i++)
	{
		// cout<<sign[i]<<endl;
		if (sign[i] == max_list)
			maxInd.push_back(i);
	}
	available_num[0] = *std::min_element(std::begin(maxInd), std::end(maxInd));
	available_num[1] = *std::max_element(std::begin(maxInd), std::end(maxInd));
	cout << available_num[0] << " " << available_num[1] << endl;

	// 进一步提炼出大于某个RSSI值的数据
	double RSSI_threshold = rssiMax * 1.2;
	int available_num_pos[2];
	for (int i = available_num[0]; i < available_num[1]; i++)
	{
		if (ant.rssi[i] > RSSI_threshold)
		{
			available_num_pos[0] = i;
			break;
		}
	}
	for (int i = available_num[1]; i > available_num[0]; i--)
	{
		if (ant.rssi[i] > RSSI_threshold)
		{
			available_num_pos[1] = i;
			break;
		}
	}

	std::vector<double> phaseMid;
	std::vector<double> rssiMid;
	std::vector<double> timestampMid;
	for (int i = available_num_pos[0]; i < available_num_pos[1]; i++)
	{
		phaseMid.push_back(ant.phase[i]);
		rssiMid.push_back(ant.rssi[i]);
		timestampMid.push_back(ant.timestamp[i]);
	}
	ant.phase = phaseMid;
	ant.rssi = rssiMid;
	ant.timestamp = timestampMid;

	cout << "The phase size is" << ant.phase.size() << endl;

	cout << "Data is filtered successfully!" << endl;
	return ant;
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
		// rID.push_back(readerID);
		// fin >> antennaID;
		// aID.push_back(antennaID);
		fin >> p;
		fin >> phase0;
		phase.push_back(phase0);
		/*fin >> timestamp_ros0;
		timestamp_ros.push_back(timestamp_ros0);*/
		fin >> rssi0;
		rssi.push_back(rssi0);
		fin >> t;
		fin >> timestamp_ros0;
		timestamp_ros0 = timestamp_ros0 / 1000000;
		timestamp_ros.push_back(timestamp_ros0);
	}
	fin.close();

	// 分类数据
	for (int i = 0; i < timestamp_ros.size(); i++)
	{
		rID.push_back(1);
		aID.push_back(id);
	}

	for (int i = 0; i < epc.size(); i++)
	{
		int flag = 0;
		// int k = sort_time[i].index;
		int k = i;
		int n = EPCVec.size();
		for (int j = 0; j < n; j++)
		{
			EPCVec[j].reader[rID[k] - 1].readerID = 1;
			EPCVec[j].reader[rID[k] - 1].ant[aID[k] - 1].antennaId = aID[k];
			if (EPCVec[j].epc == epc[k])
			{
				EPCVec[j].reader[rID[k] - 1].ant[aID[k] - 1].phase.push_back(phase[k]);
				EPCVec[j].reader[rID[k] - 1].ant[aID[k] - 1].timestamp.push_back(timestamp_ros[k]);
				EPCVec[j].reader[rID[k] - 1].ant[aID[k] - 1].rssi.push_back(rssi[k]);
				flag = 1;
			}
		}
		if (flag == 0)
		{
			EPCVec.resize(++m);
			EPCVec[n].reader[rID[k] - 1].readerID = 1;
			EPCVec[n].reader[rID[k] - 1].ant[aID[k] - 1].antennaId = aID[k];
			EPCVec[n].epc = epc[k];
			EPCVec[n].reader[rID[k] - 1].ant[aID[k] - 1].phase.push_back(phase[k]);
			EPCVec[n].reader[rID[i] - 1].ant[aID[k] - 1].timestamp.push_back(timestamp_ros[k]);
			EPCVec[n].reader[rID[i] - 1].ant[aID[k] - 1].rssi.push_back(rssi[k]);
		}
	}

	// data initial choose step
	int size0 = 40;
	for (int i = 0; i < EPCVec.size(); i++)
	{
		vector<double> phaseVec = EPCVec[i].reader[rID[i] - 1].ant[aID[i] - 1].phase;
		int minPosition = min_element(phaseVec.begin(), phaseVec.end()) - phaseVec.begin();
		if (minPosition < size0 / 2 || (phaseVec.size() - minPosition) < size0 / 2 || phaseVec.size() < size0)
		{
			EPCVec.erase(EPCVec.begin() + i);
			i--;
		}
		else
		{
			EPCVec[i].reader[rID[i] - 1].ant[aID[i] - 1].phase = unwrappingPhase(EPCVec[i].reader[0].ant[aID[i] - 1].phase);
			cout << EPCVec[i].epc << " phase  size is " << EPCVec[i].reader[rID[i] - 1].ant[aID[i] - 1].phase.size() << endl;
		}
	}
}
// 线性插值
double fitting(double x1, double x2, double y1, double y2, double x)
{
	double y;
	y = (y2 - y1) / (x2 - x1) * (x - x1) + y1;

	return y;
}

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

	for (int k = 0; k < odom.timestamp.size(); k = k + 20)
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

// tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> linear(Antenna ant, OdomData odom)
// {
// 	vector<double> LMphase_get;
// 	vector<double> LMx_get;
// 	vector<double> LMy_get;
// 	vector<double> LMw_get;
// 	vector<double> LMrssi_get;
// 	double x_get, y_get, w_get;

// 	int n = odom.timestamp.size();
// 	for (int m = 0; m < ant.timestamp.size(); m = m + 10)
// 	{
// 		if (odom.timestamp.front() < ant.timestamp[m] && ant.timestamp[m] < odom.timestamp.back())
// 		{
// 			int j = 0;
// 			for (j = 0; j < n; j++)
// 			{
// 				if (ant.timestamp[m] <= odom.timestamp[j])
// 				{
// 					j--;
// 					break;
// 				}
// 			}
// 			x_get = odom.x[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.x[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
// 			y_get = odom.y[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.y[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
// 			w_get = odom.yaw[j] * ((ant.timestamp[m] - odom.timestamp[j + 1]) / (odom.timestamp[j] - odom.timestamp[j + 1])) + odom.yaw[j + 1] * ((ant.timestamp[m] - odom.timestamp[j]) / (odom.timestamp[j + 1] - odom.timestamp[j]));
// 			LMx_get.push_back(x_get);
// 			LMy_get.push_back(y_get);
// 			LMw_get.push_back(w_get);
// 			LMphase_get.push_back(ant.phase[m]);
// 			LMrssi_get.push_back(ant.rssi[m]);
// 		}
// 	}

// 	return make_tuple(LMphase_get, LMx_get, LMy_get, LMw_get, LMrssi_get);
// }

void LM::MakeHessian(Reader reader, OdomData odom)
{
	vector<int> id;
	for (int i = 0; i < 3; i++)
	{
		if (reader.ant[i].phase.empty())
		{
			std::cout << "ant " << i + 1 << " has no data." << endl;
		}
		else
		{
			id.push_back(i);
		}
	}
	cout << "id:" << id.size() << endl;
	switch (id.size())
	{
	case 1:
		flag = 0;
		std::cout << "Can't calc." << endl;
	case 2:
		for (int i = 0; i < id.size(); i++)
		{
			reader.ant[id[i]] = dataFilter(reader.ant[id[i]]);
		}
		if (reader.ant[id[0]].phase.size() < size0 && reader.ant[id[1]].phase.size() < size0)
		{
			flag = 0;
			std::cout << "Can't calc." << endl;
		}

	case 3:
		for (int i = 0; i < id.size(); i++)
		{
			reader.ant[id[i]] = dataFilter(reader.ant[id[i]]);
		}
		if (reader.ant[id[0]].phase.size() < size0 &&
			reader.ant[id[1]].phase.size() < size0 &&
			reader.ant[id[2]].phase.size() < size0)
		{
			flag = 0;
			std::cout << "Can't calc." << endl;
		}
		else
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
		}

	default:
		break;
	}
	if (flag)
	{
		int flag1 = 1;
		Antenna ant1 = reader.ant[id[0]];
		Antenna ant2 = reader.ant[id[1]];

		// linear fitting
		tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata1_get = linear(ant1, ant2, odom);
		vector<double> LMphase1 = get<0>(LMdata1_get);
		vector<double> LMphase2 = get<1>(LMdata1_get);
		vector<double> LMx = get<2>(LMdata1_get);
		vector<double> LMy = get<3>(LMdata1_get);
		vector<double> LMw = get<4>(LMdata1_get);
		vector<double> LMrssi1 = get<5>(LMdata1_get);
		vector<double> LMrssi2 = get<6>(LMdata1_get);

		// linear fitting
		// cout<<"Linear fitting"<<endl;
		// tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata1_get = linear(ant1, odom);
		// vector<double> LMphase1 = get<0>(LMdata1_get);
		// vector<double> LMx1 = get<1>(LMdata1_get);
		// vector<double> LMy1 = get<2>(LMdata1_get);
		// vector<double> LMw1 = get<3>(LMdata1_get);
		// vector<double> LMrssi1 = get<4>(LMdata1_get);
		// tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata2_get = linear(ant2, odom);
		// vector<double> LMphase2 = get<0>(LMdata2_get);
		// vector<double> LMx2 = get<1>(LMdata2_get);
		// vector<double> LMy2 = get<2>(LMdata2_get);
		// vector<double> LMw2 = get<3>(LMdata2_get);
		// vector<double> LMrssi2 = get<4>(LMdata2_get);

		std::cout << "--1--" << std::endl;
		cout << LMphase1.size() << endl;
		if (LMphase1.size() == 0 || LMphase2.size() == 0)
		{
			flag1 = 0;
			flag = 0;
		}

		if (flag1 == 1)
		{
			double phase_diff1 = LMphase1[0];
			double phase_diff2 = LMphase2[0];

			for (int i = 0; i < LMphase1.size(); i++)
			{
				LMphase1[i] = LMphase1[i] - phase_diff1;
			}

			for (int i = 0; i < LMphase2.size(); i++)
			{
				LMphase2[i] = LMphase2[i] - phase_diff2;
			}

			// vector->eigen
			Eigen::VectorXd phase1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(LMphase1.data(), LMphase1.size());
			Eigen::VectorXd phase2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(LMphase2.data(), LMphase2.size());
			std::cout << "--2--" << std::endl;

			NewCoordinate position1 = AntennaCoordinate(LMx, LMy, z, LMw, reader.readerID, ant1.antennaId);
			vector<double> x1 = position1.x_new;
			vector<double> y1 = position1.y_new;
			double z1 = position1.z_new;

			NewCoordinate position2 = AntennaCoordinate(LMx, LMy, z, LMw, reader.readerID, ant2.antennaId);
			vector<double> x2 = position2.x_new;
			vector<double> y2 = position2.y_new;
			double z2 = position2.z_new;

			// std::cout << "--3--" << std::endl;
			//  cout<<"x坐标为:"<<endl;
			//  for(int i = 0; i <x1.size(); i++)
			//  {
			//  	cout<<x1[i]<<endl;
			//  }

			// for(int i = 0; i <x2.size(); i++)
			// {
			// 	cout<<x2[i]<<endl;
			// }
			//
			int minPosition1 = min_element(LMphase1.begin(), LMphase1.end()) - LMphase1.begin();
			int minPosition2 = min_element(LMphase2.begin(), LMphase2.end()) - LMphase2.begin();
			double a0, b0, c0;
			// cout << y1[minPosition1]<<endl;
			// cout<<LMy[minPosition1]<<endl;
			a0 = (x1[minPosition1] + x2[minPosition2]) / 2;
			if (reader.readerID == 2)
				b0 = (y1[minPosition1] + y2[minPosition2]) / 2 + y_delta;
			if (reader.readerID == 1)
				b0 = (y1[minPosition1] + y2[minPosition2]) / 2 - y_delta;

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
				// if (LMrssi1[minPosition1] > LMrssi2[minPosition2])
				c0 = z1;
			else
				c0 = z2;
			// c0=(z1+z2)/2;
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
			int u = 2;
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
						J(i, 3) = -1;
						J(i, 4) = 0;
					}
					for (int i = 0; i < n2; i++)
					{
						J(i + n1, 0) = -4 * M_PI * (x2[i] - a_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 1) = -4 * M_PI * (y2[i] - b_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 2) = -4 * M_PI * (z2 - c_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 3) = 0;
						J(i + n1, 4) = -1;
					}
				}

				// std::cout << "--9--" << std::endl;

				if (k == 1)
				{
					VectorXd phase_res(n1 + n2);
					for (int i = 0; i < n1; i++)
					{
						phase_res(i) = phaseFunction(x1[i], y1[i], z1, a0, b0, c0, phase_offset1_0);
					}
					for (int i = 0; i < n2; i++)
					{
						phase_res(i + n1) = phaseFunction(x2[i], y2[i], z2, a0, b0, c0, phase_offset2_0);
					}

					phase_diff1 = phase_res(0);
					phase_diff2 = phase_res(n1);
					for (int i = 0; i < n1; i++)
					{
						phase_res(i) = phase_res(i) - phase_diff1;
					}
					for (int i = 0; i < n2; i++)
					{
						phase_res(i + n1) = phase_res(i + n1) - phase_diff2;
					}

					d = phase - phase_res;
					H = J.transpose() * J;
					// double tao = 1e-3;
					// double max_H = 0;
					// for (int i = 0; i < H.rows(); i++)
					// {
					// 	if (H(i, i) > max_H)
					// 		max_H = J(i, i);
					// }
					// v= tao * max_H;   //找到海森矩阵对角线上最大的值，并乘tao

					step = (H + v * I).inverse() * J.transpose() * d;

					e = d.dot(d);
					continue;
				}

				// std::cout << "--10--" << std::endl;

				// std::cout << step << std::endl;

				H = J.transpose() * J;

				a_new = a_it + step(0);
				b_new = b_it + step(1);
				c_new = c_it + step(2);
				phase_offset1_new = phase_offset1_it + step(3);
				phase_offset2_new = phase_offset2_it + step(4);

				// std::cout << "--11--" << std::endl;

				for (int i = 0; i < n1; i++)
				{
					phase_new(i) = phaseFunction(x1[i], y1[i], z1, a_new, b_new, c_new, phase_offset1_new);
				}
				for (int i = 0; i < n2; i++)
				{
					phase_new(i + n1) = phaseFunction(x2[i], y2[i], z2, a_new, b_new, c_new, phase_offset2_new);
				}

				phase_diff1 = phase_new(0);
				phase_diff2 = phase_new(n1);
				for (int i = 0; i < n1; i++)
				{
					phase_new(i) = phase_new(i) - phase_diff1;
				}
				for (int i = 0; i < n2; i++)
				{
					phase_new(i + n1) = phase_new(i + n1) - phase_diff2;
				}
				// std::cout << "--12--" << std::endl;

				d_new = phase - phase_new;
				e_new = d_new.dot(d_new);
				step = (H + v * I).inverse() * J.transpose() * d_new;

				// double deltaL=step.transpose() * (v * step  - J.transpose()*d);
				// double roi = (e-e_new) / deltaL;

				// if (roi <0)
				// {
				// 	e=e_new;
				// 	a_it = a_new;
				// 	b_it = b_new;
				// 	c_it = c_new;
				// 	phase_offset1_it = phase_offset1_new;
				// 	phase_offset2_it = phase_offset2_new;
				// 	v *= max(1.0 / 3.0, 1 - pow(2 * roi - 1, 3));
				// 	u = 2;
				// }
				// else
				// {
				// 	v = u * v;
				// 	u = u*2;
				// }
				// cout<<e_new<<endl;

				if (e_new <= e)
				{
					if (e - e_new < e_min)
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
					std::cout << e_new << std::endl;
				}
				else
				{
					updateJ=0;
					v = v * 10;
					// std::cout << v << std::endl;
				}
			}
			cout << "Iteration starts at point:" << std::fixed << a0 << "  " << std::fixed << b0 << "  " << std::fixed << c0 << endl;
			cout << "Iteration ends at point:" << std::fixed << a_new << "  " << std::fixed << b_new << "  " << std::fixed << c_new << endl;
		}
	}
}
