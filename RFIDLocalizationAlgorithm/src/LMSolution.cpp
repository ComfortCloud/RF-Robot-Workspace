#include "LMSolution.h"

double phaseFunction(double x, double y, double z, double a, double b, double c, double phase_offset)
{
	double c0 = 3e8;
	double f = 920.625e6;
	double lameda = c0 / f;
	double result = (4 * M_PI * sqrt(pow(x - a, 2) + pow(y - b, 2) + pow(z - c, 2)) / lameda) - phase_offset;
	return result;
}

void LM::MakeHessian(MyReaderData* dataset, int targetID)
{
	// 从dataset中提取数据
	std::vector<int> filter_flag;
	std::vector<int> id = dataset->EPCVec[targetID].reader[0].antChosen;
	vector<double> x1 = dataset->EPCVec[targetID].reader[0].ant[id[0]].antX;
	vector<double> y1 = dataset->EPCVec[targetID].reader[0].ant[id[0]].antY;
	double z1 = dataset->EPCVec[targetID].reader[0].ant[id[0]].antZ;

	vector<double> x2 = dataset->EPCVec[targetID].reader[0].ant[id[1]].antX;
	vector<double> y2 = dataset->EPCVec[targetID].reader[0].ant[id[1]].antY;
	double z2 = dataset->EPCVec[targetID].reader[0].ant[id[1]].antZ;

	Antenna ant1 = dataset->EPCVec[targetID].reader[0].ant[id[0]];
	Antenna ant2 = dataset->EPCVec[targetID].reader[0].ant[id[1]];

	if (flag)
	{
		int flag1 = 1;


		// linear fitting
		/*cout << "Linear fitting" << endl;
		tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdataset_get = linear1(ant1, odom);
		vector<double> LMphase1 = get<0>(LMdataset_get);
		vector<double> LMx1 = get<1>(LMdataset_get);
		vector<double> LMy1 = get<2>(LMdataset_get);
		vector<double> LMw1 = get<3>(LMdataset_get);
		vector<double> LMrssi1 = get<4>(LMdataset_get);
		tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> LMdata2_get = linear1(ant2, odom);
		vector<double> LMphase2 = get<0>(LMdata2_get);
		vector<double> LMx2 = get<1>(LMdata2_get);
		vector<double> LMy2 = get<2>(LMdata2_get);
		vector<double> LMw2 = get<3>(LMdata2_get);
		vector<double> LMrssi2 = get<4>(LMdata2_get);*/


		// cout << LMphase1.size() << endl;
		if (ant1.phase.size() == 0 || ant2.phase.size() == 0)
		{
			flag1 = 0;
			flag = 0;
		}

		if (flag1 == 1)
		{
			double phase_diff1 = ant1.phase[0];
			double phase_diff2 = ant2.phase[0];

			// vector->eigen
			Eigen::VectorXd phase1 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ant1.phase.data(), ant1.phase.size());
			Eigen::VectorXd phase2 = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(ant2.phase.data(), ant2.phase.size());


			// 设置初始迭代点
			int minPosition1 = min_element(ant1.phase.begin(), ant1.phase.end()) - ant1.phase.begin();
			int minPosition2 = min_element(ant2.phase.begin(), ant2.phase.end()) - ant2.phase.begin();
			double a0, b0, c0;
			cout << "minPosition1" << minPosition1 << endl;
			cout << "minPosition2" << minPosition2 << endl;
			cout << "x1min" << x1[minPosition1] << endl;
			cout << "x2min" << x2[minPosition2] << endl;


			a0 = (x1[minPosition1] + x2[minPosition2]) / 2;
			b0 = (y1[minPosition1] + y2[minPosition2]) / 2 - 0 * y_delta;

			double rssi1_avg = 0, rssi2_avg = 0;
			for (int i = 0; i < ant1.rssi.size(); i++)
			{
				rssi1_avg += ant1.rssi[i];
				rssi2_avg += ant2.rssi[i];
			}
			rssi1_avg = rssi1_avg / ant1.rssi.size();
			rssi2_avg = rssi2_avg / ant2.rssi.size();
			if (rssi1_avg > rssi2_avg)
				c0 = z1;
			else
				c0 = z2;

			// 初始化代价函数
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
						J(i, 3) = -1;
						J(i, 4) = 0;
					}
					for (int i = 0; i < n2; i++)
					{
						J(i + n1, 0) = -4 * M_PI * (x2[i] - a_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 1) = -4 * M_PI * (y2[i] - b_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i + n1, 2) = -4 * M_PI * (z2 - c_it) / (lameda * sqrt(pow(x2[i] - a_it, 2) + pow(y2[i] - b_it, 2) + pow(z2 - c_it, 2)));
						J(i, 3) = 0;
						J(i, 4) = -1;
					}
				}

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
			cout << "Iteration starts at point:" << std::fixed << a0 + 0.2 << "  " << std::fixed << b0 << "  " << std::fixed << c0 << endl;
			cout << "Iteration ends at point:" << std::fixed << a_new + 0.2 << "  " << std::fixed << b_new << "  " << std::fixed << c_new << endl;
		}
	}
	else
		cout << "Can't calc!" << endl;
}
