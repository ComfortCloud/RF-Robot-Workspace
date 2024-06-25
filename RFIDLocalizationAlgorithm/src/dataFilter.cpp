#include "dataPreprocess.h"
#include "paramDefine.h"

// 两种数据过滤方法dataFilter（最小值段+断裂点查找）和dataFilter1（斜率变化），LM中只选了一种用，目前用的是dataFilter
int MyReaderData::dataFilter1(int epcNum, int antID)
{
	Antenna* ant = &EPCVec[epcNum].reader[0].ant[antID];

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
	cout << "flag=" << flag << endl;
	return flag;
}


int MyReaderData::dataFilter(int epcNum, int antID)
{
	Antenna* ant = &EPCVec[epcNum].reader[0].ant[antID];


	cout << "***begin to filter ....***" << endl;
	int flag = 0;
	vector<double> phase = ant->phase;
	vector<double> time = ant->timestamp;
	vector<double> rssi = ant->rssi;

	// 选取合适的数据段
	int minindex = 0;
	//minindex = std::distance(phase.begin(), std::min_element(phase.begin(), phase.end()));

	// if (phase.size() - minindex < 10)
	//     return 0;

	int gap = 5;

	// 向右查找
	int rightindex = 0;
	rightindex = phase.size();
	for (int j = minindex; j < phase.size() - gap; ++j) {
		if (phase[j + gap] - phase[j] < 0) {
			rightindex = j;
			break;
		}
	}

	// 向左查找
	int leftindex = 0;
	for (int j = 0; j < minindex - gap; ++j)
	{
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
		if (e[m] > 2 * e_avg) {
			temp.push_back(m);
		}
	}
	temp.push_back(n - 1);

	if (temp.size() == 2) {
		// pass
		left_temp = temp[0];
		right_temp = temp[1];
	}
	else if (temp.size() == 3)
	{
		if (temp[0] - 0 > n - 1 - (temp[1] + 1)) {
			left_temp = 0;
			right_temp = temp[1];
		}
		else {
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
		EPCVec[epcNum].reader[0].ant[antID].phase = phase;
		EPCVec[epcNum].reader[0].ant[antID].timestamp = time;
		EPCVec[epcNum].reader[0].ant[antID].rssi = rssi;
		flag = 1;
		cout << "The filter phase size is" << phase.size() << endl;
	}

	cout << "flag=" << flag << endl;
	return flag;
}
