#include "preprocessing.h"

using namespace Eigen;
using namespace std;

int main()
{
	MyReaderData data1;
	data1.DownloadData("../experiment/12/first", 1);
	data1.DownloadData("../experiment/12/second", 2);
	data1.DownloadData("../experiment/12/third", 3);
	OdomData odom;
	fstream fin;
	// fin.open("/home/tzq/Pointcloud-recognize-with-RFID-localization/experiment/17/1673515667.txt", ios::in);
	fin.open("../experiment/12/odom.txt", ios::in);
	while (1)
	{
		double x0;
		double y0;
		double a;
		double b;
		double yaw0;
		double timestamp_ros0;
		fin >> x0;
		if (fin.eof())
			break;
		odom.x.push_back(x0);
		fin >> y0;
		odom.y.push_back(y0);
		fin >> a;
		fin >> b;
		fin >> yaw0;
		odom.yaw.push_back(yaw0);
		fin >> timestamp_ros0;
		odom.timestamp.push_back(timestamp_ros0);
		// cout << x0 << y0 << yaw0 << timestamp_ros0 << endl;
	}
	fin.close();

	cout << "Finish reading data!" << endl;

	ofstream EPCxyz("EPCxyz.txt");
	int count = 0;
	for (int i = 0; i < data1.EPCVec.size(); i++)
	{
		if (data1.EPCVec[i].epc != "E200-001A-0411-2000-1080-0001" && data1.EPCVec[i].epc != "E200-001A-0411-0236-1040-D50E"
&& data1.EPCVec[i].epc != "E200-001A-0411-0226-1040-CDC1"

)
		{
			LM EPC_localization;
			int index;
			if (data1.EPCVec[i].reader[0].ant[0].phase.size() + data1.EPCVec[i].reader[0].ant[1].phase.size() + data1.EPCVec[i].reader[0].ant[2].phase.size() < data1.EPCVec[i].reader[1].ant[0].phase.size() + data1.EPCVec[i].reader[1].ant[1].phase.size() + data1.EPCVec[i].reader[1].ant[2].phase.size())
				index = 1;
			else
				index = 0;
			cout << data1.EPCVec[i].epc << endl;
			EPC_localization.flag = 1;
			EPC_localization.MakeHessian(data1.EPCVec[i].reader[index], odom);
			if (EPC_localization.flag)
			{
				cout << "Succeed result!" << endl;
				EPCxyz << ++count << " " << data1.EPCVec[i].epc << " " << std::fixed << EPC_localization.a_new << " " << std::fixed << EPC_localization.b_new << " " << std::fixed << EPC_localization.c_new << endl;
			}
			else
				cout << "no result" << endl;
		}
	}

	return 0;
}
