#include "../include/preprocessing_cp.h"
#include <time.h>
using namespace Eigen;
using namespace std;

int main()
{
	clock_t start, finish;
    //clock_t为CPU时钟计时单元数
    start = clock();

 	MyReaderData data1;
	data1.DownloadData("../experiment/16/first", 1);
	data1.DownloadData("../experiment/16/second", 2);
	data1.DownloadData("../experiment/16/third", 3);
	cout << "Start reading data!" << endl;
	OdomData odom;
	fstream fin;
	// fin.open("/home/tzq/Pointcloud-recognize-with-RFID-localization/experiment/17/1173511667.txt", ios::in);
	fin.open("../experiment/16/odom.txt", ios::in);
	cout << "Start reading data!" << endl;
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
		// fin>>a;
		// fin>>b;
		fin >> yaw0;
		odom.yaw.push_back(yaw0);
		fin >> timestamp_ros0;
		odom.timestamp.push_back(timestamp_ros0);
		// cout << x0 << y0 << yaw0 << timestamp_ros0 << endl;
	}
	fin.close();

	cout << "Finish reading data!" << endl;

	ofstream EPCxyz("EPCxyz_11_new.txt");
	int count = 0;
	for (int i = 0; i < data1.EPCVec.size(); i++)
	{
		if (1)
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0146-1040-79B1"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0160-1040-8274"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0172-1040-93F2"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0170-1040-93F1"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0197-1040-B05A"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0199-1040-B05B"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-2000-1080-0037"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0185-1040-9F6C"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-2171-1040-96CD"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-2000-1080-0041"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0186-1040-A549"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0182-1040-9CAF"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0146-1040-79B1"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0148-1040-79B2"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0145-1040-73C4"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0119-1040-8542"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0183-1040-9F6B"
		// || 	 data1.EPCVec[i].epc=="E200-001A-0411-0119-1040-8543")
		{
			LM EPC_localization;
			int index;
			if (data1.EPCVec[i].reader[0].ant[0].phase.size() + data1.EPCVec[i].reader[0].ant[1].phase.size() + data1.EPCVec[i].reader[0].ant[2].phase.size() < data1.EPCVec[i].reader[1].ant[0].phase.size() + data1.EPCVec[i].reader[1].ant[1].phase.size() + data1.EPCVec[i].reader[1].ant[2].phase.size())
				index = 1;
			else
				index = 0;
			cout << "------------------------------------------------------" << endl;
			cout << data1.EPCVec[i].epc << endl;
			EPC_localization.flag = 1;
			EPC_localization.MakeHessian(data1.EPCVec[i].reader[index], odom);
			if (EPC_localization.flag)
			{
				cout << "Succeed result!" << endl;
				EPCxyz << ++count << " " << data1.EPCVec[i].epc << " " << std::fixed << EPC_localization.a_new+0.2 << " " << std::fixed << EPC_localization.b_new << " " << std::fixed << EPC_localization.c_new << endl;
			}
			else
				cout << "no result" << endl;
		}
	}
    finish = clock();
    //clock()函数返回此时CPU时钟计时单元数
    cout <<endl<<"the time cost is:" << double(finish - start) / CLOCKS_PER_SEC<<endl;

	return 0;
}
