#include "dataPreprocess.h"
#include "paramDefine.h"
#include "LMSolution.h"
#include <time.h>
using namespace Eigen;
using namespace std;

int main()
{
	clock_t start, finish;
	//clock_t为CPU时钟计时单元数
	start = clock();

	MyReaderData dataset;
	dataset.DownloadDataRFID("../experiment/16/first", 1);
	dataset.DownloadDataRFID("../experiment/16/second", 2);
	dataset.DownloadDataRFID("../experiment/16/third", 3);
	cout << "Finish reading RFID data!" << endl;

	dataset.DownloadDataOdom("./experiment/16/odom.txt");
	cout << "Finish reading Odom data!" << endl;

	// 数据预处理：过滤、插值、坐标转换
	dataset.preprocess();

	ofstream EPCxyz("EPCxyz_11_new.txt");
	int count = 0;
	for (int i = 0; i < dataset.EPCVec.size(); i++)
	{
		if (1)
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0146-1040-79B1"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0160-1040-8274"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0172-1040-93F2"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0170-1040-93F1"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0197-1040-B05A"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0199-1040-B05B"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-2000-1080-0037"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0185-1040-9F6C"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-2171-1040-96CD"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-2000-1080-0041"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0186-1040-A549"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0182-1040-9CAF"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0146-1040-79B1"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0148-1040-79B2"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0145-1040-73C4"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0119-1040-8542"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0183-1040-9F6B"
			// || 	 dataset.EPCVec[i].epc=="E200-001A-0411-0119-1040-8543")
		{
			LM EPC_localization;
			int index;
			if (dataset.EPCVec[i].reader[0].ant[0].phase.size() + dataset.EPCVec[i].reader[0].ant[1].phase.size()
				+ dataset.EPCVec[i].reader[0].ant[2].phase.size() < dataset.EPCVec[i].reader[1].ant[0].phase.size()
				+ dataset.EPCVec[i].reader[1].ant[1].phase.size() + dataset.EPCVec[i].reader[1].ant[2].phase.size())
				index = 1;
			else
				index = 0;
			cout << "------------------------------------------------------" << endl;
			cout << dataset.EPCVec[i].epc << endl;
			EPC_localization.flag = 1;
			EPC_localization.MakeHessian(&dataset, i);
			if (EPC_localization.flag)
			{
				cout << "Succeed result!" << endl;
				EPCxyz << ++count << " " << dataset.EPCVec[i].epc << " " << std::fixed << EPC_localization.a_new + 0.2 << " " << std::fixed << EPC_localization.b_new << " " << std::fixed << EPC_localization.c_new << endl;
			}
			else
				cout << "no result" << endl;
		}
	}
	finish = clock();
	//clock()函数返回此时CPU时钟计时单元数
	cout << endl << "the time cost is:" << double(finish - start) / CLOCKS_PER_SEC << endl;

	return 0;
}
