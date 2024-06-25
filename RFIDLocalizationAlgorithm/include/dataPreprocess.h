#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include "paramDefine.h"

#define max(a,b) ((a)>(b))?(a):(b)
#define M_PI  3.1415926

using namespace Eigen;
using namespace std;

class MyReaderData {
private:
	int m = 0;
	int size0 = 20;

public:
	vector<EPC> EPCVec;
	OdomData odom;

	//void DownloadData(string s);
	void DownloadDataRFID(string s, int id);
	void DownloadDataOdom(string s);
	void AntennaCoordinate(int readerID, int antennaId);
	double linearFitting(double x1, double x2, double y1, double y2, double x);

	void unwrappingPhase(int EPCNum, int readerID, int antennaId);

	int dataFilter1(int epcNum, int antID);
	int dataFilter(int epcNum, int antID);

	void linear(int epcNum, int ant1, int ant2);
	void linear1(int epcNum, int ant);

	void preprocess(void);
};

#endif