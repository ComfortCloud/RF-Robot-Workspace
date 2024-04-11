#ifndef LMSOLUTION_H
#define LMSOLUTION_H

#include "dataPreprocess.h"
#include "paramDefine.h"

// LM算法
class LM
{
private:
	int size0 = 20; // 最小读取次数
	double e_min = 0.000001;
	int k_max = 100;
	double v = 1;
	double z = 0;
	double y_delta = 1; // 货架距离/2，用以修正初始点

	// 常数
	MatrixXd I = MatrixXd::Identity(5, 5);
	double c = 3e8;
	double f = 920.625e6;
	double lameda = c / f;

	double a_it, b_it, c_it, phase_offset1_it, phase_offset2_it;

	double phase_offset1_new, phase_offset2_new;

public:
	vector<string> epcId;
	vector<EPC_xyz> epc_xyz;
	double a_new, b_new, c_new;
	int flag;
	void MakeHessian(MyReaderData* dataset, int targetID);
};


double phaseFunction(double x, double y, double z, double a, double b, double c, double phase_offset);

#endif