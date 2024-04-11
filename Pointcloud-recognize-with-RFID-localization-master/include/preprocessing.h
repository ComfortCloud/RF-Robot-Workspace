#pragma once
#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <cmath>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <numeric>

#define max(a,b) ((a)>(b))?(a):(b)
#define M_PI  3.1415926
using namespace Eigen;
using namespace std;

struct Antenna
{
	int antennaId;
	std::vector<double> phase;
	std::vector<double> rssi;
	std::vector<double> timestamp;
};


struct Reader
{
	int readerID;
	Antenna ant[3];
};


struct EPC
{
	string   epc;
	Reader reader[2];
};
tuple<vector<int>, vector<int>> findPeaks(vector<double> phase);
class MyReaderData {
private:
	int m = 0;
	
public:
	vector<EPC> EPCVec;
	//void DownloadData(string s);
	void DownloadData(string s,int id);
};

typedef struct
{
	vector<double> x;
	vector<double> y;
	vector<double> yaw;
	vector<double> timestamp;
} OdomData;

// 相位计算函数
double phaseFunction(double x, double y, double z, double a, double b, double c, double phase_offset);

// 相位解缠函数
vector<double> unwrappingPhase(vector<double> phaseVec);

typedef struct
{
	vector<double> x_new;
	vector<double> y_new;
	double z_new;
} NewCoordinate;
NewCoordinate AntennaCoordinate(VectorXd x, VectorXd y, double z, VectorXd w, int readerID, int antennaId);
double fitting(double x1, double x2, double y1, double y2, double x);
tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> linear(Antenna ant1, Antenna ant2, OdomData odom);
//tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> linear(Antenna ant, OdomData odom);

// LM算法
typedef struct
{
	double x;
	double y;
	double z;
} EPC_xyz;

class LM
{
private:
	int size0 = 20; // 最小读取次数
	double e_min = 0.00001;
	int k_max = 50;
	double v =1;
	double z = 0;
	double y_delta = 1.5; // 货架距离/2，用以修正初始点

	// 常数
	MatrixXd I = MatrixXd::Identity(5, 5);
	double c = 3e8;
	double f = 920.625e6;
	double lameda = c / f;
	double a_limit_min = 1.5;
	double a_limit_max = 6.5;
	double b_limit_min = 0.5;
	double b_limit_max = 1.5;
	double c_limit_min = 0.2;
	double c_limit_max = 2;

	double a_it, b_it, c_it, phase_offset1_it, phase_offset2_it;

	double phase_offset1_new, phase_offset2_new;

	double u1=1, u2=1, u3=1, u4=1, u5=1,u6=1;

public:
	vector<string> epcId;
	vector<EPC_xyz> epc_xyz;
	double a_new, b_new, c_new;
	int flag;
	void MakeHessian(Reader reader, OdomData odom);
};

#endif
