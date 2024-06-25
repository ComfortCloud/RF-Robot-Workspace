#ifndef PARAMDEFINE_H
#define PARAMDEFINE_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
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

struct Antenna
{
	int antennaId;
	std::vector<double> phase;
	std::vector<double> rssi;
	std::vector<double> timestamp;
	std::vector<double> antX;
	std::vector<double> antY;
	double antZ;
};


struct Reader
{
	int readerID;
	Antenna ant[3];
	std::vector<int> antChosen;
};

struct EPC
{
	std::string   epc;
	Reader reader[2];
	int calcability;
};


typedef struct
{
	std::vector<double> x;
	std::vector<double> y;
	std::vector<double> yaw;
	std::vector<double> timestamp;
} OdomData;

typedef struct
{
	double x;
	double y;
	double z;
} EPC_xyz;


typedef struct
{
	std::vector<double> x_new;
	std::vector<double> y_new;
	double z_new;
} NewCoordinate;

#endif