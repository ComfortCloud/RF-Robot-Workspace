#include "dataPreprocess.h"

void MyReaderData::unwrappingPhase(int EPCNum, int readerID, int antID)
{
	vector<double> unwrappingphase;
	vector<double> phaseVec = EPCVec[EPCNum].reader[readerID].ant[antID - 1].phase;
	for (auto i = 0; i < phaseVec.size(); i++)
	{
		phaseVec[i] = 2 * M_PI - phaseVec[i] * M_PI / 180;
	}
	double phase_lb = 0.5;
	double phase_ub = 1.5;
	double diff;
	unwrappingphase.push_back(phaseVec[0]);
	for (auto i = 1; i < phaseVec.size(); i++)
	{
		diff = phaseVec[i] - phaseVec[i - 1];
		if (diff > phase_ub * M_PI)
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff - 2 * M_PI);
		}
		else if (diff < (phase_ub * M_PI) && diff >(phase_lb * M_PI))
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff - M_PI);
		}
		else if (diff < (-phase_lb * M_PI) && diff >(-phase_ub * M_PI))
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff + M_PI);
		}
		else if (diff < (-phase_ub * M_PI))
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff + 2 * M_PI);
		}
		else
		{
			unwrappingphase.push_back(unwrappingphase[i - 1] + diff);
		}
	}
	unwrappingphase.swap(EPCVec[EPCNum].reader[readerID].ant[antID - 1].phase);
	vector<double>().swap(unwrappingphase);
}
