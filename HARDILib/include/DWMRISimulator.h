#ifndef DWMRISIMULATOR_H
#define DWMRISIMULATOR_H
#include "MultiTensorField.h"
class DWMRISimulator{
	int *compartmentCount;
	double *orientations;
	double *distToBoundary;
	int nslices;
	int nrows;
	int ncols;
	int maxCompartments;
	public:
		DWMRISimulator(int _nslices, int _nrows, int _ncols, int _maxCompartments);
		void addPath(double *path, int npoints, double radius);
		void buildMultiTensorField(MultiTensorField &M);
		~DWMRISimulator();
};

#endif
