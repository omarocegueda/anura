#ifndef DTISYNTHETIC_H
#define DTISYNTHETIC_H
#include "geometryutils.h"
#include <vector>
#include <set>
//select k pdds out of the n given.
void selectPDDs(double *pdds, int n, int k, double minAngle, double *selected, int &pStart);
void createRealisticSyntheticField(int nrows2, int nrows1, int ncols, double *orientations, int nOrientations, double *randomPDDs, int nRandom, double minAngle, double longDiffusion, double transDiffusion, double b, double snr, double s0, 
								double *field, double *S0, double *pdds, double *amount, int *tcount);

#endif
