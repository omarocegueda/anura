#ifndef HORNALIGN_H
#define HORNALIGN_H
double HornAlign(double* P, double *reference, int n, int options, double *R);
double wHornAlign(double *w, double* P, double *Y, int n, int options, double R[4][4]);
double wHornAlign(double *w, double* P, double *Y, int n, int options, double *R);
void qRotation(double q[4], double R[4][4]);
void qRotation(double *q, double *R);

#endif
