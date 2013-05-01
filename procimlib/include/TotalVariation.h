#ifndef TOTALVARIATION_H
#define TOTALVARIATION_H
void filterTotalVariation_L2(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int));
void filterTotalVariation_L1(double *g, int nrows, int ncols, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int));
void filterHuber_L2(double *g, int nrows, int ncols, double alpha, double lambda, double tau, double sigma, double theta, double *x, void (*callback)(double*, int, int));
void filterTGV_L2(double *g, int nrows, int ncols, double lambda, double alpha0, double alpha1, double tau, double sigma, double theta, double *u, double *v=0);
#endif
