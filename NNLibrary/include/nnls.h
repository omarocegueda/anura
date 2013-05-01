/*Non-negative Least Squares solver
	Author: Omar Ocegueda
*/

#ifndef NNLS_H
#define NNLS_H

void computeHouseholderTransformation(double *x, int n, double **H);
void testHouseholderTransformation(void);

/**
	solves: min ||Ex-f|| s.t. x>=0
*/
int nnls(double *E, long n, long m, double *f, double *x, double *error=0);

/**
	solves: min ||Ex-f|| s.t. x>=0, considering only the columns of E given by I
*/
int nnls_subspace(double *E, int n, int m, double *f, double *x, int *I, int mm, double *error=0);


int nnls_geman_mcclure(double *E, long n, long m, double *f, double *x, double *error);
int nnls_tukey_biweight(double *E, long n, long m, double *f, double sigma, double *x, double *error);
#endif
