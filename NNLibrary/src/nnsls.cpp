/*Non-negative Sparse Least Squares solver
	Author: Omar Ocegueda
*/

#include "nnsls.h"
#include "nnls.h"
#include "ls.h"
#include "linearalgebra.h"
#include <string.h>
#include <set>
#include <math.h>
#define EPS_NNSLS 1e-10
#define INF_NNSLS 1e+31
using namespace std;

int nnsls(double *E, int n, int m, double *f, double *x, double lambda, double *error){
	double *EE=new double[(n+1)*m];
	double *ff=new double[n+1];
	memcpy(EE,E,sizeof(double)*n*m);
	memcpy(ff,f,sizeof(double)*n);
	ff[n]=0;
	double *ee=&EE[n*m];
	double sqrtLambda=sqrt(lambda);
	for(int i=0;i<m;++i){
		ee[i]=sqrtLambda;
	}
	nnls(EE,n+1,m,ff,x,error);
	if(error!=NULL){
		multMatrixVector(E, x, n, m, ff);
		double norm=0;
		for(int i=0;i<n;++i){
			norm+=SQR(ff[i]-f[i]);
		}
		(*error)=sqrt(norm/n);
	}
	delete[] EE;
	delete[] ff;
	return 0;
}


