//#define BFGS_VERBOSE
#include "bfgs.h"
#include <iostream>
#include "macros.h"
#include "linearAlgebra.h"
#include "geometryutils.h"
#include <string.h>
using namespace std;
#define linearComb(dest, a, x, b, y, n) for(int i=0;i<(n);++i){dest[i]=(a)*x[i]+(b)*y[i];}

double ternarySearch(ScalarFunction objective, double *x0, double *p, int n, double a, double b, double tol, void *data){
	double *x=new double[n];
	linearComb(x, 1, x0, a, p, n);
	double fa=objective(x, n, data);
	linearComb(x, 1, x0, b, p, n);
	double fb=objective(x, n, data);
	double a0;
	double b0;
	double fa0;
	double fb0;
	while(tol<(b-a)){
		a0=(b+2.0*a)/3.0;
		b0=(a+2.0*b)/3.0;
		linearComb(x, 1, x0, a, p, n);
		fa0=objective(x, n, data);
		linearComb(x, 1, x0, b, p, n);
		fb0=objective(x, n, data);
		if(fa0>fb0){
			a=a0;
			fa=fa0;
		}else{
			b=b0;
			fb=fb0;
		}
	}
	delete[] x;
	return 0.5*(a+b);
}

double BFGS_Sherman_Morrison(ScalarFunction objective, VectorFunction gradient, double *x0, double *B0, int n, double tol, int maxIter, void *data){
	double *p=new double[n];
	double *g=new double[n];
	double *B=new double[n*n];
	double *x=new double[n];
	double *y=new double[n];
	double *s=new double[n];
	double *yTB=new double[n];
	memcpy(x, x0, sizeof(double)*n);
	memcpy(B, B0, sizeof(double)*n*n);
	double error=tol+1;
	for(int k=0;(k<maxIter) && (tol<error);++k){
#ifdef BFGS_VERBOSE
		for(int i=0;i<n;++i){cerr<<x[i]<<"\t";}cerr<<objective(x,n, data)<<endl;
#endif
		gradient(x, n, g, n, data);
		multVecScalar<double>(g, n, -1);//g=-grad(x_{k})
		multvec(B, g, p, n);//in Sherman-Morrison variant, we keep track of the inverse of B, instead of B itself
		//------compute step bouns------
		double alpha=ternarySearch(objective, x, p, n, 0, 1, 1e-6, data);
		for(int i=0;i<n;++i){s[i]=alpha*p[i]; x[i]+=s[i];}//S_{k}=\alpha*p_{k}; x_{k+1}=x_{k}+\alpha p_{k}
		gradient(x, n, y, n, data);//y=grad(x_{k+1})
		for(int i=0;i<n;++i){y[i]+=g[i];}//y_{k}=grad(x_{k+1}) - grad(x_{k})
		//-----update B-----
		double den=dotProduct(y, s, n);
		multvec<double>(B, y, g, n);//recycle g vector: b=B*s
		//----now yTB=y^T*B (B doesnt have to be symmetric)----
		for(int i=0;i<n;++i){
			yTB[i]=0;
			for(int j=0;j<n;++j){
				yTB[i]+=y[j]*B[j*n+i];
			}
		}
		//---------------------
		double num=dotProduct(y, g, n)+den;
		double factor=num/(den*den);
		//update B (inverse) using Sherman-Morrison formula
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j){
				B[i*n+j]+=factor*s[i]*s[j] - (g[i]*s[j])/den - s[i]*yTB[j]/den;
			}
		}
		error=dotProduct(y, y, n);
		error/=n;
	}
	double sol=objective(x,n, data);
#ifdef BFGS_VERBOSE
	for(int i=0;i<n;++i){cerr<<x[i]<<"\t";}cerr<<sol<<endl;
#endif
	memcpy(x0, x, sizeof(double)*n);
	memcpy(B0, B, sizeof(double)*n*n);
	delete[] p;
	delete[] g;
	delete[] B;
	delete[] x;
	delete[] y;
	delete[] s;
	delete[] yTB;
	return sol;
}



/*double BFGS_Nocedal(ScalarFunction objective, VectorFunction gradient, double *x0, double *B0, int n, double tol, int maxIter, void *data){
	const int refinements=5;
	double f_eval;
	double *g_eval=new double[n];
	double *w= new double[n*(2*refinements+1)+2*refinements];
	double *diag=new double[n];
	for(int i=0;i<n;++i){
		diag[i]=B0[i*n+i];
	}
	int flag=0;
	for(int i=0;i<maxIter;++i){
		f_eval=objective(x0, n, data);
		gradient(x0, n, g_eval, n, data);
		lbfgs(n, refinements, x0, &f_eval, g_eval, diag, w, &flag);
		if (flag <= 0) {
			break;
		}
	}
	delete[] g_eval;
	delete[] w;
	delete[] diag;
	double retVal=objective(x0, n, data);
	return retVal;
}

*/