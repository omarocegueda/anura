#include "nnsls_pgs2.h"
#include "linearalgebra.h"
#include "geometryutils.h"
int nnsls_pgs2(double *Phi, int n, int m, double *s, double *alpha, double lambda, double *error){
    int nIterCompress = 5;
	double *G=new double[m*m];
	double *PP=new double[m*m];
	double *b=new double[m];
	double *step=new double[m];
	double *alphak=new double[m];
	double *phiAlpha=new double[n];
	for(int i=0;i<m;++i){
		for(int j=i;j<m;++j){
			double &sum=G[i*m+j];
			double *xi=&Phi[i];
			double *xj=&Phi[j];
			sum=0;
			for(int k=0;k<n;++k,xi+=m, xj+=m){
				sum+=(*xi)*(*xj);
			}
			G[j*m+i]=sum;
		}
	}
	memcpy(PP,G,sizeof(double)*m*m);
	double *diag=G;
	for(int i=0;i<m;++i, diag+=(m+1)){
		*diag+=lambda;
		step[i]=1.0/(*diag);
		b[i]=0;
		double *xi=&Phi[i];
		for(int j=0;j<n;++j, xi+=m){
			b[i]+=(*xi)*s[j];
		}
	}
	double *g=G;
	double *ss=step;
	for(int i=0;i<m;++i, ++ss){
		for(int j=0;j<m;++j, ++g){
			(*g)*=(*ss);
		}
		b[i]*=*ss;
	}
    double tol=1e-10;
	double rmse=tol+1;
	int maxIter=100;
	for(int iter=0;(iter<maxIter) && (tol<rmse);++iter){
		memcpy(alphak,alpha,sizeof(double)*m);
        //------------------------
        // Proyected Gauss Seidel iteration
        //-----------------------
		double *g=G;
		for(int k=0;k<m;++k, g+=m){
			double prod=dotProduct(g, alpha,m);
			double val=alpha[k]-prod+b[k];
            alpha[k]=MAX(val,0);
		}
        //-------------------
        // compressed subspace minimization 
        //-------------------
		for(int iterMinSb=0;iterMinSb<nIterCompress;++iterMinSb){
			g=G;
			for(int k=0;k<m;++k, g+=m)if(alpha[k]>0){
				double prod=dotProduct(g, alpha,m);
				double val=alpha[k]-prod+b[k];
				alpha[k]=MAX(val,0);
			}
            //scale factor
			multMatrixVector<double>(Phi, alpha, n, m, phiAlpha);
			double num=dotProduct(phiAlpha, s, n);
			double den=evaluateQuadraticForm(PP,alpha,m);
			multVectorScalar(alpha, num/den, m, alpha);
			//multVectorScalar(alpha, den/num, m, alpha);
		}
		//-----------------
		rmse=0;
		for(int i=0;i<m;++i){
			rmse+=SQR(alphak[i]-alpha[i]);
		}
		rmse=sqrt(rmse/m);
	}
	if(error!=NULL){
		multMatrixVector<double>(Phi, alpha, n, m, phiAlpha);
		*error=0;
		for(int i=0;i<n;++i){
			double d=phiAlpha[i]-s[i];
			*error+=d*d;
		}
		*error=sqrt(*error/n);
	}
	delete[] G;
	delete[] b;
	delete[] step;
	delete[] alphak;
	delete[] phiAlpha;
	delete[] PP;
	return 0;
}