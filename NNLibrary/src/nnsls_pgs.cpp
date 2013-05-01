#include "nnsls_pgs.h"
#include "linearalgebra.h"
#include "geometryutils.h"
#include <iostream>
#include <iomanip>
using namespace std;
//#define PGS_DEBUG

double columnCorr(double *Phi, int n, int m, int col, double *alpha, double *s){
	double *res=new double[n];
	multMatrixVector(Phi, alpha, n, m, res);
	double *x=&Phi[col];
	double corr=0;
	for(int i=0;i<n;++i, x+=m){
		corr+=res[i]*(*x);
	}
	delete[] res;
	return corr;
}
int nnsls_pgs(double *Phi, int n, int m, double *s, double *alpha, double lambda, double *error){
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
	int maxIter=50;
	double c=1;
	for(int iter=0;(iter<maxIter) && (tol<rmse);++iter){
		memcpy(alphak,alpha,sizeof(double)*m);
#ifdef PGS_DEBUG
		double sum=0;
		int nz=0;
		for(int i=0;i<m;++i){
			if(alpha[i]>0){
				++nz;
			}
			sum+=alpha[i];
		}
		cout<<endl;
		cout<<"c="<<c<<". Sum: "<<sum<<". Nz="<<nz<<":";
		for(int i=0;i<m;++i){
			if(alpha[i]>0){
				double corr=columnCorr(Phi, n, m, i, alpha, s);
				cout<<"\t"<<i<<"("<<setprecision(3)<<alpha[i]<<")";
				//cout<<"\t"<<i<<"("<<setprecision(5)<<corr<<")";
			}
		}
		

#endif
        //------------------------
        // Proyected Gauss Seidel iteration
        //-----------------------
		
		for(int t=0;t<5;++t){
			double *g=G;
			for(int k=0;k<m;++k, g+=m){
				double prod=dotProduct(g, alpha,m);
				double val=alpha[k]-prod+b[k];
				alpha[k]=MAX(val,0);
			}
		}
		
        //-------------------
        // compressed subspace minimization 
        //-------------------
		for(int iterMinSb=0;iterMinSb<nIterCompress;++iterMinSb){
			
			for(int t=0;t<10;++t){
				g=G;
				for(int k=0;k<m;++k, g+=m)if(alpha[k]>0){
					double prod=dotProduct(g, alpha,m);
					double val=alpha[k]-prod+b[k];
					alpha[k]=MAX(val,0);
				}
			}
			
            //scale factor
			multMatrixVector<double>(Phi, alpha, n, m, phiAlpha);
			double num=dotProduct(phiAlpha, s, n);
			double den=evaluateQuadraticForm(PP,alpha,m);
			c=num/den;
			multVectorScalar(alpha, num/den, m, alpha);
			//multVectorScalar(alpha, den/num, m, alpha);
		}
		//-----------------
		rmse=0;
		for(int i=0;i<m;++i){
			rmse+=SQR(alphak[i]-alpha[i]);
		}
		rmse=sqrt(rmse);
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


/*int nnsls_pgs(double *Phi, int n, int m, double *s, double *alpha, double lambda, double *error){
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
	double C=dotProduct(s,s,n);
	double *g=G;
	double *ss=step;
	for(int i=0;i<m;++i, ++ss){
		for(int j=0;j<m;++j, ++g){
			(*g)*=(*ss);
		}
		b[i]*=*ss;
	}
    double tol=1e-10;
	double epsilon=1e-8;
	double rmse=tol+1;
	int maxIter=50;
	double c=1;
	for(int iter=0;(iter<maxIter) && (tol<rmse);++iter){
		memcpy(alphak,alpha,sizeof(double)*m);
#ifdef PGS_DEBUG
		double sum=0;
		int nz=0;
		for(int i=0;i<m;++i){
			if(alpha[i]>0){
				++nz;
			}
			sum+=alpha[i];
		}
		cout<<endl;
		cout<<"c="<<c<<". Sum: "<<sum<<". Nz="<<nz<<":";
		for(int i=0;i<m;++i){
			if(alpha[i]>0){
				double corr=columnCorr(Phi, n, m, i, alpha, s);
				cout<<"\t"<<i<<"("<<setprecision(3)<<alpha[i]<<")";
				//cout<<"\t"<<i<<"("<<setprecision(5)<<corr<<")";
			}
		}
		

#endif
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
			for(int k=0;k<m;++k, g+=m)if(alpha[k]>epsilon){
				double prod=dotProduct(g, alpha,m);
				double val=alpha[k]-prod+b[k];
				alpha[k]=MAX(val,0);
			}
            //scale factor
			multMatrixVector<double>(Phi, alpha, n, m, phiAlpha);
			double num=dotProduct(phiAlpha, s, n);
			double den=evaluateQuadraticForm(PP,alpha,m);
			c=num/den;
			multVectorScalar(alpha, num/den, m, alpha);
			//multVectorScalar(alpha, den/num, m, alpha);
		}
		//-----------------
		rmse=0;
		for(int i=0;i<m;++i){
			rmse+=SQR(alphak[i]-alpha[i]);
		}
		rmse=sqrt(rmse);
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
*/