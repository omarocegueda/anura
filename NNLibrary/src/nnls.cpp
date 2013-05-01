/*Non-negative Least Squares solver
	Author: Omar Ocegueda
*/

#include "nnls.h"
#include "ls.h"
#include "linearalgebra.h"
#include "statisticsutils.h"
#include <string.h>
#include <string.h>
#include <set>
#include <math.h>
#define EPS_NNLS 1e-10
#define INF_NNLS 1e+31
using namespace std;
extern "C"{
	#include "f2c.h"
	#include "clapack.h"
}
void computeHouseholderTransformation(double *x, int n, int index, double *H){
	if(n==1){
		H[0]=1;
		return;
	}
	int sqn=n*n;
	memset(H,0,sizeof(double)*sqn);
	double sqNorm=0;
	for(double *xp=x+(n-1);x<=xp;--xp){
		sqNorm+=(*xp)*(*xp);
	}
	double norm=sqrt(sqNorm);
	double denFactor=1.0/(sqNorm-x[index]*norm);
	double *hrow=H+index*n, *hcol=H+index, *h=H, *xi=x, *xj;
	int i,j;
	for(i=0;i<n;++i, hcol+=n, ++hrow, ++xi){
		xj=x;
		for(j=0;j<n;++j, ++h, ++xj){
			*h-=*xi**xj;
		}
		*hcol+=norm**xi;
		*hrow+=norm**xi;
	}
	H[index*(n+1)]-=sqNorm;
	for(h=H+(sqn-1);H<=h;--h){
		*h*=denFactor;
	}
	for(h=H+(sqn-1);H<=h;h-=(n+1)){
		*h+=1;
	}
}

void QRDecomposition(double *A, int n, int m, double *Q, double *R){
	double *a=new double[n];
	double *b=new double[n];
	double *r=new double[n];
	double *QQ=new double[n*n];
	//double *I=new double[n*n];
	//---first column---
	for(int i=0;i<n;++i){
		a[i]=A[i*m];
	}
	computeHouseholderTransformation(a, n, 0, Q);
	memset(R,0,sizeof(double)*n*m);
	multMatrixVector<double>(Q,a,n,n,a);
	for(int i=0;i<n;++i){
		R[i*m]=a[i];
	}
	//multMatrixMatrix<double>(Q,Q,n,I);
	//------------------
	for(int j=1;j<m;++j){
		//---get j-th column from A---
		for(int i=0;i<n;++i){
			a[i]=A[i*m+j];
		}
		//----------------------------
		multMatrixVector<double>(Q,a,n,n,b);
		computeHouseholderTransformation(b, n, j, QQ);
		multMatrixVector<double>(QQ,b,n,n,r);
		multMatrixMatrix<double>(QQ,Q,n,Q);
		multMatrixMatrix<double>(QQ,R,n,n,m,R);
		for(int i=0;i<n;++i){
			R[i*m+j]=r[i];
		}
	}
}

void testHouseholderTransformation(void){
	const int N=5;
	double x[N];
	double y[N];
	double H[N*N];
	for(int i=0;i<N;++i){
		x[i]=uniform(0,1);
	}
	computeHouseholderTransformation(x, N, 0, H);
	
	
	multMatrixVector<double>(H,x,N,N,y);
	for(int i=0;i<N;++i){
		cerr<<y[i]<<",";
	}
	cerr<<endl;

	const int M=2;
	double A[N*M];
	double Q[N*N];
	double R[N*M];
	double RR[N*M];
	for(int i=0;i<N*M;++i){
		A[i]=uniform(0,1);
	}
	QRDecomposition(A, N, M, Q, R);
	multMatrixMatrix<double>(Q, A, N, N, M, RR);
	for(int i=0;i<N*M;++i){
		if(fabs(R[i]-RR[i])>EPSILON){
			cerr<<i<<endl;
		}
	}
	

}
//#define USE_BLAS
int nnls(double *E, long n, long m, double *f, double *x, double *error){
	int N=MAX(n,m);
	double *w=new double[N];
	double *wcpy=new double[m];
	double *z=new double[m];
	double *zp=new double[N];
	double *EE=new double[N*N];
	int *I=new int[m];
	set<int> Z;
	set<int> P;
	for(int i=0;i<m;++i){
		Z.insert(i);
	}

	memset(x, 0, sizeof(double)*m);
	bool finished=false;
	int maxIter=3*m;
	int iter=0;
	while((!finished) && (iter<maxIter)){
		++iter;
		//--compute E^T(f-Ex)--
#ifdef USE_BLAS
		{
			double zero=0;
			long lone=1;
			double done=1;
			dgemv_("T",&m, &n, &done,E,&m,x,&lone,&zero,w,&lone);
		}
#else
		multMatrixVector<double>(E,x,n,m,w);
#endif
		for(int i=0;i<n;++i){
			w[i]=f[i]-w[i];
		}
		
		
		
#ifdef USE_BLAS
		{
			double zero=0;
			long lone=1;
			double done=1;
			memcpy(wcpy, w, sizeof(double)*m);	
			dgemv_("N",&m, &n, &done,E,&m,wcpy,&lone,&zero,w,&lone);
		}
#else
		multVectorMatrix<double>(w,E,n,m,w);
#endif
		//--select the maximum w_i, i in Z
		set<int>::iterator sel=Z.begin();
		for(set<int>::iterator it=Z.begin();it!=Z.end();++it){
			if(w[*sel]<w[*it]){
				sel=it;
			}
		}
		//-------------------------------
		if((sel==Z.end()) || w[*sel]<EPS_NNLS){
			break;
		}
		int val=*sel;
		Z.erase(sel);
		P.insert(val);
		//----solve unrestricted least squares defined by P----
		bool nullIndicesChanged=true;
		while(nullIndicesChanged){
			int mm=P.size();
			int ii=0;
			for(set<int>::iterator it=P.begin();it!=P.end();++it, ++ii){
				I[ii]=*it;
			}
			nullIndicesChanged=false;
			solveSubsetLeastSquares(E, n, m, f, I, mm, EE, zp);
			memset(z,0,sizeof(double)*m);
			bool positive=true;
			for(int i=0;i<mm;++i){
				if(zp[i]<EPS_NNLS){
					if(-EPSILON<zp[i]){
						zp[i]=0;
					}
					positive=false;
				}
				z[I[i]]=zp[i];
			}
			//-----------------------------------------------------
			if(positive){
				memcpy(x,z,sizeof(double)*m);
				break;
			}
			//------move x towards the unrestricted least squares solution z-------
			double alpha=INF_NNLS;
			for(int i=0;i<mm;++i)if(zp[i]<=0){
				if(fabs(x[I[i]]-zp[i])<EPS_NNLS){
					continue;
				}
				double opc=x[I[i]]/(x[I[i]]-zp[i]);
				if(opc<alpha){
					alpha=opc;
				}
			}
			if(alpha==INF_NNLS){
				alpha=0;
			}
			if(fabs(alpha)>EPS_NNLS){
				for(int i=0;i<mm;++i){
					x[I[i]]+=alpha*(zp[i]-x[I[i]]);
					
				}
			}
			//------move from P to Z all indices i such that x[i]=0
			for(int i=0;i<mm;++i){
				if(fabs(x[I[i]])<EPS_NNLS){
					x[I[i]]=0;
					P.erase(I[i]);
					Z.insert(I[i]);
					nullIndicesChanged=true;
				}
			}
		}
	}
	if(error!=NULL){
#ifdef USE_BLAS		
		{
			double zero=0;
			long lone=1;
			double done=1;
			dgemv_("T",&m, &n, &done,E,&m,x,&lone,&zero,w,&lone);
		}
#else
		multMatrixVector<double>(E,x,n,m,w);
#endif
		*error=0;
		for(int i=0;i<n;++i){
			*error+=SQR(w[i]-f[i]);
		}
		//*error=*error;
		//*error=sqrt(*error/n);
	}
	delete[] w;
	delete[] wcpy;
	delete[] EE;
	delete[] I;
	delete[] zp;
	delete[] z;
	return 0;
}

int nnls_geman_mcclure(double *E, long n, long m, double *f, double *x, double *error){
	double *z=new double[n];
	for(int i=0;i<n;++i){
		z[i]=1;
	}
	double *Ecurrent=new double[n*m];
	double *fcurrent=new double[n];
	double *residual=new double[n];
	double err=1;
	int iter=0;
	int maxIter=30;
	while((1e-9<err) && (iter<maxIter)){
		for(int i=0;i<n;++i){
			for(int j=0;j<m;++j){
				Ecurrent[i*m+j]=E[i*m+j]*z[i];
			}
			fcurrent[i]=z[i]*f[i];
		}
		nnls(Ecurrent, n,m,fcurrent,x,error);
		multMatrixVector<double>(E,x,n,m,residual);
		for(int i=0;i<n;++i){
			residual[i]=SQR(residual[i]-f[i]);
		}
		err=0;
		for(int i=0;i<n;++i){
			double newz=4.0/SQR(2.0+residual[i]);
			err+=fabs(z[i]-newz);
			z[i]=newz;
		}
		++iter;
	}
	//cerr<<iter<<endl;
	delete[] z;
	delete[] Ecurrent;
	delete[] fcurrent;
	delete[] residual;
	return 0;
}


int nnls_tukey_biweight(double *E, long n, long m, double *f, double sigma, double *x, double *error){
	double *z=new double[n];
	for(int i=0;i<n;++i){
		z[i]=1;
	}
	double *Ecurrent=new double[n*m];
	double *fcurrent=new double[n];
	double *residual=new double[n];
	double err=1;
	int iter=0;
	int maxIter=30;
	while((1e-9<err) && (iter<maxIter)){
		for(int i=0;i<n;++i){
			for(int j=0;j<m;++j){
				Ecurrent[i*m+j]=E[i*m+j]*z[i];
			}
			fcurrent[i]=z[i]*f[i];
		}
		nnls(Ecurrent, n,m,fcurrent,x,error);
		multMatrixVector<double>(E,x,n,m,residual);
		for(int i=0;i<n;++i){
			residual[i]=SQR(residual[i]-f[i]);
		}
		err=0;
		for(int i=0;i<n;++i){
			double newz=0;
			if(residual[i]<sigma){
				newz=SQR(1-residual[i]/sigma);
			}
			err+=fabs(z[i]-newz);
			z[i]=newz;
		}
		++iter;
	}
	//cerr<<iter<<endl;
	delete[] z;
	delete[] Ecurrent;
	delete[] fcurrent;
	delete[] residual;
	return 0;
}


int nnls_subspace(double *E, int n, int m, double *f, double *x, int *I, int mm, double *error){
	double *EE=new double[n*mm];
	double *z=new double[mm];
	for(int i=0;i<n;++i){
		for(int j=0;j<mm;++j){
			EE[i*mm+j]=E[i*m+I[j]];
		}
	}
	nnls(EE, n, mm, f, z, error);
	memset(x, 0, sizeof(double)*m);
	for(int i=0;i<mm;++i){
		x[I[i]]=z[i];
	}
	delete[] EE;
	delete[] z;
	return 0;
}