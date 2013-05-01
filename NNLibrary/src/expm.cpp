#include "expm.h"
#include <vector>
#include "linearAlgebra.h"
using namespace std;
#define EXPM_INF 1e31
void expm_init_double(int *mVals, double *theta){
	mVals[0]=3;
	mVals[1]=5;
	mVals[2]=7;
	mVals[3]=9;
	mVals[4]=13;

	theta[0]=1.495585217958292e-002;
	theta[1]=2.539398330063230e-001;
	theta[2]=9.504178996162932e-001;
	theta[3]=2.097847961257068e+000;
	theta[4]=5.371920351148152e+000;
}
void getPadeCoefficients(int m, double *c){
	switch(m){
		case 3:{
			double C[4]={120, 60, 12, 1};
			memcpy(c, C, 4*sizeof(double));
		}
		break;
		case 5:{
			double C[6]={30240, 15120, 3360, 420, 30, 1};
			memcpy(c, C, 6*sizeof(double));
		}
		break;
		case 7:{
			double C[8]={17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1};
			memcpy(c, C, 8*sizeof(double));
		}
		break;
		case 9:{
			double C[10]={17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1};
			memcpy(c, C, 10*sizeof(double));
		}
		break;
		case 13:{
			double C[14]={64764752532480000, 32382376266240000, 7771770303897600,
                         1187353796428800,  129060195264000,   10559470521600,
                         670442572800,      33522128640,       1323241920,
                         40840800,          960960,            16380,  182,  1};
			memcpy(c, C, 14*sizeof(double));
		}
		break;
	}
}

//A must be symmetric or be given by columns
void PadeApproximant(double *A, int n, int deg, double *dest){
	double *c=new double[deg+1];
	getPadeCoefficients(deg,c);
	if(deg==13){
		double *tmpMat=new double[7*n*n];
		double *A2=&tmpMat[0];
		double *A4=&tmpMat[n*n];
		double *A6=&tmpMat[2*n*n];
		double *U=&tmpMat[3*n*n];
		double *V=&tmpMat[4*n*n];
		multMatrixMatrix(A, A, n, n, n, A2);
		multMatrixMatrix(A2, A2, n, n, n, A4);
		multMatrixMatrix(A2, A4, n, n, n, A6);
		memset(U,0,sizeof(double)*n*n);
		memset(V,0,sizeof(double)*n*n);
		double *a=A;
		double *a2=A2;
		double *a4=A4;
		double *a6=A6;
		//compute U
		double *u=U;
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j,++u,++a2, ++a4, ++a6){
				*u=c[13]**a6+c[11]**a4+c[9]**a2;
			}
		}
		multMatrixMatrix<double>(A6,U,n,U);
		u=U;
		a2=A2;
		a4=A4;
		a6=A6;
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j,++u,++a2, ++a4, ++a6){
				*u+=c[7]**a6+c[5]**a4+c[3]**a2;
				if(i==j){
					*u+=c[1];
				}
			}
		}
		multMatrixMatrix<double>(A,U,n,U);
		//compute V
		double *v=V;
		a2=A2;
		a4=A4;
		a6=A6;
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j,++v,++a2, ++a4, ++a6){
				*v=c[12]**a6+c[10]**a4+c[8]**a2;
			}
		}
		multMatrixMatrix<double>(A6,V,n,V);
		v=V;
		a2=A2;
		a4=A4;
		a6=A6;
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j,++v,++a2, ++a4, ++a6){
				*v+=c[6]**a6+c[4]**a4+c[2]**a2;
				if(i==j){
					*v+=c[0];
				}
			}
		}
		//----sum and difference----
		double *UPV=&tmpMat[5*n*n];
		double *VMU=&tmpMat[6*n*n];
		double *upv=UPV;
		double *vmu=VMU;
		u=U;
		v=V;
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j,++u,++v,++upv, ++vmu){
				*upv=*u+*v;
				*vmu=*v-*u;
			}
		}
		computeInverse(VMU,n);
		multMatrixMatrix(VMU,UPV,n,dest);
		delete[] tmpMat;
	}else{
		int nPows=1+deg/2;
		double *powA=new double[(4+nPows)*n*n];
		setIdentity<double>(powA,n);
		multMatrixMatrix<double>(A, A, n, n, n, &powA[n*n]);
		for(int j=2;j<nPows;++j){
			multMatrixMatrix<double>(&powA[(j-1)*n*n], &powA[n*n], n, n, n, &powA[j*n*n]);
		}
		double *U = &powA[nPows*n*n];
		double *V = &powA[(nPows+1)*n*n];
		double *UPV=&powA[(nPows+2)*n*n];
		double *VMU=&powA[(nPows+3)*n*n];
		memset(U, 0, sizeof(double)*n*n);
		memset(V, 0, sizeof(double)*n*n);
		for(int j=deg;j>=1;j-=2){
			linCombVector<double>(U,1.0, &powA[n*n*(j/2)], c[j],n*n,U);
		}
		multMatrixMatrix<double>(A,U,n,U);
		for(int j=deg-1;j>=0;j-=2){
			linCombVector<double>(V,1.0, &powA[n*n*((j+1)/2)], c[j],n*n,V);
		}
		linCombVector<double>(U,1.0, V, 1.0 ,n*n,UPV);
		linCombVector<double>(V,1.0, U, -1.0,n*n,VMU);
		computeInverse(VMU,n);
		multMatrixMatrix(VMU,UPV,n,dest);
		delete[] powA;
	}
	delete[] c;
}

double norm_1(double *A, int n, int m){
	double maxSum=-EXPM_INF;
	for(int j=0;j<m;++j){
		double *a=&A[j];
		double sum=0;
		for(int i=0;i<n;++i, a+=m){
			sum+=fabs(*a);
		}
		if(maxSum<sum){
			maxSum=sum;
		}
	}
	return maxSum;
}

void log2(double x, double &t, int &s){
	s=0;
	t=x;
	while(t>1){
		++s;
		t/=2;
	}
}

void expm(double *A, int n, double *_dest){
	double *dest=_dest;
	bool dellocate=false;
	if(A==_dest){
		dellocate=true;
		dest=new double[n*n];
	}

	double theta[5];
	int mVals[5];
	expm_init_double(mVals, theta);
	double normA=norm_1(A, n, n);
	if(normA<=theta[4]){
		for(int i=0;i<=4;++i){
			if(normA<=theta[i]){
				PadeApproximant(A,n,mVals[i],dest);
				break;
			}
		}
	}else{
		int s;
		double t;
		log2(normA/theta[4],t,s);
		double scale=1<<s;
		double *F=new double[n*n];
		memcpy(F,A,sizeof(double)*n*n);
		multVectorScalar(A,1.0/scale,n*n,F);
		PadeApproximant(F,n,mVals[4],dest);
		for(int i=0;i<s;++i){
			multMatrixMatrix(dest,dest,n,dest);
		}
		delete[] F;
	}
	if(dellocate){
		memcpy(_dest, dest, sizeof(double)*n*n);
		delete[] dest;
	}
	
}


