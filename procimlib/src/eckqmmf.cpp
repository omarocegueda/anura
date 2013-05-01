#include "SparseMatrix.h"
#include <string.h>
#include "linearalgebra.h"
//=======================================================
//================= EC-KQMMF=============================
//=======================================================

void computeKernelIntegralFactors(double *P, int nrows, int ncols, int K, double *kernel, double *s, double *sk1, double *sk2){
	int n=nrows*ncols;
	memset(s,0,sizeof(double)*K);
	memset(sk1,0,sizeof(double)*K*n);
	memset(sk2,0,sizeof(double)*K);
	for(int k=0;k<K;++k){
		for(int x=0;x<n;++x){
			double p=P[K*(x)+k];
			s[k]+=p*p;
			for(int y=0;y<n;++y){
				double q=P[K*(y)+k];
				sk1[K*x+k]+=q*q*kernel[x*n+y];
				sk2[k]+=p*p*q*q*kernel[x*n+y];
			}
		}
	}
}


void computeKernelIntegralFactors(double *P, SparseMatrix &kernel, int K, double *s, double *sk1, double *sk2){
	int n=kernel.n;
	memset(s,0,sizeof(double)*K);
	memset(sk1,0,sizeof(double)*K*n);
	memset(sk2,0,sizeof(double)*K);
	for(int k=0;k<K;++k){
		for(int x=0;x<n;++x){
			double p=P[K*(x)+k];
			s[k]+=p*p;
			for(int j=0;j<kernel.degree[x];++j){
				int y=kernel.edges[x][j].destination;
				double w=kernel.edges[x][j].w;
				double q=P[K*(y)+k];
				sk1[K*x+k]+=q*q*w;
				sk2[k]+=p*p*q*q*w;
			}
			//---add diagonal term---
			sk1[K*x+k]+=p*p*kernel.diagonal[x];
			sk2[k]+=p*p*p*p*kernel.diagonal[x];
		}
	}
}

void iterateP_kernel(double *P, int nrows, int ncols, int K, double lambda, double mu, double *kernel, double *s, double *sk1, double *sk2){
	const int NUM_NEIGHBORS=4;
	const int dRow[]={-1, 0, 1, 0};
	const int dCol[]={0, 1, 0, -1};
	
	double *alpha=new double[K];
	double *beta=new double[K];
	double *den=new double[K];
	int n=nrows*ncols;
	int x=0;
	for(int r=0;r<nrows;++r)for(int c=0;c<ncols;++c, ++x){
		//-----compute alpha_k(x)-----
		memset(alpha, 0, sizeof(double)*K);
		int neighCount=0;
		for(int i=0;i<NUM_NEIGHBORS;++i){
			int rr=r+dRow[i];
			int cc=c+dCol[i];
			if(IN_RANGE(rr, 0, nrows) && IN_RANGE(cc, 0, ncols)){
				++neighCount;
				int y=rr*ncols+cc;
				for(int k=0;k<K;++k){
					alpha[k]+=lambda*P[K*y+k];
				}
			}
		}
		//-----compute beta_k(x) and sum_i{alpha_i(x)/beta_i(x)}-----
		double num=0;
		for(int k=0;k<K;++k){
			beta[k]=kernel[x*n+x] - 2*sk1[K*x+k]/s[k] + sk2[k]/(s[k]*s[k]) - mu + lambda*neighCount;
			num+=alpha[k]/beta[k];
		}
		//-----compute, for each k, sum_i{beta_k(x)/beta_i(x)}-----
		for(int k=0;k<K;++k){
			den[k]=0;
			for(int i=0;i<K;++i){
				den[k]+=beta[k]/beta[i];
			}
		}
		//-----update the measure field P_k(x) for all k
		for(int k=0;k<K;++k){
			double &p=P[x*K+k];
			p=alpha[k]/beta[k]+(1-num)/(den[k]);
		}
	}
	delete[] den;
	delete[] alpha;
	delete[] beta;
}

double iterateP_kernel(double *P, double *pbuff, SparseMatrix &kernel, int K, double lambda, double mu, double *s, double *sk1, double *sk2){
	double *alpha=new double[K];
	double *beta=new double[K];
	double *den=new double[K];
	int n=kernel.n;
	double error=0;
	memcpy(pbuff, P, sizeof(double)*n*K);
	for(int x=0;x<n;++x){
		//-----compute alpha_k(x)-----
		memset(alpha, 0, sizeof(double)*K);
		for(int j=0;j<kernel.degree[x];++j){
			int y=kernel.edges[x][j].destination;
			for(int k=0;k<K;++k){
				alpha[k]+=lambda*pbuff[K*y+k];
			}
		}
		//-----compute beta_k(x) and sum_i{alpha_i(x)/beta_i(x)}-----
		double num=0;
		for(int k=0;k<K;++k){
			beta[k]=kernel.diagonal[x] - 2*sk1[K*x+k]/s[k] + sk2[k]/(s[k]*s[k]) - mu + lambda*kernel.degree[x];
			num+=alpha[k]/beta[k];
		}
		//-----compute, for each k, sum_i{beta_k(x)/beta_i(x)}-----
		for(int k=0;k<K;++k){
			den[k]=0;
			for(int i=0;i<K;++i){
				den[k]+=beta[k]/beta[i];
			}
		}
		//-----update the measure field P_k(x) for all k
		for(int k=0;k<K;++k){
			double &p=P[x*K+k];
			double newVal=alpha[k]/beta[k]+(1-num)/(den[k]);
			error+=SQR(newVal-p);
			p=newVal;
		}
	}
	error=sqrt(error/(K*n));
	delete[] den;
	delete[] alpha;
	delete[] beta;
	return error;
}

void computeSparseKernel(double sigma, double *img, int nrows, int ncols, int numBands, double *kernel){
	double sigma2=sigma*sigma;
	const int NUM_NEIGHBORS=8;
	const int dRow[]={-1, 0, 1, 0, -1, -1, 1, 1};
	const int dCol[]={0, 1, 0, -1, -1, 1, 1, -1};
	int n=nrows*ncols;
	memset(kernel,0,sizeof(double)*n*n);
	int x=0;
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c, ++x){
			kernel[x*n+x]=1;
			for(int i=0;i<NUM_NEIGHBORS;++i){
				int rr=r+dRow[i];
				int cc=c+dCol[i];
				if(IN_RANGE(rr, 0, nrows) && IN_RANGE(cc, 0, ncols)){
					int y=rr*ncols+cc;
					double d2=sqrDistance(&img[x*numBands], &img[y*numBands], numBands);
					kernel[x*n+y]=exp(-d2/sigma2);
					//kernel[x*n+y]=img0[x]*img0[y]+img1[x]*img1[y];
				}
			}
		}
	}
}

void computeDenseKernel(double sigma, double *img, int nrows, int ncols, int numBands, double *kernel){
	double sigma2=sigma*sigma;
	const int NUM_NEIGHBORS=8;
	const int dRow[]={-1, 0, 1, 0, -1, -1, 1, 1};
	const int dCol[]={0, 1, 0, -1, -1, 1, 1, -1};
	int n=nrows*ncols;
	memset(kernel,0,sizeof(double)*n*n);
	int x=0;
	for(int x=0;x<n;++x){
		kernel[x*n+x]=1;
		int ix=x/ncols;
		int jx=x%ncols;
		for(int y=0;y<n;++y){
			int iy=y/ncols;
			int jy=y%ncols;
			/*if((abs(ix-iy) + abs(jx-jy))>2){
				continue;
			}*/
			double d2=sqrDistance(&img[x*numBands], &img[y*numBands], numBands);
			/*if(d2>25){
				continue;
			}*/
			kernel[x*n+y]=exp(-d2/sigma2);
			//kernel[x*n+y]=-d2/sigma2;
			/*double prod=dotProduct(&img[x*numBands], &img[y*numBands], numBands);
			kernel[x*n+y]=prod;*/
		}
	}
}

/*void testECKQMMF(void){
	int nrows=32;
	int ncols=32;
	int n=nrows*ncols;
	int numBands=2;
	double *img=new double[numBands*nrows*ncols];
	generateSynthetic(img, nrows, ncols, numBands);
	//---show input---
	cv::Mat M0, M1;
	M0.create(nrows,ncols,CV_8UC1);
	M1.create(nrows,ncols,CV_8UC1);
	unsigned char *M0_data=(unsigned char*)M0.data;
	unsigned char *M1_data=(unsigned char*)M1.data;
	showBand(img, nrows, ncols, numBands, 0, M0_data);
	showBand(img, nrows, ncols, numBands, 1, M1_data);
	cv::imshow("X",M0);
	cv::imshow("Y",M1);
	//----------------
	double sigma=2;
	double *kernel=new double[n*n];
	
	const int maxIter=15;
	int numModels=2;
	double *P=new double[n*numModels];
	double lambda=1.5;
	double mu=0.5;

	double *s=new double[numModels];
	double *sk1=new double [n*numModels];
	double *sk2=new double[numModels];
	//----initialize P field----
	memset(P,0,sizeof(double)*n*numModels);
	for(int x=0;x<n;++x){
		//int sel=rand()%numModels;
		if((x%ncols)<5){
			P[numModels*x]=1;	
		}else if((x%ncols)>(ncols-5)){
			P[numModels*x+1]=1;
		}else{
			P[numModels*x]=0.5;	
			P[numModels*x+1]=0.5;
		}
	}
	//--------------------------
	computeDenseKernel(sigma, img, nrows, ncols, numBands, kernel);
	//computeSparseKernel(sigma, img, nrows, ncols, numBands, kernel);
	for(int iter=0;iter<maxIter;++iter){
		computeKernelIntegralFactors(P, nrows, ncols, numModels, kernel, s, sk1, sk2);
		iterateP_kernel(P, nrows, ncols, numModels, lambda, mu, kernel, s, sk1, sk2);
	}
	double *p0=new double[n];
	double *p1=new double[n];
	for(int x=0;x<n;++x){
		p0[x]=P[numModels*x+0];
		p1[x]=P[numModels*x+1];
	}
	//---show outpt---
	cv::Mat P0, P1;
	P0.create(nrows,ncols,CV_8UC1);
	P1.create(nrows,ncols,CV_8UC1);
	unsigned char *P0_data=(unsigned char*)P0.data;
	unsigned char *P1_data=(unsigned char*)P1.data;
	showImg(p0, nrows, ncols, P0_data);
	showImg(p1, nrows, ncols, P1_data);
	cv::imshow("P0",P0);
	cv::imshow("P1",P1);
	//----------------
	

	
	int key=0;
	while(key!=13){
		key=cv::waitKey(40);
	}

	delete[] p0;
	delete[] p1;
	delete[] s;
	delete[] sk1;
	delete[] sk2;
	delete[] kernel;
	delete[] P;
	delete[] img;

}*/

