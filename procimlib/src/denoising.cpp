#include "denoising.h"
#include <math.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include "bits.h"
using namespace std;
#ifndef SQR
#define SQR(x) ((x)*(x))
#endif
double getOptimalWeight(double sq_residual, EDenoisingType denoisingType, double *params){
	switch(denoisingType){
		case EDT_GEMAN_MCCLURE:
			return 4.0/SQR(2.0+sq_residual);
		break;
		case EDT_TUKEY_BIWEIGHT:
			if(params[0]<sq_residual){
				return 0;
			}
			return SQR(1-sq_residual/params[0]);
		break;
	}
	return 1;
}
int robustDenoising(double *img, int r, int c, double lambda, EDenoisingType denoisingType, double *params, double *f, double *Z){
	/*if(denoisingType==EDT_TGV){
		filterTGV_L2(img, r, c, lambda, params[0], params[1], params[2], params[3], params[4], f, NULL);
		return 0;
	}*/
	
	double *zh=Z;
	double *zv=Z+r*c;
	//--initialize line process--
	for(int i=0;i<r;++i){
		for(int j=0;j<c;++j){
			zh[i*r+c]=1;
			zv[i*r+c]=1;
		}
	}
	memcpy(f,img,sizeof(double)*r*c);
	//------------------------
	int maxIter=30;
	int iter;
	double error=1;
	double eps=1e-6;
	for(iter=0;(iter<maxIter) && (eps<error);++iter){
		//----optimize for fixed Z--
		int maxInternIter=30;
		int internIter;
		error=eps+1;
		for(internIter=0;(internIter<maxInternIter) && (eps<error);++internIter){
			//-----------------------
			error=0;
			for(int i=0;i<r;++i){
				for(int j=0;j<c;++j){
					double &f_current=f[i*c+j];
					int nCount=0;
					double fz_sum=0;
					double z_sum=0;
					if(0<i){
						++nCount;
						double f_up=f[(i-1)*c+j];
						double zv_up=zv[(i-1)*c+j];
						fz_sum+=f_up*zv_up;
						z_sum+=zv_up;
					}
					if(i<r-1){
						++nCount;
						double f_down=f[(i+1)*c+j];
						double zv_down=zv[i*c+j];
						fz_sum+=f_down*zv_down;
						z_sum+=zv_down;
					}
					if(0<j){
						++nCount;
						double f_left=f[i*c+j-1];
						double zh_left=zh[i*c+j-1];
						fz_sum+=f_left*zh_left;
						z_sum+=zh_left;
					}
					if(j<c-1){
						++nCount;
						double f_right=f[i*c+j+1];
						double zh_right=zh[i*c+j];
						fz_sum+=f_right*zh_right;
						z_sum+=zh_right;
					}
					double f_prev=f_current;
					f_current=(img[i*c+j]+lambda*fz_sum)/(1+lambda*z_sum);
					error+=fabs(f_prev-f_current);
				}
			}
			error/=(r*c);
		}
		//---solve Z for fixed f----
		error=0;
		//vertical
		for(int i=0;i<r-1;++i){
			for(int j=0;j<c;++j){
				double residual=f[i*c+j]-f[(i+1)*c+j];
				double newz=getOptimalWeight(residual*residual, denoisingType, params);
				error+=fabs(zv[i*c+j]-newz);
				zv[i*c+j]=newz;
			}
		}
		//horizontal
		for(int i=0;i<r;++i){
			for(int j=0;j<c-1;++j){
				double residual=f[i*c+j]-f[i*c+j+1];
				double newz=getOptimalWeight(residual*residual, denoisingType, params);
				error+=fabs(zh[i*c+j]-newz);
				zh[i*c+j]=newz;
			}
		}
		error/=(r*(c-1) + c*(r-1));
	}
	return 0;
}

int robustDenoisingOR(double *img, int r, int c, double lambda, EDenoisingType denoisingType, double *params, double *f, double *Z, double *M){
	double *zh=Z;
	double *zv=Z+r*c;
	//--initialize line process--
	int pos=0;
	for(int i=0;i<r;++i){
		for(int j=0;j<c;++j, ++pos){
			zh[pos]=1;
			zv[pos]=1;
			M[pos] =1;
		}
	}
	memcpy(f,img,sizeof(double)*r*c);
	//------------------------
	int maxIter=30;
	int iter;
	double error=1;
	double eps=1e-6;
	for(iter=0;(iter<maxIter) && (eps<error);++iter){
		//----optimize for fixed Z--
		int maxInternIter=30;
		int internIter;
		error=eps+1;
		for(internIter=0;(internIter<maxInternIter) && (eps<error);++internIter){
			//-----------------------
			error=0;
			for(int i=0;i<r;++i){
				for(int j=0;j<c;++j){
					double &f_current=f[i*c+j];
					int nCount=0;
					double fz_sum=0;
					double z_sum=0;
					if(0<i){
						++nCount;
						double f_up=f[(i-1)*c+j];
						double zv_up=zv[(i-1)*c+j];
						fz_sum+=f_up*zv_up;
						z_sum+=zv_up;
					}
					if(i<r-1){
						++nCount;
						double f_down=f[(i+1)*c+j];
						double zv_down=zv[i*c+j];
						fz_sum+=f_down*zv_down;
						z_sum+=zv_down;
					}
					if(0<j){
						++nCount;
						double f_left=f[i*c+j-1];
						double zh_left=zh[i*c+j-1];
						fz_sum+=f_left*zh_left;
						z_sum+=zh_left;
					}
					if(j<c-1){
						++nCount;
						double f_right=f[i*c+j+1];
						double zh_right=zh[i*c+j];
						fz_sum+=f_right*zh_right;
						z_sum+=zh_right;
					}
					double f_prev=f_current;
					f_current=(img[i*c+j]*M[i*c+j]+lambda*fz_sum)/(M[i*c+j]+lambda*z_sum);
					error+=fabs(f_prev-f_current);
				}
			}
			error/=(r*c);
		}
		//---solve M for fixed f----
		error=0;
		//vertical
		pos=0;
		for(int i=0;i<r;++i){
			for(int j=0;j<c;++j, ++pos){
				double residual=f[pos]-img[pos];
				double newM=getOptimalWeight(residual*residual, denoisingType, params);
				error+=fabs(M[pos]-newM);
				M[pos]=newM;
			}
		}

		//---solve Z for fixed f----
		error=0;
		//vertical
		for(int i=0;i<r-1;++i){
			for(int j=0;j<c;++j){
				double residual=f[i*c+j]-f[(i+1)*c+j];
				double newz=getOptimalWeight(residual*residual, denoisingType, params);
				error+=fabs(zv[i*c+j]-newz);
				zv[i*c+j]=newz;
			}
		}
		//horizontal
		for(int i=0;i<r;++i){
			for(int j=0;j<c-1;++j){
				double residual=f[i*c+j]-f[i*c+j+1];
				double newz=getOptimalWeight(residual*residual, denoisingType, params);
				error+=fabs(zh[i*c+j]-newz);
				zh[i*c+j]=newz;
			}
		}
		error/=(r*(c-1) + c*(r-1));
	}
	return 0;
}

int robustVolumeDenoising(double *img, int r, int c, int s, double lambda, EDenoisingType denoisingType, double *params, double *f){
	const int NUM_NEIGHBORS=6;
	/*int dRow[]	={-1, 0, 1,  0,-1, 1, 1,-1, 0,  -1, 0, 1, 0,-1, 1, 1,-1,  -1, 0, 1, 0,-1, 1, 1,-1,  0};
	int dCol[]	={ 0, 1, 0, -1, 1, 1,-1,-1, 0,   0, 1, 0,-1, 1, 1,-1,-1,   0, 1, 0, -1, 1, 1,-1,-1, 0};
	int dSlice[]={-1,-1,-1, -1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1 ,1 ,1,  1};*/
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	int invNeigh[NUM_NEIGHBORS];
	for(int i=0;i<NUM_NEIGHBORS;++i){
		for(int j=i+1;j<NUM_NEIGHBORS;++j){
			if((dRow[i]==-dRow[j]) && (dCol[i]==-dCol[j]) && (dSlice[i]==-dSlice[j])){
				invNeigh[i]=j;
				invNeigh[j]=i;
			}
		}
	}
	int nvoxels=s*r*c;
	double *Z=new double[NUM_NEIGHBORS*nvoxels];
	//--initialize line process--
	for(int pos=NUM_NEIGHBORS*nvoxels-1;pos>=0;--pos){
		Z[pos]=1;
	}
	memcpy(f,img,sizeof(double)*nvoxels);
	//------------------------
	int maxIter=30;
	int iter;
	double error=1;
	double eps=1e-6;
	for(iter=0;(iter<maxIter) && (eps<error);++iter){
		//----optimize for fixed Z--
		int maxInternIter=30;
		int internIter;
		error=eps+1;
		for(internIter=0;(internIter<maxInternIter) && (eps<error);++internIter){
			//-----------------------
			error=0;
			int currentPos=0;
			for(int k=0;k<s;++k){
				for(int i=0;i<r;++i){
					for(int j=0;j<c;++j, ++currentPos){
						double &f_current=f[currentPos];
						int nCount=0;
						double fz_sum=0;
						double z_sum=0;
						for(int nn=0;nn<NUM_NEIGHBORS;++nn){
							int ii=i+dRow[nn];
							int jj=j+dCol[nn];
							int kk=k+dSlice[nn];
							if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
								int neighPos=kk*r*c+ii*c+jj;
								++nCount;
								double f_neigh=f[neighPos];
								double z_neigh=Z[currentPos*NUM_NEIGHBORS+nn];
								fz_sum+=f_neigh*z_neigh;
								z_sum+=z_neigh;
							}
						}
						double f_prev=f_current;
						f_current=(img[currentPos]+lambda*fz_sum)/(1+lambda*z_sum);
						error+=fabs(f_prev-f_current);
					}
				}
			}
			error/=(r*c*s);
		}
		//---solve Z for fixed f----
		error=0;
		int currentPosition=0;
		for(int k=0;k<s;++k){
			for(int i=0;i<r;++i){
				for(int j=0;j<c;++j, ++currentPosition){
					for(int nn=0;nn<NUM_NEIGHBORS;++nn){
						int ii=i+dRow[nn];
						int jj=j+dCol[nn];
						int kk=k+dSlice[nn];
						int neighPosition=kk*r*c+ii*c+jj;
						if(neighPosition<currentPosition){
							continue;//it will be computed later ----->
						}
						if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
							double residual=f[currentPosition]-f[neighPosition];
							double newz=getOptimalWeight(residual*residual, denoisingType, params);
							error+=fabs(Z[currentPosition*NUM_NEIGHBORS+nn]-newz);
							Z[currentPosition*NUM_NEIGHBORS+nn]=newz;
							Z[neighPosition*NUM_NEIGHBORS+invNeigh[nn]]=newz;// <-----here
						}
					}
				}
			}
		}
		//----------------------------------
		error/=(nvoxels);
	}
	delete[] Z;
	return 0;

}


int robustVolumeDenoisingOR(double *img, int r, int c, int s, double lambda, EDenoisingType denoisingType, double *params, double *f){
	const int NUM_NEIGHBORS=6;
	/*int dRow[]	={-1, 0, 1,  0,-1, 1, 1,-1, 0,  -1, 0, 1, 0,-1, 1, 1,-1,  -1, 0, 1, 0,-1, 1, 1,-1,  0};
	int dCol[]	={ 0, 1, 0, -1, 1, 1,-1,-1, 0,   0, 1, 0,-1, 1, 1,-1,-1,   0, 1, 0, -1, 1, 1,-1,-1, 0};
	int dSlice[]={-1,-1,-1, -1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1 ,1 ,1,  1};*/
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	int invNeigh[NUM_NEIGHBORS];
	for(int i=0;i<NUM_NEIGHBORS;++i){
		for(int j=i+1;j<NUM_NEIGHBORS;++j){
			if((dRow[i]==-dRow[j]) && (dCol[i]==-dCol[j]) && (dSlice[i]==-dSlice[j])){
				invNeigh[i]=j;
				invNeigh[j]=i;
			}
		}
	}
	int nvoxels=s*r*c;
	double *Z=new double[NUM_NEIGHBORS*nvoxels];
	double *M=new double[nvoxels];
	//--initialize line process--
	for(int pos=NUM_NEIGHBORS*nvoxels-1;pos>=0;--pos){
		Z[pos]=1;
	}
	for(int pos=nvoxels-1;pos>=0;--pos){
		M[pos]=1;
	}
	memcpy(f,img,sizeof(double)*nvoxels);
	//------------------------
	int maxIter=30;
	int iter;
	double error=1;
	double eps=1e-6;
	for(iter=0;(iter<maxIter) && (eps<error);++iter){
		//----optimize for fixed Z--
		int maxInternIter=30;
		int internIter;
		error=eps+1;
		for(internIter=0;(internIter<maxInternIter) && (eps<error);++internIter){
			//-----------------------
			error=0;
			int currentPos=0;
			for(int k=0;k<s;++k){
				for(int i=0;i<r;++i){
					for(int j=0;j<c;++j, ++currentPos){
						double &f_current=f[currentPos];
						int nCount=0;
						double fz_sum=0;
						double z_sum=0;
						for(int nn=0;nn<NUM_NEIGHBORS;++nn){
							int ii=i+dRow[nn];
							int jj=j+dCol[nn];
							int kk=k+dSlice[nn];
							if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
								int neighPos=kk*r*c+ii*c+jj;
								++nCount;
								double f_neigh=f[neighPos];
								double z_neigh=Z[currentPos*NUM_NEIGHBORS+nn];
								fz_sum+=f_neigh*z_neigh;
								z_sum+=z_neigh;
							}
						}
						double f_prev=f_current;
						f_current=(img[currentPos]*M[currentPos]+lambda*fz_sum)/(M[currentPos]+lambda*z_sum);
						error+=fabs(f_prev-f_current);
					}
				}
			}
			error/=(r*c*s);
		}
		//---solve Z for fixed f----
		//error=0;
		int currentPosition=0;
		for(int k=0;k<s;++k){
			for(int i=0;i<r;++i){
				for(int j=0;j<c;++j, ++currentPosition){
					for(int nn=0;nn<NUM_NEIGHBORS;++nn){
						int ii=i+dRow[nn];
						int jj=j+dCol[nn];
						int kk=k+dSlice[nn];
						int neighPosition=kk*r*c+ii*c+jj;
						if(neighPosition<currentPosition){
							continue;//it will be computed later ----->
						}
						if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
							double residual=f[currentPosition]-f[neighPosition];
							double newz=getOptimalWeight(residual*residual, denoisingType, params);
							//error+=fabs(Z[currentPosition*NUM_NEIGHBORS+nn]-newz);
							Z[currentPosition*NUM_NEIGHBORS+nn]=newz;
							Z[neighPosition*NUM_NEIGHBORS+invNeigh[nn]]=newz;// <-----here
						}
					}
				}
			}
		}
		//----------------------------------
		//error/=(nvoxels);
		//---solve M for fixed f----
		//error=0;
		currentPosition=0;
		for(int k=0;k<s;++k){
			for(int i=0;i<r;++i){
				for(int j=0;j<c;++j, ++currentPosition){
					double residual=f[currentPosition]-img[currentPosition];
					double newM=getOptimalWeight(residual*residual, denoisingType, params);
					//error+=fabs(M[currentPosition]-newM);
					M[currentPosition]=newM;
				}
			}
		}
	}
	delete[] Z;
	delete[] M;
	return 0;

}


int averageVolumeDenoising(double *img, int r, int c, int s, double lambda, double *params, double *f, unsigned char *mask){
	const int NUM_NEIGHBORS=26;
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	/*int cnt[3][3][3];
	memset(cnt,0,sizeof(cnt));
	for(int i=0;i<NUM_NEIGHBORS;++i){
		cnt[1+dRow[i]][1+dCol[i]][1+dSlice[i]]++;
		if(cnt[1+dRow[i]][1+dCol[i]][1+dSlice[i]]>1){
			cerr<<"!!!"<<endl;
		}
	}*/
	int nvoxels=s*r*c;
	int currentPos=0;
	for(int k=0;k<s;++k){
		for(int i=0;i<r;++i){
			for(int j=0;j<c;++j, ++currentPos){
				if((mask!=NULL) && (mask[currentPos]==0)){
					f[currentPos]=QNAN64;
					continue;
				}
				double &f_current=f[currentPos];
				int nCount=1;
				double nSum=img[currentPos];
				for(int nn=0;nn<NUM_NEIGHBORS;++nn){
					int ii=i+dRow[nn];
					int jj=j+dCol[nn];
					int kk=k+dSlice[nn];
					if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
						int neighPos=kk*r*c+ii*c+jj;
						if((mask!=NULL) && (mask[neighPos]==0)){
							continue;
						}
						double f_neigh=img[neighPos];
						++nCount;
						nSum+=f_neigh;
					}
				}
				f_current=nSum/nCount;
			}
		}
	}
	return 0;	
}

int selectiveAverageVolumeDenoising(double *img, int r, int c, int s, double *denoised, unsigned char *mask, double *neighFeature, double neighSimThr){
	const int NUM_NEIGHBORS=26;
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	/*int cnt[3][3][3];
	memset(cnt,0,sizeof(cnt));
	for(int i=0;i<NUM_NEIGHBORS;++i){
		cnt[1+dRow[i]][1+dCol[i]][1+dSlice[i]]++;
		if(cnt[1+dRow[i]][1+dCol[i]][1+dSlice[i]]>1){
			cerr<<"!!!"<<endl;
		}
	}*/
	int nvoxels=s*r*c;
	int currentPos=0;
	int dropped=0;
	int notDropped=0;
	for(int k=0;k<s;++k){
		for(int i=0;i<r;++i){
			for(int j=0;j<c;++j, ++currentPos){
				if((mask!=NULL) && (mask[currentPos]==0)){
					denoised[currentPos]=QNAN64;
					continue;
				}
				double &f_current=denoised[currentPos];
				int nCount=1;
				double nSum=img[currentPos];
				for(int nn=0;nn<NUM_NEIGHBORS;++nn){
					int ii=i+dRow[nn];
					int jj=j+dCol[nn];
					int kk=k+dSlice[nn];
					if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
						int neighPos=kk*r*c+ii*c+jj;
						if((mask!=NULL) && (mask[neighPos]==0)){
							continue;
						}
						if(fabs(neighFeature[currentPos]-neighFeature[neighPos])>neighSimThr){
							++dropped;
							continue;
						}
						++notDropped;
						double f_neigh=img[neighPos];
						++nCount;
						nSum+=f_neigh;
					}
				}
				f_current=nSum/nCount;
			}
		}
	}
	return dropped;	
}

int medianVolumeDenoising(double *img, int r, int c, int s, double lambda, double *params, double *f){
	const int NUM_NEIGHBORS=26;
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	int nvoxels=s*r*c;
	double neighVals[NUM_NEIGHBORS+1];
	int currentPos=0;
	for(int k=0;k<s;++k){
		for(int i=0;i<r;++i){
			for(int j=0;j<c;++j, ++currentPos){
				double &f_current=f[currentPos];
				neighVals[0]=img[currentPos];
				int nCount=1;
				for(int nn=0;nn<NUM_NEIGHBORS;++nn){
					int ii=i+dRow[nn];
					int jj=j+dCol[nn];
					int kk=k+dSlice[nn];
					if((0<=ii) && (ii<r) && (0<=jj) && (jj<c) && (0<=kk) && (kk<s)){
						int neighPos=kk*r*c+ii*c+jj;
						double f_neigh=img[neighPos];
						neighVals[nCount]=f_neigh;
						++nCount;
					}
				}
				sort(neighVals, neighVals+nCount);
				if(nCount%2){
					f_current=neighVals[nCount/2];
				}else{
					f_current=0.5*(neighVals[(nCount-1)/2]+neighVals[nCount/2]);
				}
			}
		}
	}
	return 0;
}


int volumeDenoising_TGV(double *img, int nrows, int ncols, int nslices, double lambda, double *params, double *f){
	double *g=img;
	double *ff=f;
	for(int s=0;s<nslices;++s, g+=(nrows*ncols), ff+=(nrows*ncols)){
		//memcpy(ff, g, sizeof(double)*nrows*ncols);
		memset(ff, 0, sizeof(double)*nrows*ncols);
		filterTGV_L2(g, nrows, ncols, lambda, params[0], params[1], params[2], params[3], params[4], ff, NULL);
	}
	return 0;
}


int robust3DVectorFieldDenoisingOR(double *img, int nrows, int ncols, int nslices, int dim, double lambda, EDenoisingType denoisingType, double *params, double *f, double *lineProc){
	const int NUM_NEIGHBORS=6;
	/*int dRow[]	={-1, 0, 1,  0,-1, 1, 1,-1, 0,  -1, 0, 1, 0,-1, 1, 1,-1,  -1, 0, 1, 0,-1, 1, 1,-1,  0};
	int dCol[]	={ 0, 1, 0, -1, 1, 1,-1,-1, 0,   0, 1, 0,-1, 1, 1,-1,-1,   0, 1, 0, -1, 1, 1,-1,-1, 0};
	int dSlice[]={-1,-1,-1, -1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1 ,1 ,1,  1};*/
	int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	int invNeigh[NUM_NEIGHBORS];
	for(int i=0;i<NUM_NEIGHBORS;++i){
		for(int j=i+1;j<NUM_NEIGHBORS;++j){
			if((dRow[i]==-dRow[j]) && (dCol[i]==-dCol[j]) && (dSlice[i]==-dSlice[j])){
				invNeigh[i]=j;
				invNeigh[j]=i;
			}
		}
	}
	int nvoxels=nslices*nrows*ncols;
	double *Z=new double[NUM_NEIGHBORS*nvoxels];//line process
	double *M=new double[nvoxels*dim];//outlier process
	//--initialize line process--
	for(int pos=NUM_NEIGHBORS*nvoxels-1;pos>=0;--pos){
		Z[pos]=1;
	}
	for(int pos=nvoxels*dim-1;pos>=0;--pos){
		M[pos]=1;
	}
	memcpy(f,img,sizeof(double)*nvoxels*dim);
	//------------------------
	int maxIter=30;
	int iter;
	double error=1;
	double eps=1e-6;
	for(iter=0;(iter<maxIter) && (eps<error);++iter){
		//----optimize for fixed Z--
		int maxInternIter=30;
		int internIter;
		error=eps+1;
		for(internIter=0;(internIter<maxInternIter) && (eps<error);++internIter){
			//-----------------------
			error=0;
			int currentPos=0;
			int currentVox=0;
			for(int k=0;k<nslices;++k){
				for(int i=0;i<nrows;++i){
					for(int j=0;j<ncols;++j, ++currentVox){
						for(int l=0;l<dim;++l, ++currentPos){
							double &f_current=f[currentPos];
							int nCount=0;
							double fz_sum=0;
							double z_sum=0;
							for(int nn=0;nn<NUM_NEIGHBORS;++nn){
								int ii=i+dRow[nn];
								int jj=j+dCol[nn];
								int kk=k+dSlice[nn];
								if((0<=ii) && (ii<nrows) && (0<=jj) && (jj<ncols) && (0<=kk) && (kk<nslices)){
									int neighPos=(kk*nrows*ncols+ii*ncols+jj)*dim+l;
									++nCount;
									double f_neigh=f[neighPos];
									double z_neigh=Z[currentVox*NUM_NEIGHBORS+nn];
									fz_sum+=f_neigh*z_neigh;
									z_sum+=z_neigh;
								}
							}
							double f_prev=f_current;
							f_current=(img[currentPos]*M[currentPos]+lambda*fz_sum)/(M[currentPos]+lambda*z_sum);
							error+=fabs(f_prev-f_current);
						}
					}
				}
			}
			error/=(nvoxels*dim);
		}
		//---solve Z for fixed f----
		//error=0;
		int currentPosition=0;
		for(int k=0;k<nslices;++k){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++currentPosition){
					for(int nn=0;nn<NUM_NEIGHBORS;++nn){
						int ii=i+dRow[nn];
						int jj=j+dCol[nn];
						int kk=k+dSlice[nn];
						int neighPosition=kk*nrows*ncols+ii*ncols+jj;
						if(neighPosition<currentPosition){
							continue;//it will be computed later ----->
						}
						if((0<=ii) && (ii<nrows) && (0<=jj) && (jj<ncols) && (0<=kk) && (kk<nslices)){
							double residual=0;
							for(int l=0;l<dim;++l){
								residual+=SQR(f[currentPosition*dim+l]-f[neighPosition*dim+l]);
							}
							double newz=getOptimalWeight(residual, denoisingType, params);
							//error+=fabs(Z[currentPosition*NUM_NEIGHBORS+nn]-newz);
							Z[currentPosition*NUM_NEIGHBORS+nn]=newz;
							Z[neighPosition*NUM_NEIGHBORS+invNeigh[nn]]=newz;// <-----here
						}
					}
				}
			}
		}
		//----------------------------------
		//error/=(nvoxels);
		//---solve M for fixed f----
		//error=0;
		currentPosition=0;
		for(int k=0;k<nslices;++k){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j){
					for(int l=0;l<dim;++l, ++currentPosition){
						double residual=f[currentPosition]-img[currentPosition];
						double newM=getOptimalWeight(residual*residual, denoisingType, params);
						//error+=fabs(M[currentPosition]-newM);
						M[currentPosition]=newM;
					}
				}
			}
		}
	}
	if(lineProc!=NULL){
		memcpy(lineProc, Z, sizeof(double)*nvoxels*NUM_NEIGHBORS);
	}
	delete[] Z;
	delete[] M;
	return 0;
}

