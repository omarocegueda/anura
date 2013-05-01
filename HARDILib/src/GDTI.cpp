#include "GDTI.h"
#include <iostream>
#include <math.h>
#include <vector>
#include "linearalgebra.h"
#include "nifti1_io.h"
#include "utilities.h"
#include "dtiutils.h"
#include "histogram.h"
#include "bits.h"
#include <algorithm>
using namespace std;

#define GDTI_NEIGH_SIZE 26
int GDTI_dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
int GDTI_dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
int GDTI_dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};


void printMatrix(double *M, int n, int m, const char *fname){
	FILE *F=fopen(fname, "w");
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			fprintf(F,"%E\t",M[i*m+j]);
		}
		fprintf(F,"\n");
	}
	fclose(F);
}

GDTI::GDTI(int _rank, double _b, double *orientations, int _numOrientations){
	if(_rank%2>0){
		cerr<<"Invalid tensor rank "<<_rank<<endl;
		return;
	}
	b=_b;
	numGradients=_numOrientations;
	rank=_rank;
	numCoefficients=((rank+2)*(rank+1))/2;
	allocate(_numOrientations);
	initialize(b, orientations, numGradients);
	useNeighbors=false;
}

GDTI::~GDTI(){
	dellocate();
}

int GDTI::getRank(void){
	return rank;
}

int GDTI::getNumCoefficients(void){
	return numCoefficients;
}

void GDTI::allocate(int _numGradients){
	numGradients=_numGradients;
	coeffPositions=new int[numCoefficients*rank];
	coeffMultiplicities=new int[numCoefficients];
	nxsyszs=new int[numCoefficients*3];
	M=new double[numCoefficients*numGradients];
	MtMinv=new double[numCoefficients*numCoefficients];
	gradients=new double[3*_numGradients];
	
}

void GDTI::dellocate(void){
	DELETE_ARRAY(coeffPositions);
	DELETE_ARRAY(coeffMultiplicities);
	DELETE_ARRAY(nxsyszs);
	DELETE_ARRAY(M);
	DELETE_ARRAY(MtMinv);
	DELETE_ARRAY(gradients);
}
#define SET_POSITION_2(i, A, B) coeffPositions[2*(i)]=A; coeffPositions[2*(i)+1]=B;
#define SET_POSITION_4(i, A, B, C, D) coeffPositions[4*(i)]=A;	coeffPositions[4*(i)+1]=B;	coeffPositions[4*(i)+2]=C; coeffPositions[4*(i)+3]=D;
#define SET_POSITION_6(i, A, B, C, D, E, F) coeffPositions[6*(i)]=A;	coeffPositions[6*(i)+1]=B;	coeffPositions[6*(i)+2]=C; coeffPositions[6*(i)+3]=D;	coeffPositions[6*(i)+4]=D; coeffPositions[6*(i)+5]=F;
#define SET_NXSYSZS(i, A, B, C) nxsyszs[3*(i)]=A;	nxsyszs[3*(i)+1]=B;	nxsyszs[3*(i)+2]=C;
void GDTI::initialize(double b, double *orientations, int numOrientations){
	const int X=0;
	const int Y=1;
	const int Z=2;
	if(numGradients!=numOrientations){
		numGradients=numOrientations;
		DELETE_ARRAY(gradients);
		gradients=new double[numGradients];
	}
	memcpy(gradients, orientations, sizeof(double)*3*numGradients);
	switch(rank){
		case 2:
			SET_POSITION_2(0,X,X);	SET_NXSYSZS(0, 2, 0, 0);	coeffMultiplicities[0]=1;
			SET_POSITION_2(1,Y,Y);	SET_NXSYSZS(1, 0, 2, 0);	coeffMultiplicities[1]=1;
			SET_POSITION_2(2,Z,Z);	SET_NXSYSZS(2, 0, 0, 2);	coeffMultiplicities[2]=1;
			SET_POSITION_2(3,X,Y);	SET_NXSYSZS(3, 1, 1, 0);	coeffMultiplicities[3]=2;
			SET_POSITION_2(4,X,Z);	SET_NXSYSZS(4, 1, 0, 1);	coeffMultiplicities[4]=2;
			SET_POSITION_2(5,Y,Z);	SET_NXSYSZS(5, 0, 1, 1);	coeffMultiplicities[5]=2;
		break;
		case 4:
			SET_POSITION_4(0,X,X,X,X);	SET_NXSYSZS(0, 4, 0, 0);	coeffMultiplicities[0]=1;
			SET_POSITION_4(1,Y,Y,Y,Y);	SET_NXSYSZS(1, 0, 4, 0);	coeffMultiplicities[1]=1;
			SET_POSITION_4(2,Z,Z,Z,Z);	SET_NXSYSZS(2, 0, 0, 4);	coeffMultiplicities[2]=1;
			SET_POSITION_4(3,X,X,X,Y);	SET_NXSYSZS(3, 3, 1, 0);	coeffMultiplicities[3]=4;
			SET_POSITION_4(4,X,X,X,Z);	SET_NXSYSZS(4, 3, 0, 1);	coeffMultiplicities[4]=4;
			SET_POSITION_4(5,Y,Y,Y,X);	SET_NXSYSZS(5, 1, 3, 0);	coeffMultiplicities[5]=4;
			SET_POSITION_4(6,Y,Y,Y,Z);	SET_NXSYSZS(6, 0, 3, 1);	coeffMultiplicities[0]=4;
			SET_POSITION_4(7,Z,Z,Z,X);	SET_NXSYSZS(7, 1, 0, 3);	coeffMultiplicities[1]=4;
			SET_POSITION_4(8,Z,Z,Z,Y);	SET_NXSYSZS(8, 0, 1, 3);	coeffMultiplicities[2]=4;
			SET_POSITION_4(9,X,X,Y,Y);	SET_NXSYSZS(9,  2, 2, 0);	coeffMultiplicities[3]=6;
			SET_POSITION_4(10,X,X,Z,Z);	SET_NXSYSZS(10, 2, 0, 2);	coeffMultiplicities[4]=6;
			SET_POSITION_4(11,Y,Y,Z,Z);	SET_NXSYSZS(11, 0, 2, 2);	coeffMultiplicities[5]=6;
			SET_POSITION_4(12,X,X,Y,Z);	SET_NXSYSZS(12, 2, 1, 1);	coeffMultiplicities[0]=12;
			SET_POSITION_4(13,Y,Y,X,Z);	SET_NXSYSZS(13, 1, 2, 1);	coeffMultiplicities[1]=12;
			SET_POSITION_4(14,Z,Z,X,Y);	SET_NXSYSZS(14, 1, 1, 2);	coeffMultiplicities[2]=12;
		break;
		default:
			cerr<<"Tensor rank "<<rank<<" not supported."<<endl;
			return;
	}

	for(int j=0;j<numCoefficients;++j){//build i-th equation
		for(int i=0;i<numOrientations;++i){//each orientation defines an equation
			double prod=1;
			int *pos=&coeffPositions[j*rank];//positions for j-th coefficient
			for(int k=0;k<rank;++k){
				prod*=orientations[3*i+pos[k]];
			}
			M[i*numCoefficients+j]=b*coeffMultiplicities[j]*prod;
		}
	}

	for(int i=0;i<numCoefficients;++i){
		for(int j=0;j<numCoefficients;++j){
			double sum=0;
			for(int k=0;k<numOrientations;++k){
				sum+=M[k*numCoefficients+i]*M[k*numCoefficients+j];
			}
			MtMinv[i*numCoefficients+j]=sum;
		}
	}
	computeInverse(MtMinv, numCoefficients);
}

double GDTI::solve(double S0, double *Si, double *tensor){
	for(int c=0;c<numCoefficients;++c){
		double sum=0;
		for(int i=0;i<numGradients;++i){
			double diff;
			if(Si[i]<1e-8){//FIX-ME
				diff=log(S0)-log(1e-8);
			}else{
				diff=log(S0)-log(Si[i]);
			}
			sum+=M[i*numCoefficients+c]*diff;
		}
		tensor[c]=sum;
	}
	//solve for tensor
	multMatrixVector(MtMinv,tensor,numCoefficients,numCoefficients,tensor);
	//compute mean diffusivity
	double MD=0;
	switch(rank){
		case 2://compute mean diffusity , rank 2
			MD=(tensor[0]+tensor[1]+tensor[2])/3.0;
		break;
		case 4://compute mean diffusity , rank 4
			MD=(tensor[0]+tensor[1]+tensor[2] + 2 * (tensor[9]+tensor[10]+tensor[11]))*0.2;
		break;
		case 6://compute mean diffusity , rank 6
			MD=(tensor[0]+tensor[1]+tensor[2] + 3 * (tensor[9]+tensor[10]+tensor[11]+tensor[12]+tensor[13]+tensor[14]) + 6*tensor[27] )/7.0;
		break;
	}
	return MD;
}

void GDTI::solveField(double *S0, double *DW, int nslices, int nrows, int ncols, double *fittedTensors, double *eigenvalues, double *pdds){
	double *tensor=NULL;
	if(fittedTensors==NULL){
		tensor=new double[numCoefficients];
	}
	double eVal[3];
	double eVec[9];
	double *Sx=new double[numGradients];
	int nvox=nslices*nrows*ncols;
	for(int ps=0;ps<nslices;++ps){
		for(int pc=0;pc<ncols;++pc){
			for(int pr=0;pr<nrows;++pr){
				int pos=ps*(nrows*ncols)+pr*ncols+pc;
				bool valid=true;
				if(S0[pos]<1e-8){//FIX-ME
					valid=false;
				}else{
					memcpy(Sx, &DW[numGradients*pos], sizeof(double)*numGradients);
					for(int i=0;i<numGradients;++i){
						if(Sx[i]<1e-8){//FIX-ME
							valid=false;
							break;
						}
					}
				}
				if(!valid){
					if(fittedTensors!=NULL){
						tensor=&fittedTensors[pos*numCoefficients];
						for(int i=0;i<numCoefficients;++i){
							tensor[i]=SNAN64;
						}
					}
					if(pdds!=NULL){
						pdds[3*pos  ]=SNAN64;
						pdds[3*pos+1]=SNAN64;
						pdds[3*pos+2]=SNAN64;
					}
					if(eigenvalues!=NULL){
						eigenvalues[3*pos  ]=SNAN64;
						eigenvalues[3*pos+1]=SNAN64;
						eigenvalues[3*pos+2]=SNAN64;
					}
					continue;
				}
				if(fittedTensors!=NULL){
					tensor=&fittedTensors[pos*numCoefficients];
				}
				double md=solve(S0[pos],Sx, tensor);
				forceNonnegativeTensor(tensor,eVec, eVal);
				if(pdds!=NULL){
					int maxIndex=getMaxIndex(eVal,3);
					memcpy(&pdds[3*pos], &eVec[3*maxIndex], 3*sizeof(double));
				}
				sort(eVal, eVal+3);
				if(eigenvalues!=NULL){
					memcpy(&eigenvalues[3*pos], eVal, 3*sizeof(double));
				}
			}
		}
	}
	delete[] Sx;
	if(fittedTensors==NULL){
		delete[] tensor;
	}
}

void GDTI::createMask(double *S0, double *DW, int nslices, int nrows, int ncols, double *fittedTensors, double *eigenvalues, double *pdds, unsigned char *mask, double thrFA, double thrS0, double thrAvSignal){
	int nvox=nslices*nrows*ncols;
	double *eVal=eigenvalues;
	if(eigenvalues==NULL){
		eVal=new double[3*nvox];
	}
	solveField(S0, DW, nslices, nrows, ncols, fittedTensors, eVal, pdds);
	
	double *fa=new double[nvox];
	int cnt[4]={0,0,0,0};
	FILE *F_av=fopen("av.txt","w");
	for(int ps=0;ps<nslices;++ps){
		for(int pc=0;pc<ncols;++pc){
			for(int pr=0;pr<nrows;++pr){
				int pos=ps*(nrows*ncols)+pr*ncols+pc;
				mask[pos]=0;
				if(!isNumber(eVal[3*pos])){
					fa[pos]=SNAN64;
					continue;
				}
				double *dwSignal=&DW[numGradients*pos];
				double s0=S0[pos];
				double avSignal=0;
				for(int i=0;i<numGradients;++i){
					avSignal+=dwSignal[i]/s0;
				}
				avSignal/=numGradients;
				fprintf(F_av, "%0.15lf\n", avSignal);

				double *v=&eVal[3*pos];
				double linear, planar, spherical;
				int tt=computeLinearPlanarSphericalCoeffs(v, linear, planar, spherical);
				if(tt==1){
					tt=1;
				}
				if(tt>=0){
					mask[pos]|=tt;
					cnt[tt]++;
				}
				fa[pos]=computeFractionalAnisotropy(v);
				
				
				if(S0[pos]>thrS0){
					mask[pos]|=S0_BIT;
				}
				if(fa[pos]>thrFA){
					mask[pos]|=FA_BIT;
				}
				/*if((thrAvSignal<=0)||((avSignal>0.1)&&(avSignal<thrAvSignal))){
					mask[pos]|=AVSIGNAL_BIT;
				}*/
				if((thrAvSignal<=0)||(avSignal<thrAvSignal)){
					mask[pos]|=AVSIGNAL_BIT;
				}
			}
		}
	}
	fclose(F_av);
	Histogram hist(fa,nvox,100);
	double thr=hist.getLastPeak(5);
	double maxFA=0;
	for(int i=0;i<nvox;++i)if(isNumber(fa[i])){
		if(fa[i]>maxFA){
			maxFA=fa[i];
		}
	}
	//thr=thr+2.0*(maxFA-thr)/3.0;
	//thr=0.5*(maxFA+thr);
	//thr=maxFA;
	FILE *F=fopen("fa.txt","w");
	for(int i=0;i<nvox;++i)if(isNumber(fa[i])){
		fprintf(F, "%0.15lf\n", fa[i]);
		if(thr<=fa[i]){
			mask[i]|=FA_HIST_BIT;
		}
	}
	fclose(F);
	
	delete[] fa;
	if(eigenvalues==NULL){
		delete[] eVal;
	}
	
}

int GDTI::computeLinearPlanarSphericalCoeffs(double *lambda, double &linear, double &planar, double &spherical){
	double sum=lambda[0]+lambda[1]+lambda[2];
	if(fabs(sum)<EPS_GDTI){
		linear=planar=spherical=0;
		return -1;
	}
	linear=(lambda[2]-lambda[1])/sum;
	planar=2.0*(lambda[1]-lambda[0])/sum;
	spherical=3*lambda[0]/sum;
	if(linear>planar){
		if(linear>spherical){
			return 1;//linear
		}else{
			return 3;//spherical
		}
	}else if(planar>spherical){
		return 2;//planar
	}
	return 3;//spherical
}

int GDTI::computeAverageProfile(double *eigenvalues, int nvoxels, unsigned char *mask, unsigned char filterMask, unsigned char filterVal, double *averageProfile){
	int nSamples=0;
	averageProfile[0]=0;
	averageProfile[1]=0;
	averageProfile[2]=0;
	double *eval=eigenvalues;
	for(int i=0;i<nvoxels;++i, eval+=3)if((mask[i]&filterMask)==filterVal){
		averageProfile[0]+=eval[0];
		averageProfile[1]+=eval[1];
		averageProfile[2]+=eval[2];
		++nSamples;
	}
	if(nSamples==0){
		averageProfile[0]=SNAN64;
		averageProfile[1]=SNAN64;
		averageProfile[2]=SNAN64;
		return 0;
	}
	averageProfile[0]/=nSamples;
	averageProfile[1]/=nSamples;
	averageProfile[2]/=nSamples;
	return nSamples;
}

void GDTI::setUseNeighbors(bool flag){
	useNeighbors=flag;
}

double GDTI::get_b(void){
	return b;
}

double *GDTI::getGradients(void){
	return gradients;
}

int GDTI::getNumGradients(void){
	return numGradients;
}


//------------------------------------------------------------------------------------------



GDTI_output::GDTI_output(){
	mask=NULL;
	fractionalAnisotropy=NULL;
	pdd=NULL;
	meanDifusivity=NULL;
	lambdas=NULL;
}
	
GDTI_output::~GDTI_output(){
	DELETE_ARRAY(mask);
	DELETE_ARRAY(fractionalAnisotropy);
	DELETE_ARRAY(pdd);
	DELETE_ARRAY(meanDifusivity);
	DELETE_ARRAY(lambdas);
}



int fitGDTI(const GDTI_params &params, GDTI_output &output){
	double *orientations=NULL;
	int numOrientations=0;
	int *s0Indices=NULL;
	int numS0=0;
	
	int nr, nc, ns;
	double *dwVolume=NULL;
	double *s0Volume=NULL;

	loadOrientations(params.orientationsName, orientations, numOrientations, s0Indices, numS0);
	loadDWMRIFiles(params.dwNames, s0Indices, numS0, s0Volume, dwVolume, nr, nc, ns);

	
	int len=nr*nc*ns;
	//---get mask---
	unsigned char *mask=new unsigned char[len];
	getMaximumConnectedComponentMask(s0Volume, nr, nc, ns, mask);
	//--------------

	GDTI H(params.rank, params.b, orientations, numOrientations);

	double s0=0;
	double *dwSignal=new double[numOrientations];
	double *tensor=new double[H.getNumCoefficients()];

	double *PDD=new double[nr*nc*ns*3];
	double *lambdas=new double[nr*nc*ns*3];
	double *fractionalAnisotropy=new double[nr*nc*ns];
	double *meanDiffusivity=new double[nr*nc*ns];

	memset(PDD, 0, sizeof(double)*nr*nc*ns*3);
	memset(lambdas, 0, sizeof(double)*nr*nc*ns*3);
	memset(fractionalAnisotropy, 0, sizeof(double)*nr*nc*ns);
	memset(meanDiffusivity, 0, sizeof(double)*nr*nc*ns);

	double eigenValues[3];
	double eigenVectors[9];
	for(int ps=0;ps<ns;++ps){
		for(int pc=0;pc<nc;++pc){
			for(int pr=0;pr<nr;++pr){
				int pos=ps*(nr*nc)+(nr-1-pr)*nc+pc;
				double *pdd=&PDD[3*pos];
				double *evPos=&lambdas[3*pos];
				if(mask[pos]){
					getDWSignalAtVoxel(s0Volume, dwVolume, nr, nc, ns, numOrientations, pr, pc, ps, s0, dwSignal);
					double md=H.solve(s0,dwSignal, tensor);
					forceNonnegativeTensor(tensor,eigenVectors, eigenValues);
					int idx=int(getMaxIndex(eigenValues,3));
					sort(eigenValues, eigenValues+3);
					memcpy(evPos, eigenValues, sizeof(double)*3);
					memcpy(pdd, &eigenVectors[3*idx], sizeof(double)*3);
					//compute mean diffusivity and fractional anisotropy
					
					meanDiffusivity[pos]=md;
					double fa=computeFractionalAnisotropy(eigenValues, md);
					fa=MIN(fa, 1);
					fractionalAnisotropy[pos]=fa;
				}
			}
		}
	}
	delete[] orientations;
	delete[] s0Indices;
	delete[] dwVolume;
	delete[] s0Volume;
	delete[] dwSignal;
	delete[] tensor;
	output.fractionalAnisotropy=fractionalAnisotropy;
	output.lambdas=lambdas;
	output.mask=mask;
	output.meanDifusivity=meanDiffusivity;
	output.nc=nc;
	output.nr=nr;
	output.ns=ns;
	output.pdd=PDD;
	return 0;
}

