#include "Tensor.h"
#include "string.h"
#include "macros.h"
#include "utilities.h"
#include "geometryutils.h"
#include "linearalgebra.h"
#include "statisticsutils.h"
#include <iostream>
#include <algorithm>
#include <math.h>
using namespace std;
Tensor::Tensor(){
	setDefault();
}

Tensor::~Tensor(){
}

void Tensor::setDefault(void){
	memset(this, 0, sizeof(Tensor));
}
//--accessors--
double Tensor::getVolumeFraction(void){
	return volumeFraction;
}

double Tensor::getDiffusivity(int k){
	return diffusivities[k];
}

double *Tensor::getDiffusivities(void){
	return diffusivities;
}

double *Tensor::getRotationMatrix(void){
	return rotationMatrix;
}

void Tensor::setVolumeFraction(double vf){
	volumeFraction=vf;
}

void Tensor::setDiffusivity(int k, double diff){
	diffusivities[k]=diff;
	prodDiffusivities=diffusivities[0]*diffusivities[1]*diffusivities[2];
}

void Tensor::setDiffusivities(double lambda_min, double lambda_middle, double lambda_max){
	diffusivities[0]=lambda_min;
	diffusivities[1]=lambda_middle;
	diffusivities[2]=lambda_max;
	prodDiffusivities=diffusivities[0]*diffusivities[1]*diffusivities[2];
}

void Tensor::setDiffusivities(double *lambda){
	memcpy(diffusivities, lambda, sizeof(double)*3);
	prodDiffusivities=diffusivities[0]*diffusivities[1]*diffusivities[2];
}

void Tensor::getPDD(double *pdd)const{
	pdd[0]=rotationMatrix[2];
	pdd[1]=rotationMatrix[5];
	pdd[2]=rotationMatrix[8];
}

void Tensor::setRotationMatrixFromPDD(double *pdd){
	double e[3]={0,0,1};
	fromToRotation(e, pdd, rotationMatrix);
}

double Tensor::computeFractionalAnisotropy(void){
	double L=(diffusivities[0]+diffusivities[1]+diffusivities[2])/3.0;
	double sumSq=SQR(diffusivities[0])+SQR(diffusivities[1])+SQR(diffusivities[2]);
	double sumSq1=SQR(diffusivities[0]-L)+SQR(diffusivities[1]-L)+SQR(diffusivities[2]-L);
	double FA=sqrt(1.5*sumSq1/sumSq);
	return FA;
}

void Tensor::computeODF(double *directions, int nDirections, double *ODF){
	memset(ODF, 0, sizeof(double)*nDirections);
	double *R=rotationMatrix;
	double *lambda=diffusivities;
	double Di[9]={//Di=D^{-1}*R'
			R[0]/lambda[0], R[3]/lambda[0], R[6]/lambda[0],
			R[1]/lambda[1], R[4]/lambda[1], R[7]/lambda[1],
			R[2]/lambda[2], R[5]/lambda[2], R[8]/lambda[2]
	};
	multMatrixMatrix<double>(R,Di,3,Di);
	for(int idx=0;idx<nDirections;++idx){
		double *r=&directions[3*idx];
		double eval=evaluateQuadraticForm(Di, r, 3);
		eval=sqrt(eval);
		eval=eval*eval*eval;
		ODF[idx]+=volumeFraction/(4*M_PI*eval*sqrt(prodDiffusivities));
	}
	
	double sum=0;
	for(int i=0;i<nDirections;++i){
		sum+=ODF[i];
	}
	for(int i=0;i<nDirections;++i){
		ODF[i]/=sum;
	}
}

double Tensor::computeSignal(double *bCoord){
	double b=sqrt(SQR(bCoord[0])+SQR(bCoord[1])+SQR(bCoord[2]));
	double bCoordNormalized[3]={bCoord[0], bCoord[1], bCoord[2]};
	if(b>0){
		bCoordNormalized[0]/=b;
		bCoordNormalized[1]/=b;
		bCoordNormalized[2]/=b;
	}
	double signal=0;
	
	double *R=rotationMatrix;
	double *lambda=diffusivities;
	double Di[9]={//Di=D*R'
			R[0]*lambda[0], R[3]*lambda[0], R[6]*lambda[0],
			R[1]*lambda[1], R[4]*lambda[1], R[7]*lambda[1],
			R[2]*lambda[2], R[5]*lambda[2], R[8]*lambda[2]
	};
	multMatrixMatrix<double>(R,Di,3,Di);
	double eval=evaluateQuadraticForm(Di, bCoordNormalized, 3);
	signal+=volumeFraction*exp(-b*eval);
	
	return signal;
}

void Tensor::addNoise(double *S, int len, double sigma, double *Sn){
	if(sigma<0){
		cerr<<"Error: sigma must be >= 0"<<endl;
		return;
	}
	for(int j=0;j<len;++j){
		Sn[j]=generateRician(S[j], sigma);
	}
}

void Tensor::acquireWithScheme(double *b, double *gradList, int nDir, double sigma, double *S){
	memset(S, 0, sizeof(double)*nDir);
	for(int i=0;i<nDir;++i){
		double grad[3]={b[i]*gradList[3*i], b[i]*gradList[3*i+1], b[i]*gradList[3*i+2]};
		S[i]=computeSignal(grad);
		addNoise(S,nDir,sigma,S);
	}
}

void Tensor::acquireWithScheme(double b, double *gradList, int nDir, double sigma, double *S){
	memset(S, 0, sizeof(double)*nDir);
	for(int i=0;i<nDir;++i){
		double grad[3]={b*gradList[3*i], b*gradList[3*i+1], b*gradList[3*i+2]};
		S[i]=computeSignal(grad);
	}
	if(sigma>0){
		addNoise(S,nDir,sigma,S);
	}
}

void Tensor::rotationFromAngles(double azimuth, double zenith){
	double *M=rotationMatrix;
	M[0]=cos(azimuth)*cos(zenith);	M[1]=-sin(azimuth);	M[2]=cos(azimuth)*sin(zenith);
	M[3]=sin(azimuth)*cos(zenith);	M[4]=cos(azimuth);	M[5]=sin(azimuth)*sin(zenith);
	M[6]=-sin(zenith);				M[7]=0;				M[8]=cos(zenith);
}
		
void Tensor::loadFromTxt(FILE *F){
	fscanf(F, "%lf", &volumeFraction);
	double *R=rotationMatrix;
	double *lambda=diffusivities;
	fscanf(F, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &R[0], &R[3], &R[6], &R[1], &R[4], &R[7],&R[2], &R[5], &R[8]);
	fscanf(F, "%lf%lf%lf", &lambda[0], &lambda[1], &lambda[2]);
	prodDiffusivities=lambda[0]*lambda[1]*lambda[2];
}

void Tensor::saveToTxt(FILE *F){
	fprintf(F, "%lf", volumeFraction);
	double *R=rotationMatrix;
	double *lambda=diffusivities;
	fprintf(F, "\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf", R[0], R[3], R[6], R[1], R[4], R[7],R[2], R[5], R[8]);
	fprintf(F, "\t%lf\t%lf\t%lf", lambda[0], lambda[1], lambda[2]);
	fprintf(F, "\n");
}

void Tensor::fitFromSignal(double S0, double *S, GDTI &H){
	double tensor[6];
	double eVec[9];
	double eVal[3];
	double md=H.solve(S0,S, tensor);
	forceNonnegativeTensor(tensor,eVec, eVal);
	int maxIndex=getMaxIndex(eVal,3);
	setRotationMatrixFromPDD(&eVec[3*maxIndex]);
	sort(eVal, eVal+3);
	setDiffusivities(eVal);
	setVolumeFraction(1);
}

