#ifdef USE_QT
#include <QtGui>
#include <QGLWidget>
#include <GL/glu.h>
#endif
#include "MultiTensor.h"
#include <string.h>
#include <macros.h>
#include <iostream>
#include "statisticsutils.h"
#include "geometryutils.h"
#include "GDTI.h"
#include "DBF.h"
#include "dtiutils.h"
#include "nnls.h"
#include "lars.h"
#include "cv.h"
#include "highgui.h"
#include "Tensor.h"
#include <algorithm>
#include <vector>
#include "nnsls_pgs.h"
#include "clustering.h"
#include "expm.h"
#include "bfgs.h"
using namespace std;
#define RUN_ITERATIVE
#define MULTITENSOR_VERSION 4
//#define ITERATIVE_VISUAL_DEBUG
MultiTensor::MultiTensor(){
	initNull();
}

MultiTensor::MultiTensor(int nCompartments){
	if(nCompartments<=0){
		initNull();
		return;
	}
	allocate(nCompartments);
	for(int i=0;i<numCompartments;++i){
		double *lambda=&diffusivities[3*i];
		volumeFractions[0]=1.0/numCompartments;
		lambda[0]=0.3e-3;
		lambda[1]=0.3e-3;
		lambda[2]=1.7e-3;
		/*lambda[0]=0.000000001393979;
		lambda[1]=0.000000001393979;
		lambda[2]=0.000000001974300;*/
		prodDiffusivities[i]=lambda[0]*lambda[1]*lambda[2];
	}
	rotationFromAngles(0, M_PI_2, &rotationMatrices[0]);
	if(nCompartments>1){
		rotationFromAngles(M_PI_2, M_PI_2, &rotationMatrices[9]);
	}
	selected=false;
}

double *MultiTensor::getAlpha(void){
	return alpha;
}

double MultiTensor::getMaxAlpha(void){
	if(nAlpha<=0){
		return -1;
	}
	int sel=0;
	for(int i=1;i<nAlpha;++i){
		if(alpha[sel]<alpha[i]){
			sel=i;
		}
	}
	return alpha[sel];
}

int MultiTensor::getMaxAlphaIndex(void){
	if(nAlpha<=0){
		return -1;
	}
	int sel=0;
	for(int i=1;i<nAlpha;++i){
		if(alpha[sel]<alpha[i]){
			sel=i;
		}
	}
	return sel;
}

int *MultiTensor::getGroups(void){
	return groups;
}

void MultiTensor::setGroups(vector<set<int> > &vsGroups){
	if(nAlpha==0){
		return;
	}
	if(groups==NULL){
		groups=new int[nAlpha];
	}
	memset(groups, -1, sizeof(int)*nAlpha);
	for(unsigned i=0;i<vsGroups.size();++i){
		for(set<int>::iterator it=vsGroups[i].begin();it!=vsGroups[i].end();++it){
			groups[*it]=i;
		}
	}
}

void MultiTensor::setGroups(std::vector<std::set<std::pair<double, int> > > &vsGroups){
	if(nAlpha==0){
		return;
	}
	if(groups==NULL){
		groups=new int[nAlpha];
	}
	memset(groups, -1, sizeof(int)*nAlpha);
	for(unsigned i=0;i<vsGroups.size();++i){
		for(set<pair<double, int> >::iterator it=vsGroups[i].begin();it!=vsGroups[i].end();++it){
			groups[it->second]=i;
		}
	}
}

void MultiTensor::setGroups(int *_groups, int n){
	if(groups==NULL){
		groups=new int[n];
	}
	memcpy(groups, _groups, sizeof(int)*n);
}

int MultiTensor::getNumAlpha(void){
	return nAlpha;
}

void MultiTensor::setSortedCoefficientsAndDirections(double *_alpha, double *_directions, int _nAlpha){
	vector<pair<double, int> > v;
	for(int i=0;i<_nAlpha;++i)if(_alpha[i]>0){
		v.push_back(make_pair(_alpha[i], i));
		
	}
	if(directions!=NULL){
		DELETE_ARRAY(directions);
	}
	if(alpha!=NULL){
		DELETE_ARRAY(alpha);
	}
	sort(v.rbegin(), v.rend());
	nAlpha=v.size();
	directions=new double[3*nAlpha];
	alpha=new double[nAlpha];
	for(int i=0;i<nAlpha;++i){
		alpha[i]=v[i].first;
		memcpy(&directions[3*i], &_directions[3*v[i].second], sizeof(double)*3);
	}
}

void MultiTensor::setAlpha(double *_alpha, int _nAlpha){
	if(_alpha==NULL){
		alpha=NULL;
		nAlpha=0;
		return;
	}
	if(nAlpha!=_nAlpha){
		DELETE_ARRAY(alpha);
		alpha=new double[_nAlpha];
	}
	nAlpha=_nAlpha;
	memcpy(alpha, _alpha, sizeof(double)*nAlpha);
}

void MultiTensor::setDirections(double *_directions, int _nDirections){
	DELETE_ARRAY(directions);
	directions=new double[_nDirections*3];
	memcpy(directions, _directions, sizeof(double)*_nDirections*3);
}

double *MultiTensor::getDirections(void){
	return directions;
}

void MultiTensor::copyComponentFrom(MultiTensor &M, int from, int to){
	double pdd[3];
	M.getPDD(from,pdd);
	double *lambda=M.getDiffusivities(from);
	double vf=M.getVolumeFraction(from);

	this->setRotationMatrixFromPDD(to, pdd);
	this->setDiffusivities(to,lambda);
	this->setVolumeFraction(to,vf);
}

void MultiTensor::copyFrom(MultiTensor &M){
	dellocate();
	if(M.getNumCompartments()==0){
		return;
	}
	allocate(M.getNumCompartments());
	memcpy(volumeFractions, M.getVolumeFractions(), sizeof(double)*numCompartments);
	memcpy(rotationMatrices, M.getRotationMatrices(), sizeof(double)*numCompartments*9);
	memcpy(diffusivities, M.getDiffusivities(), sizeof(double)*numCompartments*3);
	memcpy(prodDiffusivities, M.getProdDiffusivities(), sizeof(double)*numCompartments);
	nAlpha=M.getNumAlpha();
	DELETE_ARRAY(alpha);
	if(M.getNumAlpha()>0){
		alpha=new double[nAlpha];
		memcpy(alpha, M.getAlpha(), sizeof(double)*nAlpha);
		if(M.getDirections()!=NULL){
			setDirections(M.getDirections(), nAlpha);
		}
		if(M.getGroups()!=NULL){
			setGroups(M.getGroups(), nAlpha);
		}
	}
}

void MultiTensor::createFromAlphaProfile(int _nAlpha, double *_alpha, double *_directions, double *_diffProfile){
	nAlpha=_nAlpha;
	vector<pair<double, int> > v;
	for(int i=0;i<_nAlpha;++i)if(_alpha[i]>0){
		v.push_back(make_pair(_alpha[i], i));
	}
	sort(v.rbegin(), v.rend());
	dellocate();
	allocate(v.size());
	nAlpha=v.size();
	alpha=new double[nAlpha];
	directions=new double[3*nAlpha];
	for(int i=0;i<nAlpha;++i){
		alpha[i]=v[i].first;
		memcpy(&directions[3*i], &_directions[3*v[i].second], sizeof(double)*3);
		setRotationMatrixFromPDD(i,&directions[3*i]);
		setDiffusivities(i,_diffProfile);
	}
	setVolumeFractions(alpha);
}

void MultiTensor::initDefault(void){
	allocate(2);
	volumeFractions[0]=0.5;
	volumeFractions[1]=0.5;
	diffusivities[0]=0.3e-3;
	diffusivities[1]=0.3e-3;
	diffusivities[2]=1.7e-3;
	diffusivities[3]=0.3e-3;
	diffusivities[4]=0.3e-3;
	diffusivities[5]=1.7e-3;
	for(int i=0;i<numCompartments;++i){
		double *lambda=&diffusivities[3*i];
		prodDiffusivities[i]=lambda[0]*lambda[1]*lambda[2];
	}
	rotationFromAngles(0, M_PI_2, &rotationMatrices[0]);
	rotationFromAngles(M_PI_2, M_PI_2, &rotationMatrices[9]);
	alpha=NULL;
	nAlpha=0;
	selected=false;
}

void MultiTensor::initNull(void){
	numCompartments=0;
	volumeFractions=NULL;
	diffusivities=NULL;
	prodDiffusivities=NULL;
	rotationMatrices=NULL;
	alpha=NULL;
	directions=NULL;
	nAlpha=0;
	groups=NULL;
	selected=false;
	compartmentSegmentation=NULL;
}

void MultiTensor::allocate(int nCompartments){
	if(nCompartments==0){
		dellocate();
		numCompartments=0;
		return;
	}
	nAlpha=0;
	alpha=NULL;
	directions=NULL;
	groups=NULL;
	numCompartments=nCompartments;
	volumeFractions=new double[numCompartments];
	compartmentSegmentation=new int[numCompartments];
	memset(compartmentSegmentation, 0, sizeof(int)*numCompartments);
	prodDiffusivities=new double[numCompartments];
	diffusivities=new double[3*numCompartments];
	rotationMatrices=new double[9*numCompartments];
}

void MultiTensor::dellocate(void){
	DELETE_ARRAY(volumeFractions);
	DELETE_ARRAY(compartmentSegmentation);
	DELETE_ARRAY(prodDiffusivities);
	DELETE_ARRAY(diffusivities);
	DELETE_ARRAY(rotationMatrices);
	DELETE_ARRAY(alpha);
	DELETE_ARRAY(directions);
	DELETE_ARRAY(groups);
	nAlpha=0;
	numCompartments=0;
}

void MultiTensor::dropSmallPeaks(double prop){
	if(numCompartments==0){
		return;
	}
	int maxCoef=getMaxIndex(volumeFractions, numCompartments);
	double thr=volumeFractions[maxCoef]*prop;
	int pos=0;
	double sumFractions=0;
	for(int i=0;i<numCompartments;++i){
		if(thr<=volumeFractions[i]){
			sumFractions+=volumeFractions[i];
			if(pos<i){
				volumeFractions[pos]=volumeFractions[i];
				compartmentSegmentation[pos]=compartmentSegmentation[i];
				memcpy(&rotationMatrices[9*pos], &rotationMatrices[9*i], sizeof(double)*9);
				memcpy(&diffusivities[3*pos], &diffusivities[3*i], sizeof(double)*3);
				prodDiffusivities[pos]=prodDiffusivities[i];
			}
			++pos;
		}else if(groups!=NULL){
			for(int j=0;j<nAlpha;++j){
				if(groups[j]==i){
					groups[j]=-1;
				}
			}
		}
	}
	numCompartments=pos;
	for(int i=0;i<numCompartments;++i){
		volumeFractions[i]/=sumFractions;
	}
}

//==============accessors================
void MultiTensor::setVolumeFractions(double *vf){
	memcpy(volumeFractions, vf, sizeof(double)*numCompartments);
}

void MultiTensor::setVolumeFraction(int k, double vf){
	volumeFractions[k]=vf;
}

void MultiTensor::setCompartmentSegmentation(int k, int lab){
	compartmentSegmentation[k]=lab;
}

void MultiTensor::setCompartmentSegmentation(int *seg){
	memcpy(compartmentSegmentation, seg, sizeof(int)*numCompartments);
}

void MultiTensor::setRotationMatrix(int k, double *R){
	if(k>=numCompartments){
		return;
	}
	memcpy(&rotationMatrices[9*k], R, sizeof(double)*9);
}

void MultiTensor::setRotationMatrix(int k, double azimuth, double zenith){
	if((0<=k) && (k<numCompartments)){
		rotationFromAngles(azimuth, zenith, &rotationMatrices[9*k]);
	}
}

void MultiTensor::setRotationMatrixFromPDD(int k, double *pdd){
	double e[3]={0,0,1};
	if((0<=k) && (k<numCompartments)){
		fromToRotation(e, pdd, &rotationMatrices[9*k]);
	}
}

void MultiTensor::setDiffusivities(int k, double lambdaMin, double lambdaMid, double lambdaMax){
	double *lambda=&diffusivities[3*k];
	lambda[0]=lambdaMin;
	lambda[1]=lambdaMid;
	lambda[2]=lambdaMax;
	prodDiffusivities[k]=lambdaMin*lambdaMid*lambdaMax;
}

void MultiTensor::setDiffusivities(int k, double *_lambda){
	double *lambda=&diffusivities[3*k];
	memcpy(lambda, _lambda, sizeof(double)*3);
	prodDiffusivities[k]=_lambda[0]*_lambda[1]*_lambda[2];
}

void MultiTensor::setDiffusivities(double *_lambda){
	for(int k=0;k<numCompartments;++k){
		double *lambda=&diffusivities[3*k];
		memcpy(lambda, _lambda, sizeof(double)*3);
		prodDiffusivities[k]=_lambda[0]*_lambda[1]*_lambda[2];
	}
	
}
//=======================================


//Compute the "TRUE ODF" corresponding to this model
void MultiTensor::computeODF(double *directions, int nDirections, double *ODF)const{
	memset(ODF, 0, sizeof(double)*nDirections);
	
	for(int i=0;i<numCompartments;++i){
		double *R=&rotationMatrices[i*9];
		double *lambda=&diffusivities[i*3];
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
			ODF[idx]+=volumeFractions[i]/(4*M_PI*eval*sqrt(prodDiffusivities[i]));
		}
	}
	double sum=0;
	for(int i=0;i<nDirections;++i){
		sum+=ODF[i];
	}
	for(int i=0;i<nDirections;++i){
		ODF[i]/=sum;
	}
}

//estimate the FA for each fiber compartment
void MultiTensor::computeFractionalAnisotropy(double *FA){
	for(int i=0;i<numCompartments;++i){
		double *tmp=&diffusivities[3*i];
		double L=(tmp[0]+tmp[1]+tmp[2])/3.0;
		double sumSq=SQR(tmp[0])+SQR(tmp[1])+SQR(tmp[2]);
		double sumSq1=SQR(tmp[0]-L)+SQR(tmp[1]-L)+SQR(tmp[2]-L);
		FA[i]=sqrt(1.5*sumSq1/sumSq);
	}
}
MultiTensor::~MultiTensor(){
	dellocate();
}
int MultiTensor::getNumCompartments(void)const{
	return numCompartments;
}

double MultiTensor::getVolumeFraction(int k)const{
	return volumeFractions[k];
}

double MultiTensor::getMaxVolumeFraction(void){
	int sel=0;
	for(int i=1;i<numCompartments;++i){
		if(volumeFractions[sel]<volumeFractions[i]){
			sel=i;
		}
	}
	return volumeFractions[sel];
}

int MultiTensor::getMaxVolumeFractionIndex(void){
	int sel=0;
	for(int i=1;i<numCompartments;++i){
		if(volumeFractions[sel]<volumeFractions[i]){
			sel=i;
		}
	}
	return sel;
}


double *MultiTensor::getVolumeFractions(void){
	return volumeFractions;
}
double *MultiTensor::getRotationMatrices(void){
	return rotationMatrices;
}
int *MultiTensor::getCompartmentSegmentation(void){
	return compartmentSegmentation;
}

int MultiTensor::getCompartmentSegmentation(int k){
	return compartmentSegmentation[k];
}

double *MultiTensor::getDiffusivities(void){
	return diffusivities;
}

double *MultiTensor::getDiffusivities(int k){
	return &diffusivities[3*k];
}

double *MultiTensor::getProdDiffusivities(void){
	return prodDiffusivities;
}

//Probe the signal at a given q-space coordinate
double MultiTensor::computeSignal(double *bCoord){
	double b=sqrt(SQR(bCoord[0])+SQR(bCoord[1])+SQR(bCoord[2]));
	double bCoordNormalized[3]={bCoord[0], bCoord[1], bCoord[2]};
	if(b>0){
		bCoordNormalized[0]/=b;
		bCoordNormalized[1]/=b;
		bCoordNormalized[2]/=b;
	}
	double signal=0;
	for(int i=0;i<numCompartments;++i){
		double *R=&rotationMatrices[i*9];
		double *lambda=&diffusivities[i*3];
		double Di[9]={//Di=D*R'
			R[0]*lambda[0], R[3]*lambda[0], R[6]*lambda[0],
			R[1]*lambda[1], R[4]*lambda[1], R[7]*lambda[1],
			R[2]*lambda[2], R[5]*lambda[2], R[8]*lambda[2]
		};
		multMatrixMatrix<double>(R,Di,3,Di);
		double eval=evaluateQuadraticForm(Di, bCoordNormalized, 3);
		signal+=volumeFractions[i]*exp(-b*eval);
	}
	return signal;
}

//Add Rician noise to the signal
void MultiTensor::addNoise(double *S, int len, double sigma, double *Sn){
	if(sigma<0){
		cerr<<"Error: sigma must be >= 0"<<endl;
		return;
	}
	for(int j=0;j<len;++j){
		Sn[j]=generateRician(S[j], sigma);
	}
}


//Probe the signal at several q-space positions
void MultiTensor::acquireWithScheme(double *b, double *gradList, int nDir, double sigma, double *S){
	memset(S, 0, sizeof(double)*nDir);
	for(int i=0;i<nDir;++i){
		double grad[3]={b[i]*gradList[3*i], b[i]*gradList[3*i+1], b[i]*gradList[3*i+2]};
		S[i]=computeSignal(grad);
		addNoise(S,nDir,sigma,S);
	}
}

//Probe the signal at several q-space positions
void MultiTensor::acquireWithScheme(double b, double *gradList, int nDir, double sigma, double *S){
	memset(S, 0, sizeof(double)*nDir);
	for(int i=0;i<nDir;++i){
		double grad[3]={b*gradList[3*i], b*gradList[3*i+1], b*gradList[3*i+2]};
		S[i]=computeSignal(grad);
	}
	if(sigma>0){
		addNoise(S,nDir,sigma,S);
	}
	
}

/**Compute a rotation matrix corresponding to the orientation (azimuth,zenith)
*	azimuth (phi):	angle in the x-y plane
*	zenith  (theta):	angle from z axis
*/
void MultiTensor::rotationFromAngles(double azimuth, double zenith, double *M){
	M[0]=cos(azimuth)*cos(zenith);	M[1]=-sin(azimuth);	M[2]=cos(azimuth)*sin(zenith);
	M[3]=sin(azimuth)*cos(zenith);	M[4]=cos(azimuth);	M[5]=sin(azimuth)*sin(zenith);
	M[6]=-sin(zenith);				M[7]=0;				M[8]=cos(zenith);
}

void MultiTensor::setSelected(bool b){
	selected=b;
}

bool MultiTensor::isSelected(void){
	return selected;
}

const double MultiTensor::getMinAngleDegrees(void)const{
	if(numCompartments<=1){
		return 180;
	}
	double pddi[3];
	double pddj[3];
	double minAngle=1e10;
	for(int i=0;i<numCompartments;++i){
		getPDD(i,pddi);
		for(int j=i+1;j<numCompartments;++j){
			getPDD(j,pddj);
			double opc=getAbsAngleDegrees(pddi, pddj,3);
			minAngle=MIN(minAngle,opc);
		}
	}
	return minAngle;
}

const double MultiTensor::getMaxIntraAngleDegrees(void)const{
	if(groups==NULL){
		return -1;//error
	}
	set<int> L;
	for(int i=0;i<nAlpha;++i){
		L.insert(groups[i]);
	}
	double retVal=0;
	for(set<int>::iterator it=L.begin();it!=L.end();++it){
		for(int i=0;i<nAlpha;++i)if(groups[i]==*it){
			for(int j=i+1;j<nAlpha;++j)if(groups[j]==*it){
				double opc=getAbsAngleDegrees(&directions[3*i], &directions[3*j],3);
				retVal=MAX(retVal, opc);
			}
		}
	}
	return retVal;
}

void MultiTensor::loadFromTxt(FILE *F){
	fscanf(F, "%d", &numCompartments);
	if(numCompartments!=-4){
		numCompartments=numCompartments;
	}
	int version=0;
	if(numCompartments<0){
		version=-numCompartments;
		fscanf(F, "%d", &numCompartments);
	}
	allocate(numCompartments);
	for(int i=0;i<numCompartments;++i){
		fscanf(F, "%lf", &volumeFractions[i]);
	}
	for(int i=0;i<numCompartments;++i){
		double *R=&rotationMatrices[9*i];
		fscanf(F, "%lf%lf%lf%lf%lf%lf%lf%lf%lf", &R[0], &R[3], &R[6], &R[1], &R[4], &R[7],&R[2], &R[5], &R[8]);
	}
	for(int i=0;i<numCompartments;++i){
		double *lambda=&diffusivities[3*i];
		fscanf(F, "%lf%lf%lf", &lambda[0], &lambda[1], &lambda[2]);
		/*lambda[0]=1e-9;
		lambda[1]=1e-9;
		lambda[2]=5e-9;*/
		prodDiffusivities[i]=lambda[0]*lambda[1]*lambda[2];
	}
	if(version>0){
		fscanf(F,"%d",&nAlpha);
		if(nAlpha>0){
			alpha=new double[nAlpha];
			for(int i=0;i<nAlpha;++i){
				fscanf(F, "%lf", &alpha[i]);
			}
			if(version>1){
				directions=new double[3*nAlpha];
				for(int i=0;i<nAlpha;++i){
					fscanf(F, "%lf%lf%lf", &directions[3*i], &directions[3*i+1], &directions[3*i+2]);
				}
			}
			if(version>2){
				groups=new int[nAlpha];
				for(int i=0;i<nAlpha;++i){
					fscanf(F, "%d", &groups[i]);
				}
			}
		}
			
		
		if(version>3){
			if(numCompartments>0){
				compartmentSegmentation=new int[numCompartments];
			}
			for(int i=0;i<numCompartments;++i){
				fscanf(F, "%d", &compartmentSegmentation[i]);
			}
		}
	}

}


void MultiTensor::saveToTxt(FILE *F){
	fprintf(F, "%d", -MULTITENSOR_VERSION);//version 
	fprintf(F, "\t%d", numCompartments);
	for(int i=0;i<numCompartments;++i){
		fprintf(F, "\t%0.15lf", volumeFractions[i]);
	}
	fprintf(F, "\n");
	for(int i=0;i<numCompartments;++i){
		double *R=&rotationMatrices[9*i];
		fprintf(F, "\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf\t%0.15lf", R[0], R[3], R[6], R[1], R[4], R[7],R[2], R[5], R[8]);
	}
	fprintf(F, "\n");
	for(int i=0;i<numCompartments;++i){
		double *lambda=&diffusivities[3*i];
		fprintf(F, "\t%0.15lf\t%0.15lf\t%0.15lf\n", lambda[0], lambda[1], lambda[2]);
	}
	fprintf(F,"%d",nAlpha);
	for(int i=0;i<nAlpha;++i){
		fprintf(F,"\t%0.15lf",alpha[i]);
	}
	for(int i=0;i<nAlpha;++i){
		if(directions==NULL){
			fprintf(F,"\t%0.15lf\t%0.15lf\t%0.15lf",0, 0, 0);
		}else{
			fprintf(F,"\t%0.15lf\t%0.15lf\t%0.15lf",directions[3*i], directions[3*i+1], directions[3*i+2]);
		}
		
	}
	fprintf(F, "\n");
	for(int i=0;i<nAlpha;++i){
		if(groups==NULL){
			fprintf(F,"\t-1");
		}else{
			fprintf(F,"\t%d",groups[i]);
		}
	}
	fprintf(F, "\n");
	for(int i=0;i<numCompartments;++i){
		if(compartmentSegmentation==NULL){
			fprintf(F,"\t-1");
		}else{
			fprintf(F,"\t%d",compartmentSegmentation[i]);
		}
	}
	fprintf(F, "\n");
}

void MultiTensor::saveToBinary(FILE *F){
	int ver=-MULTITENSOR_VERSION;
	fwrite(&ver, sizeof(int), 1, F);
	fwrite(&numCompartments, sizeof(int), 1, F);
	fwrite(volumeFractions, sizeof(double), numCompartments, F);
	fwrite(rotationMatrices, sizeof(double), 9*numCompartments, F);
	fwrite(diffusivities, sizeof(double), 3*numCompartments, F);
	
	fwrite(&nAlpha, sizeof(int), 1, F);
	fwrite(alpha, sizeof(double), nAlpha, F);
	fwrite(directions, sizeof(double), 3*nAlpha, F);
	if(groups==NULL){
		int flag=-1;
		for(int i=0;i<nAlpha;++i){
			fwrite(&flag, sizeof(int), 1, F);
		}
	}else{
		fwrite(groups, sizeof(int), nAlpha, F);
	}
	if(compartmentSegmentation==NULL){
		int flag=-1;
		for(int i=0;i<numCompartments;++i){
			fwrite(&flag, sizeof(int), 1, F);
		}
	}else{
		fwrite(compartmentSegmentation, sizeof(int), numCompartments, F);
	}
}

void MultiTensor::loadFromBinary(FILE *F){
	fread(&numCompartments, sizeof(int), 1, F);
	int version=0;
	if(numCompartments<0){
		version=-numCompartments;
		fread(&numCompartments, sizeof(int), 1, F);
	}
	allocate(numCompartments);
	fread(volumeFractions, sizeof(double), numCompartments, F);
	fread(rotationMatrices, sizeof(double), 9*numCompartments, F);
	fread(diffusivities, sizeof(double), 3*numCompartments, F);
	for(int i=0;i<numCompartments;++i){
		double *lambda=&diffusivities[3*i];
		/*lambda[0]=1e-9;
		lambda[1]=1e-9;
		lambda[2]=5e-9;*/
		prodDiffusivities[i]=lambda[0]*lambda[1]*lambda[2];
	}
	if(version>0){
		fread(&nAlpha, sizeof(int), 1, F);
		alpha=new double[nAlpha];
		fread(alpha, sizeof(double), nAlpha, F);
		if(version>1){
			directions=new double[3*nAlpha];
			fread(directions, sizeof(double), 3*nAlpha, F);
		}
		if(version>2){
			groups=new int[nAlpha];
			fread(groups, sizeof(int), nAlpha, F);
		}
		if(version>3){
			compartmentSegmentation=new int[numCompartments];
			fread(compartmentSegmentation, sizeof(int), numCompartments, F);
		}
	}
}

void MultiTensor::getPDD(int k, double *pdd)const{
	if((k<0) || (k>=numCompartments)){
		return;
	}
	double *M=&rotationMatrices[9*k];
	//pick last column
	pdd[0]=M[2];
	pdd[1]=M[5];
	pdd[2]=M[8];
}

void MultiTensor::getPDDs(double *pdd)const{
	for(int i=0;i<numCompartments;++i){
		getPDD(i, &pdd[3*i]);
	}
}

int checkDirections(int pos, int label, double refAngle, double *alpha, double *DBFDirections, int numDirections, int *labels){
	if(labels[pos]>=0){
		return 0;
	}
	labels[pos]=label;
	int checked=1;
	for(int i=0;i<numDirections;++i)if((labels[i]<0) && (alpha[i]>0)){
		double angle=getAbsAngleDegrees(&DBFDirections[3*pos], &DBFDirections[3*i], 3);
		if(angle<refAngle){
			checked+=checkDirections(i,label,refAngle,alpha,DBFDirections, numDirections, labels);
		}
	}
	return checked;
}

void MultiTensor::split_angle_threshold(double angle, GDTI &H, double *DBFDirections, double *DBFunctions, double transDiffusion, double longDiffusion){
	if(alpha==NULL){
		return;
	}
	int *labels=new int[nAlpha];
	memset(labels, -1, sizeof(int)*nAlpha);
	int numGroups=0;
	for(int i=0;i<nAlpha;++i)if(alpha[i]>0){
		int checked=checkDirections(i, numGroups, angle, alpha, DBFDirections, nAlpha, labels);
		if(checked>0){
			++numGroups;
		}
	}
	if(numGroups>3){
		numGroups=3;
	}
	vector<set<pair<double, int> > > groups(numGroups);
	double *sumCoeff=new double[numGroups];
	memset(sumCoeff, 0, sizeof(double)*numGroups);
	double sumAlpha=0;
	for(int i=0;i<nAlpha;++i)if((alpha[i]>0)&&(labels[i]<numGroups)){
		groups[labels[i]].insert(make_pair(alpha[i], i));
		sumCoeff[labels[i]]+=alpha[i];
		sumAlpha+=alpha[i];
	}
	double lim=sumAlpha/numGroups;
	double *alpha_backup=alpha;
	int nAlpha_backup=nAlpha;
	alpha=NULL;
	nAlpha=0;
	allocate(numGroups);
	alpha=alpha_backup;
	nAlpha=nAlpha_backup;


	int numGradients=H.getNumGradients();
	double *synthetic=new double[numGradients];
	for(int i=0;i<numGroups;++i){
		memset(synthetic, 0, sizeof(double)*numGradients);
		for(set<pair<double, int> >::iterator it=groups[i].begin();it!=groups[i].end();++it){
			for(int j=0;j<numGradients;++j){
				synthetic[j]+=(it->first/sumCoeff[i])*DBFunctions[j*nAlpha+(it->second)];
			}
		}
		MultiTensor tensor;
		createMultiTensorSingleComponent(synthetic, H, DBFDirections, groups[i], tensor);
		tensor.setDiffusivities(0,transDiffusion, transDiffusion, longDiffusion);
		setTensorAt(i,tensor);
		//setVolumeFraction(i,lim);
		setVolumeFraction(i,sumCoeff[i]);
	}

	delete[] synthetic;
	delete[] sumCoeff;
	delete[] labels;
}

void MultiTensor::group_exponential(int k, GDTI &H, double *DBFDirections, double *DBFunctions, double transDiffusion, double longDiffusion){
	if(alpha==NULL){
		return;
	}
	if(nAlpha<k){
		k=nAlpha;
	}
	double *simmatrix=new double[nAlpha*nAlpha];
	double *exp_simmatrix=new double[nAlpha*nAlpha];
	int pos=0;
	for(int i=0;i<nAlpha;++i){
		for(int j=0;j<nAlpha;++j,++pos){
			double prod=dotProduct(&DBFDirections[3*i], &DBFDirections[3*j], 3);
			simmatrix[pos]=fabs(prod);
		}
	}
	//normalizeDinamicRange(simmatrix, nAlpha*nAlpha);
	for(int i=0;i<nAlpha;++i){
		simmatrix[i*(nAlpha+1)]=0;
	}
	saveMatrix(simmatrix, nAlpha,nAlpha,"simmatrix.txt");
	expm(simmatrix, nAlpha, exp_simmatrix);
	saveMatrix(exp_simmatrix, nAlpha,nAlpha,"exp_simmatrix.txt");
	for(int i=0;i<nAlpha;++i){
		exp_simmatrix[i*(nAlpha+1)]-=1.0;
	}
	for(int i=nAlpha*nAlpha-1;i>=0;--i){
		exp_simmatrix[i]-=simmatrix[i];
	}
	const int maxIter=10;
	double *means=new double[nAlpha*k];
	double *probs=new double[nAlpha*k];
	fuzzyKMeans(exp_simmatrix, nAlpha, nAlpha, k, 1.0, maxIter, means, probs, true);
	double coeff[5]={0,0,0,0,0};
	int numGradients=H.getNumGradients();
	double *combinedSignals=new double[numGradients*k];
	double *meanDirs=new double[3*k];
	memset(combinedSignals, 0, sizeof(double)*numGradients*k);
	memset(meanDirs, 0, sizeof(double)*3*k);
		for(int i=0;i<nAlpha;++i){
			int label=-1;
			for(int c=0;c<k;++c){
				if(probs[k*i+c]>0){
					label=c;
					break;
				}
			}
			coeff[label]+=alpha[i];
			for(int j=0;j<numGradients;++j){
				combinedSignals[label*numGradients+j]+=alpha[i]*DBFunctions[i+j*nAlpha];
			}
			for(int j=0;j<3;++j){
				meanDirs[3*label+j]+=alpha[i]*DBFDirections[3*i+j];
			}
		}
		int clustersFound=0;
		for(int i=0;i<k;++i)if(coeff[i]>0){
			++clustersFound;
			for(int j=0;j<numGradients;++j){
				combinedSignals[i*numGradients+j]/=coeff[i];
			}
			for(int j=0;j<3;++j){
				meanDirs[3*i+j]/=coeff[i];
			}
		}
		double *tmpAlpha=alpha;
		int tmpNAlpha=nAlpha;
		setAlpha(NULL,0);
		allocate(clustersFound);
		setAlpha(tmpAlpha, tmpNAlpha);
		clustersFound=0;
		for(int c=0;c<k;++c)if(coeff[c]>0){
			//------fit best rank-2 tensor-----
			/*Tensor T;
			T.fitFromSignal(1,&combinedSignals[numGradients*c],H);
			setTensorAt(clustersFound,T);
			setDiffusivities(clustersFound, transDiffusion, transDiffusion, longDiffusion);
			*/
			//------use the simple mean direction--------
			setRotationMatrixFromPDD(clustersFound, &meanDirs[3*c]);
			setDiffusivities(clustersFound, transDiffusion, transDiffusion, longDiffusion);
			setVolumeFraction(clustersFound,coeff[c]);
			++clustersFound;
		}
	delete[] means;
	delete[] probs;
	delete[] simmatrix;
	delete[] exp_simmatrix;
}

void MultiTensor::split(int k, GDTI &H, double *DBFDirections, double *DBFunctions, double transDiffusion, double longDiffusion){
	if(alpha==NULL){
		return;
	}
	double sumAlpha=0;
	int positiveCount=0;
	for(int i=0;i<nAlpha;++i)if(alpha[i]>0){
		++positiveCount;
		sumAlpha+=alpha[i];
	}
	if(positiveCount<k){
		k=positiveCount;
	}

	vector<set<pair<double, int> > > groups(k);
	double *sumCoeff=new double[k];
	memset(sumCoeff, 0, sizeof(double)*k);
	for(int j=0;j<k;++j){
		sumCoeff[j]=alpha[j];
		groups[j].insert(make_pair(alpha[j], j));
	}
	double lim=sumAlpha/k;

	if(k<positiveCount){
		for(int i=k;i<nAlpha;++i)if(alpha[i]>0){
			int best=-1;
			int secondBest=-1;
			double minAngle=0;
			double secondMinAngle=0;
			for(int j=0;j<k;++j)if(sumCoeff[j]<lim){
				double opc=getAbsAngleDegrees(&DBFDirections[3*i], &DBFDirections[3*j], 3);
				if((best<0) || (opc<minAngle)){
					secondMinAngle=minAngle;
					secondBest=best;
					minAngle=opc;
					best=j;
				}
			}
			if((sumCoeff[best]+alpha[i]<=lim) || (secondBest<0)){
				groups[best].insert(make_pair(alpha[i], i));
				sumCoeff[best]+=alpha[i];
			}else{//split
				double bestFraction=lim-sumCoeff[best];//fill best coefficient
				if(lim-sumCoeff[secondBest]<alpha[i]-bestFraction){//cannot feasibly split in two best coefficients
					bestFraction=alpha[i]-(lim-sumCoeff[secondBest]);//assign to best as much as possible
				}
				groups[best].insert(make_pair(bestFraction, i));
				sumCoeff[best]+=bestFraction;
				groups[secondBest].insert(make_pair(alpha[i]-bestFraction, i));
				sumCoeff[secondBest]+=alpha[i]-bestFraction;
			}
		}	
	}

	double *alpha_backup=alpha;
	int nAlpha_backup=nAlpha;
	alpha=NULL;
	nAlpha=0;
	allocate(k);
	alpha=alpha_backup;
	nAlpha=nAlpha_backup;


	int numGradients=H.getNumGradients();
	double *synthetic=new double[numGradients];
	for(int i=0;i<k;++i){
		memset(synthetic, 0, sizeof(double)*numGradients);
		for(set<pair<double, int> >::iterator it=groups[i].begin();it!=groups[i].end();++it){
			for(int j=0;j<numGradients;++j){
				synthetic[j]+=(it->first/sumCoeff[i])*DBFunctions[j*nAlpha+(it->second)];
			}
		}
		MultiTensor tensor;
		createMultiTensorSingleComponent(synthetic, H, DBFDirections, groups[i], tensor);
		tensor.setDiffusivities(0,transDiffusion, transDiffusion, longDiffusion);
		setTensorAt(i,tensor);
		//setVolumeFraction(i,lim);
		setVolumeFraction(i,sumCoeff[i]);
	}

	delete[] synthetic;
	delete[] sumCoeff;
}

void MultiTensor::split(GDTI &H, double *DBFDirections, double *DBFunctions){
	if(alpha==NULL){
		return;
	}
	vector<pair<double, int> > vAlpha;
	double halfEnergy=0;
	for(int i=0;i<nAlpha;++i)if(alpha[i]>0){
		vAlpha.push_back(make_pair(alpha[i], i));
		halfEnergy+=alpha[i];
	}
	if(vAlpha.size()<2){
		return;
	}
	halfEnergy*=0.5;
	sort(vAlpha.rbegin(), vAlpha.rend());
	
	int base0=vAlpha[0].second;
	int base1=vAlpha[1].second;
	double *dir0=&DBFDirections[3*base0];
	double *dir1=&DBFDirections[3*base1];
	
	int numGradients=H.getNumGradients();
	double *function0=new double[numGradients];
	double *function1=new double[numGradients];
	vector<set<pair<double, int> > > groups(2);
	double energy0=0;
	double energy1=0;
	if(vAlpha[0].first>0.5){
		groups[0].insert(make_pair(0.5, base0));
		groups[1].insert(make_pair(vAlpha[0].first-0.5, base1));
		energy0=0.5;
		energy1=vAlpha[0].first-0.5;
	}else{
		groups[0].insert(make_pair(vAlpha[0].first, base0));
		energy0=vAlpha[0].first;
	}
	groups[1].insert(make_pair(vAlpha[1].first, base1));
	energy1+=vAlpha[1].first;
	
	//distribute alpha-energy
	for(unsigned i=2;i<vAlpha.size();++i){
		double *currentDir=&DBFDirections[3*vAlpha[i].second];
		double angle0=getAbsAngleDegrees(dir0,currentDir,3);
		double angle1=getAbsAngleDegrees(dir1,currentDir,3);
		if((angle0<angle1) && (energy0+vAlpha[i].first<halfEnergy)){
			energy0+=vAlpha[i].first;
			groups[0].insert(vAlpha[i]);
		}else if(angle0<angle1){//partial alpha-coefficient with respect to group 0
			double remaining=vAlpha[i].first-(halfEnergy-energy0);
			groups[0].insert(make_pair(halfEnergy-energy0, vAlpha[i].second));
			groups[1].insert(make_pair(remaining, vAlpha[i].second));
			energy0=halfEnergy;
			energy1+=remaining;
		}else if(energy1+vAlpha[i].first<halfEnergy){
			energy1+=vAlpha[i].first;
			groups[1].insert(vAlpha[i]);
		}else{//partial alpha-coefficient with respect to group 1
			double remaining=vAlpha[i].first-(halfEnergy-energy1);
			groups[1].insert(make_pair(halfEnergy-energy1, vAlpha[i].second));
			groups[0].insert(make_pair(remaining, vAlpha[i].second));
			energy1=halfEnergy;
			energy0+=remaining;
		}
	}
	//-------synthesize signals-------
	memset(function0, 0, sizeof(double)*numGradients);
	for(set<pair<double, int> >::iterator it=groups[0].begin();it!=groups[0].end();++it){
		for(int j=0;j<numGradients;++j){
			function0[j]+=(it->first/energy0)*DBFunctions[j*nAlpha+(it->second)];
		}
	}
	memset(function1, 0, sizeof(double)*numGradients);
	for(set<pair<double, int> >::iterator it=groups[1].begin();it!=groups[1].end();++it){
		for(int j=0;j<numGradients;++j){
			function1[j]+=(it->first/energy1)*DBFunctions[j*nAlpha+(it->second)];
		}
	}
	
	MultiTensor tensor0;
	MultiTensor tensor1;
	createMultiTensorSingleComponent(function0, H, DBFDirections, groups[0], tensor0);
	createMultiTensorSingleComponent(function1, H, DBFDirections, groups[1], tensor1);
	double *alpha_backup=alpha;
	int nAlpha_backup=nAlpha;
	alpha=NULL;
	nAlpha=0;
	allocate(2);
	alpha=alpha_backup;
	nAlpha=nAlpha_backup;
	setGroups(groups);
	double sumVF=tensor0.getVolumeFraction(0)+tensor1.getVolumeFraction(0);
	this->setTensorAt(0, tensor0);
	this->setTensorAt(1, tensor1);
	halfEnergy*=2;
	volumeFractions[0]=halfEnergy*volumeFractions[0]/sumVF;
	volumeFractions[1]=halfEnergy*volumeFractions[1]/sumVF;
	delete[] function0;
	delete[] function1;
}



void MultiTensor::recomputeBestR2Tensor(GDTI &H){
	if(numCompartments<2){
		return;
	}
	int numGradients=H.getNumGradients();
	double *synthetic=new double[numGradients];
	acquireWithScheme(H.get_b(),H.getGradients(), numGradients, 0, synthetic);
	double tensor[6];
	H.solve(1, synthetic, tensor);
	double eVal[3];
	double eVec[9];
	forceNonnegativeTensor(tensor,eVec, eVal);
	int maxIndex=getMaxIndex(eVal, 3);
	sort(eVal, eVal+3);
	this->allocate(1);
	this->setRotationMatrixFromPDD(0, &eVec[3*maxIndex]);
	this->setDiffusivities(0,eVal);
	this->setVolumeFraction(0,1);
	delete[] synthetic;
}

int nextZero(double *v, int n, int pos){
	do{
		++pos;
	}while((pos<n) && (v[pos]>0));
	if(pos>=n){
		return -1;
	}
	return pos;

}
void printProblem(double *Phi, int numGradients, int numDBFDirections, double *S){
	FILE *F=fopen("problem.txt","w");
	fprintf(F, "A=[");
	for(int i=0;i<numGradients;++i){
		for(int j=0;j<numDBFDirections;++j){
			fprintf(F,"%E",Phi[i*numDBFDirections+j]);
			if(j<numDBFDirections-1){
				fprintf(F,",");
			}else{
				if(i<numGradients-1){
					fprintf(F,";");
				}else{
					fprintf(F,"];\n");
				}
			}
		}
	}
	fprintf(F, "b=[");
	for(int i=0;i<numGradients;++i){
		fprintf(F, "%E", S[i]);
		if(i<numGradients-1){
			fprintf(F,",");
		}else{
			fprintf(F,"];\n");
		}
	}
	fclose(F);
}

void buildMultiTensor(double *S, double *alpha, double *DBFDirections, double *Phi, int numGradients, int numDBFDirections, 
					  double lambdaMin, double lambdaMid, double lambdaMax, MultiTensor &result){
	vector<pair<double, int> > v;
	for(int i=0;i<numDBFDirections;++i)if(alpha[i]>0){
		v.push_back(make_pair(alpha[i], i));
	}
	sort(v.rbegin(), v.rend());

	int *I=NULL;
	int mm=0;
	if((v.size()>1) && (v[0].first<3*v[1].first)){
		if((v.size()>2) && (0.9*v[1].first<v[2].first)){//three components
			mm=3;
		}else{//two components
			mm=2;
		}
	}else{//one component
		mm=1;
	}
	I=new int[mm];
	for(int i=0;i<mm;++i){
		I[i]=v[i].second;
	}
	double errorNNLARS=-1;
	nnls_subspace(Phi, numGradients, numDBFDirections, S, alpha, I, mm, &errorNNLARS);
	result.allocate(mm);
	for(int i=0;i<mm;++i){
		result.setDiffusivities(i,lambdaMin, lambdaMid, lambdaMax);
		result.setRotationMatrixFromPDD(i, &DBFDirections[3*v[i].second]);
		result.setVolumeFraction(i,alpha[I[i]]);
	}
	delete[] I;
}


class DBFStructData{
	public:
		double b;
		double *signal;
		double *vectorProducts;
		int numDBFDirections;
		int numGradients;
		double *alpha;

		DBFStructData(double _b, double *_signal, double *_vectorProducts, int _numDBFDirections, int _numGradients, double *_alpha){
			b=_b;
			signal=_signal;
			vectorProducts=_vectorProducts;
			numDBFDirections=_numDBFDirections;
			numGradients=_numGradients;
			alpha=_alpha;
		}
		~DBFStructData(){
		}
};

double DBFObjectiveFunction(double *x, int _n, void *_data){
	DBFStructData *data=(DBFStructData *)_data;
	int n=data->numGradients;
	int m=data->numDBFDirections;
	double *w=data->vectorProducts;
	double *alpha=data->alpha;
	double *signal=data->signal;
	double lambda_T=x[0];
	double lambda_D=x[1];
	double b=data->b;
	int pos=0;
	double c=exp(-b*lambda_T);
	double retVal=0;
	for(int k=0;k<n;++k){
		double residual=0;
		for(int i=0;i<m;++i,++pos){
			residual+=alpha[i]*exp(-b*lambda_D*w[pos]);
		}
		residual=c*residual -signal[k];
		retVal+=residual*residual;
	}
	return retVal*0.5;
}

void DBFGradientProfile(double *x, int _n, double *y, int _m, void *_data){
	DBFStructData *data=(DBFStructData *)_data;
	int n=data->numGradients;
	int m=data->numDBFDirections;
	double *w=data->vectorProducts;
	double *alpha=data->alpha;
	double *signal=data->signal;
	double lambda_T=x[0];
	double lambda_D=x[1];
	double b=data->b;
	int pos=0;
	double c=exp(-b*lambda_T);

	double &dLS=y[0];
	double &dLD=y[1];
	dLS=0;
	dLD=0;
	for(int k=0;k<n;++k){
		double residual=0;
		double fit=0;
		double dPhiDld=0;
		for(int i=0;i<m;++i,++pos){
			double currentEval=alpha[i]*exp(-b*lambda_D*w[pos]);
			fit+=currentEval;
			dPhiDld+=w[pos]*currentEval;
		}
		fit*=c;
		dPhiDld*=c*(-b);
		//--
		residual=fit-signal[k];
		dLS-=b*residual*fit;
		//--
		dLD+=residual*dPhiDld;
	}
}

void DBFHessianProfile(double *x, int _n, double *y, int _m, void *_data){
	DBFStructData *data=(DBFStructData *)_data;
	int n=data->numGradients;
	int m=data->numDBFDirections;
	double *w=data->vectorProducts;
	double *alpha=data->alpha;
	double *signal=data->signal;
	double lambda_T=x[0];
	double lambda_D=x[1];
	double b=data->b;
	int pos=0;
	double c=exp(-b*lambda_T);

	double &dLSS=y[0];
	double &dLDD=y[3];
	double &dLDS=y[1];

	dLSS=0;
	dLDD=0;
	dLDS=0;
	for(int k=0;k<n;++k){
		double fit=0;
		double dPhiDld=0;
		double dPhiDld2=0;
		for(int i=0;i<m;++i,++pos){
			double currentEval=alpha[i]*exp(-b*lambda_D*w[pos]);
			fit+=currentEval;
			dPhiDld+=w[pos]*currentEval;
			dPhiDld2+=w[pos]*w[pos]*currentEval;
		}
		fit*=c;
		dPhiDld*=c*(-b);
		dPhiDld2*=c*(b*b);
		//--
		double residual=fit-signal[k];
		//--
		dLSS+=b*b*residual*fit+fit*fit;
		//--
		dLDS-=b*(fit*dPhiDld+residual*dPhiDld);
		//--
		dLDD+=residual*dPhiDld2+dPhiDld*dPhiDld;
	}
	y[2]=y[1];
}


void MultiTensor::fitDBFToSignal(double *S, GDTI &H, double *DBFDirections, int numDBFDirections, double b, double longDiffusion, double transDiffusion, bool iterateDiffProp){
	//---
	double *gradients=H.getGradients();
	int numGradients=H.getNumGradients();
	//---
	double *directions=new double[3*numDBFDirections];
	double *Phi=new double[numGradients*numDBFDirections];
	double *currentAlpha=new double[numDBFDirections];
	double *newAlpha=new double[numDBFDirections];
	memcpy(directions, DBFDirections, sizeof(double)*3*numDBFDirections);
	for(int i=0;i<numDBFDirections;++i){
		//computeDiffusionFunction(&directions[3*i], transDiffusion, transDiffusion, longDiffusion, b, gradients, numGradients, &Phi[i], numDBFDirections);
		computeDiffusionFunction(&directions[3*i], transDiffusion, longDiffusion-transDiffusion, b, gradients, numGradients, &Phi[i], numDBFDirections);
	}


	const double tolerance=1e-9;
	int maxIter=2000;
	int iter=0;
	double prevErr=-1;
	nnls(Phi, numGradients, numDBFDirections, S, currentAlpha, &prevErr);

	double *synthetic=new double[numGradients];
	int nnCount=numDBFDirections;
	double derr=tolerance+1;
	double originalTransDiffusion=transDiffusion;
	double originalLongDiffusion=longDiffusion;
	int MAX_DBF_DIR=4;
	bool stopCriterion=false;
	


	DBFStructData data(H.get_b(),S,NULL,numDBFDirections, numGradients, currentAlpha);


	while(!stopCriterion){
		
		++iter;
		
#ifdef ITERATIVE_VISUAL_DEBUG
		//nnlars(Phi, numGradients, numDBFDirections, S, currentAlpha, 0.3*iter, &prevErr, false);
		double currentDiffProfile[3]={transDiffusion, transDiffusion, longDiffusion};
		MultiTensorField tmpField(1,1,1);
		tmpField.getVoxels()->createFromAlphaProfile(numDBFDirections, currentAlpha, directions, currentDiffProfile);
		ostringstream os;
		os<<"iteration_"<<iter<<".txt";
		tmpField.saveToTxt(os.str());
		/*if(iter<maxIter){
			continue;
		}*/
		//break;
		/*cv::Mat M;
		showOrientationHistogram(currentAlpha, NULL, numDBFDirections, directions, 100, numDBFDirections*4, M);
		cv::imshow("Alpha",M);
		int key=0;
		while(key!=13){
			key=cv::waitKey(10);
		}*/
#endif
		vector<pair<double, int> > positive;
		vector<int > zeros;
		for(int i=0;i<numDBFDirections;++i){
			if(currentAlpha[i]>0){
				positive.push_back(make_pair(currentAlpha[i], i));
			}else{
				zeros.push_back(i);
			}
		}

		int ps=positive.size();
		int newSize=(ps*(ps+1))/2;
		double *newDirections=new double[3*newSize];
		sort(positive.rbegin(), positive.rend());
		for(unsigned i=0;i<positive.size();++i){
			memcpy(&newDirections[3*i], &directions[3*positive[i].second],sizeof(double)*3);
		}
		int pos=positive.size();
		for(unsigned i=0;i<positive.size();++i){
			double *pdd0=&directions[3*positive[i].second];
			for(unsigned j=i+1;j<positive.size();++j){
				double *pdd1=&directions[3*positive[j].second];
				double sumFractions=positive[i].first+positive[j].first;
				double newPdd[3]={0.5*(pdd0[0]+pdd1[0]), 0.5*(pdd0[1]+pdd1[1]), 0.5*(pdd0[2]+pdd1[2])};
				double npddn=sqrt(SQR(newPdd[0])+SQR(newPdd[1])+SQR(newPdd[2]));
				newPdd[0]/=npddn;
				newPdd[1]/=npddn;
				newPdd[2]/=npddn;
				/*bool drop=false;
				for(int kk=0;kk<pos;++kk){
					double angle=getAbsAngleDegrees(newPdd, &newDirections[3*kk],3);
					if(angle<0.5){
						drop=true;
						break;
					}
				}
				if(drop){
					continue;
				}*/
				memcpy(&newDirections[3*pos], newPdd, sizeof(double)*3);
				++pos;
			}
		}
		delete[] directions;
		directions=newDirections;
		newDirections=NULL;
		numDBFDirections=pos;

		delete[] currentAlpha;
		currentAlpha=new double[numDBFDirections];
		memset(currentAlpha,0, sizeof(double)*numDBFDirections);
		for(unsigned i=0;i<positive.size();++i){
			currentAlpha[i]=positive[i].first;
		}

		delete[] Phi;
		Phi=new double[numGradients*numDBFDirections];
		for(int i=0;i<numDBFDirections;++i){
			//computeDiffusionFunction(&directions[3*i], transDiffusion, transDiffusion, longDiffusion, b, gradients, numGradients, &Phi[i], numDBFDirections);
			computeDiffusionFunction(&directions[3*i], transDiffusion, longDiffusion-transDiffusion, b, gradients, numGradients, &Phi[i], numDBFDirections);
		}
		
			
		delete[] newAlpha;
		newAlpha=new double[numDBFDirections];
		double currentErr=-1;
		nnls(Phi, numGradients, numDBFDirections, S, newAlpha, &currentErr);
		//nnsls_pgs(Phi, numGradients, numDBFDirections, S, newAlpha, -0.005, &currentErr);
		derr=0;
		if(currentErr<prevErr){
			derr=prevErr-currentErr;
			prevErr=currentErr;
		}else{
			derr=0;
		}
		int nzCount=0;
		double maxCoeff=0;
		double minCoeff=1e10;
		double sumAlpha=0;
		for(int i=0;i<numDBFDirections;++i)if(newAlpha[i]>0){
			sumAlpha+=newAlpha[i];
			++nzCount;
			maxCoeff=MAX(maxCoeff, newAlpha[i]);
			minCoeff=MIN(minCoeff, newAlpha[i]);
		}
		//---new tests---
		//cerr<<"s. alpha:"<<sumAlpha<<endl;
		/*if(true){
			double deltaLambda_T=-log(sumAlpha)/b;
			cerr<<"Old L_T="<<transDiffusion<<".\tDelta L_T="<<deltaLambda_T<<endl;
			transDiffusion+=deltaLambda_T;
			longDiffusion+=deltaLambda_T;
		}*/
		/*bool profileChanged=false;
		double deltaProfile=0;
		if(iterateDiffProp && ((nzCount>MAX_DBF_DIR)) && (derr<=tolerance) && (3*transDiffusion<longDiffusion)){
			double x[2]={transDiffusion, longDiffusion-transDiffusion};
			double B[4]={1.0, 0.0, 0.0, 1.0};
			double tol=1e-9;
			int maxIter=20;
			int pos=0;
			double *vectorProducts=new double[numGradients*numDBFDirections];
			for(int ii=0;ii<numGradients;++ii){
				for(int jj=0;jj<numDBFDirections;++jj, ++pos){
					double prod=dotProduct(&directions[3*jj], &gradients[3*ii], 3);
					vectorProducts[pos]=prod*prod;
				}
			}
			data.alpha=newAlpha;
			data.vectorProducts=vectorProducts;
			data.numDBFDirections=numDBFDirections;

			double obj=BFGS_Sherman_Morrison(DBFObjectiveFunction, DBFGradientProfile, x, B, 2, tol, maxIter, (void *)&data);
			delete[] vectorProducts;
			if((fabs(transDiffusion-x[0])>1e-9) || (fabs(longDiffusion-(x[1]+x[0]))>1e-9)){
				transDiffusion=x[0];
				longDiffusion=x[1]+x[0];
				prevErr=1e10;
				profileChanged=true;
				deltaProfile=fabs(transDiffusion-x[0])+fabs(longDiffusion-(x[1]+x[0]));
			}
			//longDiffusion/=1.05;
		}
		stopCriterion=((iterateDiffProp==false)||((nzCount<=MAX_DBF_DIR) ) || (!profileChanged)) && (derr<tolerance);
		if((0<=maxIter) && (maxIter<=iter)){
			stopCriterion=true;
		}
		if((maxIter>0) && (2*iter>maxIter)){
			cerr<<"nzCount="<<nzCount<<".\tDeltaProfile="<<deltaProfile<<".\tdError="<<derr<<endl;
		}
		memcpy(currentAlpha,newAlpha,sizeof(double)*numDBFDirections);
		*/
		//------------
		//----old version----
		if(iterateDiffProp && ((nzCount>MAX_DBF_DIR)) && (derr<=tolerance) && (3*transDiffusion<longDiffusion)){
			//transDiffusion*=1.2;
			longDiffusion/=1.05;// modifying the longitudinal diffusion instead of the transversal diffusion is slightly better
			prevErr=1e10;
		}
		stopCriterion=((iterateDiffProp==false)||((nzCount<=MAX_DBF_DIR) ) || (longDiffusion<=3*transDiffusion)) && (derr<tolerance);
		if((0<=maxIter) && (maxIter<=iter)){
			stopCriterion=true;
		}
		memcpy(currentAlpha,newAlpha,sizeof(double)*numDBFDirections);
		//------------------
	}
	double sumAlpha=0;
	for(int i=0;i<numDBFDirections;++i)if(currentAlpha[i]>0){
		sumAlpha+=currentAlpha[i];
	}
	//=====================enforce sparsity========================
	if(iterateDiffProp){
		double pgs_err=-1;
		nnsls_pgs(Phi, numGradients, numDBFDirections, S, currentAlpha, -0.1, &pgs_err);
	}
	
	//=====================sort and drop zeros=====================
	transDiffusion=originalTransDiffusion;
	longDiffusion=originalLongDiffusion;
	vector<pair<double, int> > v;
	sumAlpha=0;
	for(int i=0;i<numDBFDirections;++i)if(currentAlpha[i]>0){
		v.push_back(make_pair(currentAlpha[i], i));
		sumAlpha+=currentAlpha[i];
	}
	int nzCount=v.size();
	sort(v.rbegin(), v.rend());
	double *sortedDirs=new double[nzCount*3];
	//warning: Phi will have more space than needed
	for(int i=0;i<nzCount;++i){
		memcpy(&sortedDirs[3*i], &directions[3*v[i].second], sizeof(double)*3);
		currentAlpha[i]=v[i].first;
	}
	setAlpha(currentAlpha, nzCount);
	setDirections(sortedDirs, nzCount);
	double currentDiffProfile[3]={transDiffusion, transDiffusion, longDiffusion};
	createFromAlphaProfile(nzCount, currentAlpha, sortedDirs, currentDiffProfile);
	//================c-means===============
	maxIter=50;
	double minDist=1e20;
	bool initialized=false;
	int *labels=new int[nzCount];
	for(int K=1;K<=3;++K){
		double *means=NULL;
		double *probs=NULL;
		int clustersFound=angularFuzzyKMeans(sortedDirs, currentAlpha,nzCount, 3, K, 1.0, maxIter, means, probs);
		//----normalize the centroids---
		double coeff[3]={0,0,0};
		for(int i=0;i<nzCount;++i){
			int label=-1;
			for(int c=0;c<K;++c){
				if(probs[K*i+c]>0){
					label=c;
					break;
				}
			}
			labels[i]=label+1;
			coeff[label]+=currentAlpha[i];
		}
		MultiTensor opc;
		opc.allocate(clustersFound);
		int cnt=0;
		for(int c=0;c<K;++c)if(coeff[c]>0){
			opc.setDiffusivities(cnt,transDiffusion, transDiffusion, longDiffusion);
			opc.setRotationMatrixFromPDD(cnt, &means[3*c]);
			opc.setVolumeFraction(cnt,coeff[c]);
			++cnt;
		}


			opc.acquireWithScheme(b,gradients, numGradients, 0,synthetic);
			double opcDist=euclideanDistanceSQR(synthetic, S, numGradients);
			if((!initialized) || (opcDist<minDist)){
				initialized=true;
				copyFrom(opc);
				minDist=opcDist;
				setGroups(labels, nzCount);
			}

		delete[] means;
		delete[] probs;
	}
	dropSmallPeaks(0.3);
	setAlpha(currentAlpha, nzCount);
	setDirections(sortedDirs, nzCount);
	//------------- re-compute the diffusivity profile-------------
	/*int pos=0;
	int nc=this->getNumCompartments();
	double *vectorProducts=new double[numGradients*nc];
	double *sc=this->getVolumeFractions();
	
	for(int ii=0;ii<numGradients;++ii){
		for(int jj=0;jj<nc;++jj, ++pos){
			double pdd[3];
			this->getPDD(jj,pdd);
			double prod=dotProduct(pdd, &gradients[3*ii], 3);
			vectorProducts[pos]=prod*prod;
		}
	}
	data.alpha=sc;
	data.vectorProducts=vectorProducts;
	data.numDBFDirections=nc;
	double x[2]={transDiffusion, longDiffusion-transDiffusion};

	double B[4];
	DBFHessianProfile(x, 2, B, 4, (void *)&data);
	computeInverse(B,2);
	double tol=1e-9;
	double obj=BFGS_Sherman_Morrison(DBFObjectiveFunction, DBFGradientProfile, x, B, 2, tol, 100, (void *)&data);
	//double obj=BFGS_Nocedal(DBFObjectiveFunction, DBFGradientProfile, x, B, 2, tol, 100, (void *)&data);
	delete[] vectorProducts;
	for(int i=0;i<nc;++i){
		this->setDiffusivities(i, x[0], x[0], x[0]+x[1]);
	}
	
	transDiffusion=x[0];
	longDiffusion=x[1]+x[0];*/
	//-------------------------------------------------------------
#ifdef ITERATIVE_VISUAL_DEBUG
	/*cv::Mat M;
	showOrientationHistogram(currentAlpha, NULL, numDBFDirections, directions, 100, numDBFDirections*4, M);
	cv::imshow("Alpha",M);
	int key=0;
	while(key!=' '){
		key=cv::waitKey(10);
	}*/
#endif
	delete[] synthetic;
	delete[] directions;
	delete[] Phi;
	delete[] currentAlpha;
	delete[] sortedDirs;
	//delete[] labels;
}

void MultiTensor::fitDBFToSignal_baseline(double *S, GDTI &H, double *DBFDirections, double *&Phi, int numDBFDirections, double longDiffusion, double transDiffusion, std::vector<std::set<int> > &neighborhoods, double BPT){
	double *gradients=H.getGradients();
	int numGradients=H.getNumGradients();
	double b=H.get_b();
	//---
	if(Phi==NULL){
		Phi=new double[numGradients*numDBFDirections];
		for(int i=0;i<numDBFDirections;++i){
			computeDiffusionFunction(&DBFDirections[3*i], transDiffusion, longDiffusion-transDiffusion, b, gradients, numGradients, &Phi[i], numDBFDirections);
		}
	}
	double *alpha=new double[numDBFDirections];
	double prevErr=-1;
	nnls(Phi, numGradients, numDBFDirections, S, alpha, &prevErr);
	groupCoefficients(alpha, DBFDirections, numDBFDirections, neighborhoods, BPT, transDiffusion, longDiffusion, *this);
	delete[] alpha;
}
void MultiTensor::setTensorAt(int k, MultiTensor &T, int sel){
	if((k>=numCompartments) || (T.getNumCompartments()<=sel)){
		return;
	}
	this->volumeFractions[k]=T.getVolumeFraction(sel);
	this->setRotationMatrix(k,&(T.getRotationMatrices()[9*sel]));
	this->setDiffusivities(k,&(T.getDiffusivities()[3*sel]));
}

void MultiTensor::setTensorAt(int k, Tensor &T){
	if((k>=numCompartments) || (k<0)){
		return;
	}
	this->volumeFractions[k]=T.getVolumeFraction();
	this->setRotationMatrix(k,T.getRotationMatrix());
	this->setDiffusivities(k,T.getDiffusivities());
}

void MultiTensor::fitMultiTensor(double S0, double *S, double *gradients, int numGradients, 
								 double b, double *DBFDirections, int numDBFDirections){

	//---estimate diffusivities---
	double longDiffusion;
	double transDiffusion;
	GDTI H(2, b, gradients, numGradients);
	double R2tensor[6];
	double eVec[9];
	double eVal[3];
	double md=H.solve(S0,S, R2tensor);
	forceNonnegativeTensor(R2tensor,eVec, eVal);
	sort(eVal, eVal+3);
	double linear, planar, spherical;
	int tensorType=H.computeLinearPlanarSphericalCoeffs(eVal, linear, planar, spherical);
	if(tensorType==1){//linear
		longDiffusion=eVal[2];
		transDiffusion=0.5*(eVal[0]+eVal[1]);
	}else if(tensorType==2){
		longDiffusion=0.5*(eVal[2]+eVal[1]);
		transDiffusion=eVal[0];
	}else{
		longDiffusion=eVal[2];
		transDiffusion=eVal[0];
	}
	
	//-----solve DBF----
	DBF dbfInstance(b, longDiffusion,transDiffusion,gradients, numGradients, DBFDirections, numDBFDirections);
	dbfInstance.setSolverType(DBFS_NNLS);
	double *alpha=new double[numDBFDirections];
	dbfInstance.solve(S, alpha, -1, false, NULL);
	//-----Group coefficients and get big peaks----
	double *RES_pdds=new double[3*(numDBFDirections+1)];
	double *RES_amount=new double[numDBFDirections+1];
	int RES_count;
	vector<set<int> > neighborhoods;
	int clusteringNeighSize=16;
	double bigPeaksThreshold=0.3;
	buildNeighborhood(DBFDirections, numDBFDirections, clusteringNeighSize, neighborhoods);
	groupCoefficients(alpha, DBFDirections, numDBFDirections, neighborhoods, RES_pdds, RES_amount, RES_count);
	getBigPeaks(bigPeaksThreshold, RES_pdds, RES_amount, RES_count);
	//---build multi-tensor----
	this->dellocate();
	this->allocate(RES_count);
	for(int i=0;i<RES_count;++i){
		this->setDiffusivities(i, transDiffusion, transDiffusion, longDiffusion);
	}
	this->setVolumeFractions(RES_amount);
	for(int k=0;k<RES_count;++k){
		this->setRotationMatrixFromPDD(k,&RES_pdds[3*k]);
	}
	delete[] RES_pdds;
	delete[] RES_amount;
	//------------------------------

}
int MultiTensor::numLeavesContentionTree(double b){
	int *isContained=new int[numCompartments];
	memset(isContained,0,sizeof(int)*numCompartments);
	for(int i=0;i<numCompartments;++i){
		double pdd_i[3];
		getPDD(i,pdd_i);
		double *R_i=&rotationMatrices[i*9];
		double *lambda_i=&diffusivities[i*3];
		double Di[9]={//Di=D*R'
			R_i[0]*lambda_i[0], R_i[3]*lambda_i[0], R_i[6]*lambda_i[0],
			R_i[1]*lambda_i[1], R_i[4]*lambda_i[1], R_i[7]*lambda_i[1],
			R_i[2]*lambda_i[2], R_i[5]*lambda_i[2], R_i[8]*lambda_i[2]
		};
		multMatrixMatrix<double>(R_i,Di,3,Di);
		for(int j=i+1;j<numCompartments;++j){
			double pdd_j[3];
			getPDD(j,pdd_j);
			double *R_j=&rotationMatrices[j*9];
			double *lambda_j=&diffusivities[j*3];
			double Dj[9]={//Dj=D*Rj'
				R_j[0]*lambda_j[0], R_j[3]*lambda_j[0], R_j[6]*lambda_j[0],
				R_j[1]*lambda_j[1], R_j[4]*lambda_j[1], R_j[7]*lambda_j[1],
				R_j[2]*lambda_j[2], R_j[5]*lambda_j[2], R_j[8]*lambda_j[2]
			};
			multMatrixMatrix<double>(R_j,Dj,3,Dj);
			//is i contained in j?
			double max_i=evaluateQuadraticForm(Di, pdd_i, 3);
			double i_eval_j=evaluateQuadraticForm(Dj, pdd_i, 3);
			max_i=volumeFractions[i]*exp(-b*max_i);
			i_eval_j=volumeFractions[j]*exp(-b*i_eval_j);
			if(max_i>i_eval_j){//it is 
				isContained[i]++;
			}
			//is j contained in i?
			double max_j=evaluateQuadraticForm(Dj, pdd_j, 3);
			double j_eval_i=evaluateQuadraticForm(Di, pdd_j, 3);
			max_j=volumeFractions[j]*exp(-b*max_j);
			j_eval_i=volumeFractions[i]*exp(-b*j_eval_i);
			if(max_j>j_eval_i){//it is
				isContained[j]++;
			}
		}
	}
	int numLeaves=0;
	for(int i=0;i<numCompartments;++i){
		if(isContained[i]==0){//i is a leave
			++numLeaves;
		}
	}
	delete[] isContained;
	return numLeaves;
}



#ifdef USE_QT

void MultiTensor::drawArrows(double px, double py, double pz, bool showGroupColors){
	if((nAlpha>0) && (directions!=NULL)){//it points to "somewhere", and "somewhere" is a valid array
		double position[3]={px,py,pz};
		double maxAlpha=getMaxAlpha();
		for(int i=0;i<nAlpha;++i){
			if(alpha[i]>0){
				double *direction=&directions[3*i];
				float arrowColor[3]={fabs(direction[0]), fabs(direction[1]), fabs(direction[2])};
				if(showGroupColors && (groups!=NULL) && (groups[i]!=-1)){//first priority: groups
					arrowColor[0]=(groups[i]&(1<<2))?0.75:0.25;
					arrowColor[1]=(groups[i]&(1<<1))?0.75:0.25;
					arrowColor[2]=(groups[i]&(1<<0))?0.75:0.25;
				}
				double e[3]={0,0,-1};
				double M[9];
				fromToRotation(direction, e, M);
				double T[16]={
					M[0], M[1]	, M[2]	, 0,
					M[3], M[4]	, M[5]	, 0,
					M[6], M[7]	, M[8]	, 0,
					px	,	py	,	pz	, 1
				};
				glMatrixMode(GL_MODELVIEW);
				glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arrowColor);
				if(selected){
					glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
				}else{
					glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
				}
				glPushMatrix();
					glLoadMatrixd(T);
					GLUquadricObj* obj = gluNewQuadric();
					double arrowLength=alpha[i];
					//double arrowHeadRadius=0.1*alpha[i]/sumAlpha;
					double arrowHeadRadius=0.03;
					double arrowShaftRadius=0.5*arrowHeadRadius;
					double arrowHeadLength=0.2*arrowLength;
					double arrowShaftLength=arrowLength-arrowHeadLength;

					gluCylinder( obj, arrowShaftRadius, arrowShaftRadius, arrowShaftLength, 20, 20); 
					glTranslated(0,0,arrowShaftLength);
					gluCylinder( obj, arrowHeadRadius, 0, arrowHeadLength, 20, 20); 
					gluDeleteQuadric(obj);
				glPopMatrix();
			}
		}
	}else{
		double maxFrac=getMaxVolumeFraction();
		for(int i=0;i<numCompartments;++i){
			double pdd[3];
			getPDD(i,pdd);
			float arrowColor[3]={fabs(pdd[0]), fabs(pdd[1]), fabs(pdd[2])};
			double e[3]={0,0,-1};
			double M[9];
			fromToRotation(pdd, e, M);
			double T[16]={
				M[0], M[1]	, M[2]	, 0,
				M[3], M[4]	, M[5]	, 0,
				M[6], M[7]	, M[8]	, 0,
				px	,	py	,	pz	, 1
			};
			glMatrixMode(GL_MODELVIEW);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arrowColor);
			if(selected){
				glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
			}else{
				glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
			}
			glPushMatrix();
				glLoadMatrixd(T);
				GLUquadricObj* obj = gluNewQuadric();
				double arrowLength=volumeFractions[i];
				double arrowHeadRadius=0.03;
				double arrowShaftRadius=0.5*arrowHeadRadius;
				double arrowHeadLength=0.2*arrowLength;
				double arrowShaftLength=arrowLength-arrowHeadLength;

				gluCylinder( obj, arrowShaftRadius, arrowShaftRadius, arrowShaftLength, 20, 20); 
				glTranslated(0,0,arrowShaftLength);
				gluCylinder( obj, arrowHeadRadius, 0, arrowHeadLength, 20, 20); 
				gluDeleteQuadric(obj);
			glPopMatrix();
		}
	}
}

#ifdef USE_QT
void drawSphericalMesh(double *X, int n, int m){
	/*double dPhi=2.0*M_PI/(n-1);
	double dTheta=M_PI/(m-1);
	double *X=new double[3*n*m];*/
	/*int pos=0;
	//compute point coordinates
	for(int i=0;i<n;++i){
		double cosPhi=cos(i*dPhi);
		double sinPhi=sin(i*dPhi);
		for(int j=0;j<m;++j,++pos){
			double cosTheta=cos(i*dTheta);
			double sinTheta=sin(i*dTheta);
			double r=M[i*m+j];
			X[3*pos]=r*cosPhi*sinTheta;
			X[3*pos+1]=r*sinPhi*sinTheta;
			X[3*pos+2]=r*cosTheta;
		}
	}*/
	double *N=new double[3*n*m];
	int neighTriangles_dRow[6][3]={{0, -1, -1},{0, 0, -1},{0, 1, 0},{0,  1, 1},{0,  0,  1},{0, 1,  0}};
	int neighTriangles_dCol[6][3]={{0,  1,  0},{0, 1,  1},{0, 0, 1},{0, -1, 0},{0, -1, -1},{0, 0, -1}};
	
	//compute point normals
	int pos=0;
	double minNorm=1e10;
	double maxNorm=-1;
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j,++pos){
			double opc=dotProduct(&X[3*pos], &X[3*pos],3);
			minNorm=MIN(minNorm,opc);
			maxNorm=MAX(maxNorm,opc);
			double *currentNormal=&N[3*pos];
			memset(currentNormal,0,sizeof(double)*3);
			for(int k=0;k<6;++k){
				int ai=i;
				int aj=j;
				int bi=(n+i+neighTriangles_dRow[k][1])%n;
				int bj=(m+j+neighTriangles_dCol[k][1])%m;
				int ci=(n+i+neighTriangles_dRow[k][2])%n;
				int cj=(m+j+neighTriangles_dCol[k][2])%m;
				int a=ai*m+aj;
				int b=bi*m+bj;
				int c=ci*m+cj;

				double nrm[3];
				triangleNormal(&X[3*a], &X[3*b], &X[3*c], nrm);
				double dProd=dotProduct(nrm, &X[3*a], 3);
				if(dProd<0){
					currentNormal[0]-=nrm[0];
					currentNormal[1]-=nrm[1];
					currentNormal[2]-=nrm[2];
				}else{
					currentNormal[0]+=nrm[0];
					currentNormal[1]+=nrm[1];
					currentNormal[2]+=nrm[2];
				}
				
			}
			double nrm=sqrt(SQR(currentNormal[0])+SQR(currentNormal[1])+SQR(currentNormal[2]));
			currentNormal[0]/=nrm;
			currentNormal[1]/=nrm;
			currentNormal[2]/=nrm;
		}
	}
	double normDiff=maxNorm-minNorm;
	//draw triangles
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			double *xa=&X[3*(i*m+j)];
			double *xb=&X[3*(((i+1)%n)*m+j)];
			double *xc=&X[3*(i*m+(j+1)%m)];
			double *na=&N[3*(i*m+j)];
			double *nb=&N[3*(((i+1)%n)*m+j)];
			double *nc=&N[3*(i*m+(j+1)%m)];
			double r,g,b;
			float color[3];
			double nrm=(dotProduct(xa,xa,3)-minNorm)/normDiff;
			getIntensityColor<double>(255*nrm, r, g, b);
			
			color[0]=r/255.0;
			color[1]=g/255.0;
			color[2]=b/255.0;
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
			glNormal3dv(na); glVertex3dv(xa);
			glNormal3dv(nb); glVertex3dv(xb);
			glNormal3dv(nc); glVertex3dv(xc);


			nrm=(dotProduct(xa,xa,3)-minNorm)/normDiff;
			getIntensityColor<double>(255*nrm, r, g, b);
			color[0]=r/255.0;
			color[1]=g/255.0;
			color[2]=b/255.0;
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, color);
			xa=&X[3*(((i+1)%n)*m+j)];
			xb=&X[3*(((i+1)%n)*m+(j+1)%m)];
			xc=&X[3*(i*m+(j+1)%m)];
			na=&N[3*(((i+1)%n)*m+j)];
			nb=&N[3*(((i+1)%n)*m+(j+1)%m)];
			nc=&N[3*(i*m+(j+1)%m)];
			glNormal3dv(na); glVertex3dv(xa);
			glNormal3dv(nb); glVertex3dv(xb);
			glNormal3dv(nc); glVertex3dv(xc);

		}
	}
	delete[] N;
}
#endif

void MultiTensor::drawDiffusionFunction(double px, double py, double pz, int type){//0=ADF, 1=ODF
	double T[16]={
		1, 0	, 0	, 0,
		0, 1	, 0	, 0,
		0, 0	, 1	, 0,
		px	,	py	,	pz	, 1
	};
	if(selected){
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	}else{
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	}
	int resolution=100;
	int nSamplingPoints=resolution*resolution;
	double *samplingPoints=new double[3*nSamplingPoints];
	double *S=new double[nSamplingPoints];
	double dTheta=2.0*M_PI/resolution;
	double dPhi=2.0*M_PI/resolution;
	int pos=0;
	for(int i=0;i<resolution;++i){
		double cosPhi=cos(i*dPhi);
		double sinPhi=sin(i*dPhi);
		for(int j=0;j<resolution;++j,++pos){
			double cosTheta=cos(j*dTheta);
			double sinTheta=sin(j*dTheta);
			double x=cosPhi*sinTheta;
			double y=sinPhi*sinTheta;
			double z=cosTheta;
			samplingPoints[3*pos]=x;
			samplingPoints[3*pos+1]=y;
			samplingPoints[3*pos+2]=z;
		}
	}
	if(type==0){
		acquireWithScheme(2500000000,samplingPoints,nSamplingPoints,0,S);
		/*acquireWithScheme(1000,samplingPoints,nSamplingPoints,0,S);
		for(int i=nSamplingPoints-1;i>=0;--i){
			S[i]=-log(S[i])/10;
		}*/
	}else{
		computeODF(samplingPoints,nSamplingPoints, S);
		for(int i=nSamplingPoints-1;i>=0;--i){
			S[i]*=3000;
		}
	}
	for(int i=nSamplingPoints-1;i>=0;--i){
		samplingPoints[3*i]*=S[i];
		samplingPoints[3*i+1]*=S[i];
		samplingPoints[3*i+2]*=S[i];
	}

	glMatrixMode(GL_MODELVIEW);
	if(selected){
		glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
	}else{
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	}
	glPushMatrix();
		glLoadMatrixd(T);
		glBegin(GL_TRIANGLES);
		drawSphericalMesh(samplingPoints,resolution, resolution);
		glEnd();
	glPopMatrix();
	delete[] samplingPoints;
	delete[] S;
	return;
}

void MultiTensor::drawSampledFunction(double px, double py, double pz, double *samplingPoints, int n){
	
	double *S=new double[n];
	
		acquireWithScheme(2500000000,samplingPoints,n,0,S);
		/*acquireWithScheme(1000,samplingPoints,nSamplingPoints,0,S);
		for(int i=nSamplingPoints-1;i>=0;--i){
			S[i]=-log(S[i])/10;
		}*/
	

	int resolution=10;
	
	GLUquadricObj* obj = gluNewQuadric();

	glMatrixMode(GL_MODELVIEW);
	
		for(int i=0;i<n;++i){
			double coords[3]={
				samplingPoints[3*i+0]*S[i],
				samplingPoints[3*i+1]*S[i],
				samplingPoints[3*i+2]*S[i]
			};
			
			float pddColor[3]={
				fabs(coords[0]),
				fabs(coords[1]),
				fabs(coords[2])
			};
			double sum=pddColor[0]+pddColor[1]+pddColor[2];
			pddColor[0]/=sum;
			pddColor[1]/=sum;
			pddColor[2]/=sum;
			double T[16]={
				1, 0	, 0	, 0,
				0, 1	, 0	, 0,
				0, 0	, 1	, 0,
				px+coords[0]	,	py+coords[1]	,	pz+coords[2]	, 1
			};
			glPushMatrix();
			glLoadMatrixd(T);
			glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pddColor);
			gluSphere(obj, 0.0075, resolution, resolution);
			glPopMatrix();
		}
	

	gluDeleteQuadric(obj);
	delete[] S;
}



void MultiTensor::drawEllipsoid(int k, double px, double py, double pz, double intensity, bool clusterColors){
	for(int c=0;c<numCompartments;++c){
		if(clusterColors){
			if((k!=-1)&&(compartmentSegmentation[c]!=k)){
				continue;
			}
		}else if((k!=-1) && (c!=k)){
			continue;
		}
		double sc=volumeFractions[c];
		double pdd[3];
		double *diffProfile=getDiffusivities(c);
		getPDD(c,pdd);
		double maxVal=MAX(MAX(fabs(pdd[0]), fabs(pdd[1])),fabs(pdd[2]));
		if(maxVal<EPSILON){
			continue;
		}
		double e[3]={0,0,1};
		double M[9];
		fromToRotation(pdd, e, M);
		double T[16]={
			M[0], M[1]	, M[2]	, 0,
			M[3], M[4]	, M[5]	, 0,
			M[6], M[7]	, M[8]	, 0,
			px	,	py	,	pz	, 1
		};
		float pddColor[3];
		if(clusterColors && (compartmentSegmentation!=NULL)){
			pddColor[0]=1.0*(((compartmentSegmentation[c]/2)&1)!=0);
			pddColor[1]=1.0*(((compartmentSegmentation[c]/2)&2)!=0);
			pddColor[2]=1.0*(((compartmentSegmentation[c]/2)&4)!=0);
		}else if(intensity<0){
			pddColor[0]=fabs(pdd[0]);
			pddColor[1]=fabs(pdd[1]);
			pddColor[2]=fabs(pdd[2]);
		}else{
			double r,g,b;
			getIntensityColor<double>(255*intensity, r, g, b);
			pddColor[0]=r/255.0;
			pddColor[1]=g/255.0;
			pddColor[2]=b/255.0;
		}
		if(selected){
			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		}else{
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		}
		
		glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pddColor);
		glMatrixMode(GL_MODELVIEW);
		
		int resolution=10;
		glPushMatrix();
			glLoadMatrixd(T);
			GLUquadricObj* obj = gluNewQuadric();
			glScaled(diffProfile[0]/diffProfile[2], diffProfile[1]/diffProfile[2], 1);
			gluSphere(obj, 0.75*sc, resolution, resolution);
			gluDeleteQuadric(obj);
		glPopMatrix();
	}
}
//-----------------------------------------------------------------------
double colinearityMeasure(double *dirA, double *dirB, double *displacementDir, double d0, double theta0, double dTheta0){
	double d2=SQR(displacementDir[0])+SQR(displacementDir[1])+SQR(displacementDir[2]);
	double angle0=getAbsAngle(dirA,displacementDir,3);
	double angle1=getAbsAngle(dirB,displacementDir,3);
	double U0=d2/(d0*d0)+(2-cos(2*angle0)-cos(2*angle1))/(1-cos(2*theta0))+(1-cos(2*(angle0-angle1)))/(1-cos(2*dTheta0));
	return exp(-U0);
}

double parallelityMeasure(double *dirA, double *dirB, double *displacementDir, double d0, double theta0, double dTheta0){
	double d2=SQR(displacementDir[0])+SQR(displacementDir[1])+SQR(displacementDir[2]);
	double angle0=getAbsAngle(dirA,displacementDir,3);
	double angle1=getAbsAngle(dirB,displacementDir,3);
	double U1=d2/(d0*d0)+(2-cos(2*angle0-0.5*M_PI)-cos(2*angle1-0.5*M_PI))/(1-cos(2*theta0))+(1-cos(2*(angle0-angle1)))/(1-cos(2*dTheta0));
	return exp(-U1);
}

void assignTensors(double *pddsA, int n, double *pddsB, int m, double *jSegment, int *assignment, Hungarian &hungarianSolver){
	int maxCompartments=MAX(n, m);
	if((n*m)==0){
		memset(assignment, -1, sizeof(int)*maxCompartments);
		return;
	}
	
	hungarianSolver.setSize(maxCompartments);
	hungarianSolver.setAllCosts(1e5);
	double *p=pddsA;
	for(int i=0;i<n;++i, p+=3){
		double *q=pddsB;
		for(int j=0;j<m;++j, q+=3){
			double angle=getAbsAngle(p,q,3);
			hungarianSolver.setCost(i,j,angle);
		}
	}
	double cost=hungarianSolver.solve();
	
	hungarianSolver.getAssignment(assignment);
	cost-=1e5*abs(n-m);	
}

#endif