#include "MultiTensorFieldExperimentParameters.h"
#include <vector>
#include <set>
#include <iostream>
#include <iomanip>
#include "GDTI.h"
#include "dtiutils.h"
#include "linearalgebra.h" 
#include "statisticsutils.h"
#include "geometryutils.h"
#include "nnls.h"
#include "stdio.h"
#include <sstream>
#include "DirectoryListing.h"
#include "utilities.h"
#include <algorithm>
#include <cv.h>
#include <highgui.h>
#include "denoising.h"
#include "volume_io.h"
#include "tictoc.h"
#include "macros.h"
#include "icp.h"
#include "histogram.h"
#include "morphological.h"
#include "histogram.h"
#include "bits.h"
#include "mrtrixio.h"
#include "ls.h"
#ifdef USE_GSL
	#include "CSDeconv.h"
	#include "SphericalHarmonics.h"
#include "SHEvaluator.h"
#endif
using namespace std;
#define LCI_THR (25.0)
//#define RUN_GRID_SEARCH
//#define THIN_PROFILE
#define USE_ALRAM_MASK
//#define ALRAM_MASK_NAME "challenge_test_SNR30_omar_LCI_filtered_mask.nii"
#define ALRAM_MASK_NAME "challenge_SNR10_LCI_filtered_mask.nii"


#ifdef RUN_GRID_SEARCH
	const double thrStart=0.1;
	const int thrSteps=5;
	const double thrDelta=0.05;

	const int neighSizeStart=6;
	const int neighSizeSteps=25;
	const int neighSizeDelta=1;
#endif
/*
R^2:
	4 neigh: MTFE_NEIGH_SIZE=4
	8 neigh: MTFE_NEIGH_SIZE=20
R^2:
	6 neigh:  MTFE_NEIGH_SIZE=6
	14 neigh: MTFE_NEIGH_SIZE=14
	26 neigh: MTFE_NEIGH_SIZE=26
*/
#define MTFE_NEIGH_SIZE 26
int MTFE_dRow[]=	{-1, 0, 1,  0,-1, 1, 1,-1, 0,  -1, 0, 1, 0,-1, 1, 1,-1,  -1, 0, 1, 0,-1, 1, 1,-1, 0};
int MTFE_dCol[]=	{ 0, 1, 0, -1, 1, 1,-1,-1, 0,   0, 1, 0,-1, 1, 1,-1,-1,  0, 1, 0, -1, 1, 1,-1,-1, 0};
int MTFE_dSlice[]=	{-1,-1,-1, -1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1 ,1 ,1, 1};
MultiTensorFieldExperimentParameters::MultiTensorFieldExperimentParameters(){
	inputType=MTFIT_GTGenerator;
	groundTruthType=MTFGTT_GTGenerator;
	loaded=false;
	strcpy(gtFname, "");
	
	strcpy(solutionFname,"");
	strcpy(solutionListFname,"");
	strcpy(basisDirsFname,"");
	strcpy(outputDir,"");
	strcpy(reconstructionFname,"recovered.txt");
	
	strcpy(dwFname,"");
	clusteringNeighSize=6;
	bigPeaksThreshold=0.2;
	alpha0=0;
	alpha1=0;
	b=0;
	WM_S0Threshold=-1;
	WM_FAThreshold=-1;
	WM_AVSignalThreshold=-1;
	WM_erosionSteps=0;
	denoisingTest=false;
	fitWithNeighbors=false;
	fitGDTIOnly=false;
	useFAIndicator=false;
	useGDTIPostProcessing=false;
	applyTensorSplitting=false;
	strcpy(gradientDirsFname, "");
	strcpy(ODFDirFname, "");
	saveODFField=false;
	regularizeAlphaField=false;
	iterativeBasisUpdate=false;
	iterateDiffProf=false;
	evaluate=false;
	evaluateList=false;
	nonparametric=false;
	SNR=0;
	numSamplings=1;
	solverType=DBFS_NNLS;
	reconstructed=NULL;
	lambda=0.9;
	spatialSmoothing=false;

	strcpy(nTensFname, "");
	strcpy(sizeCompartmentFname, "");
	strcpy(fPDDFname,"");
	strcpy(WM_maskFname,"");

	strcpy(rawDataBaseDir, "");
	strcpy(dwListFname,"");

	s0Volume=NULL;
	dwVolume=NULL;

	gradients=NULL;
	numGradients=0;

	DBFDirections=NULL;
	numDBFDirections=0;
	GT=NULL;
	reconstructed=NULL;
	ODFDirections=NULL;
	
}

MultiTensorFieldExperimentParameters::~MultiTensorFieldExperimentParameters(){
	if(GT!=NULL){
		delete GT;
	}
	if(reconstructed!=NULL){
		delete reconstructed;
	}

	DELETE_ARRAY(ODFDirections);
	DELETE_ARRAY(gradients);
	DELETE_ARRAY(DBFDirections);
}

void MultiTensorFieldExperimentParameters::printConfig(void){
	cerr<<"inputType\t"<<inputType<<endl;
	cerr<<"groundTruthType\t"<<groundTruthType<<endl;
	cerr<<"gtFname\t"<<gtFname<<endl;
	cerr<<"basisDirsFname\t"<<basisDirsFname<<endl;
	cerr<<"clusteringNeighSize\t"<<clusteringNeighSize<<endl;
	cerr<<"bigPeaksThreshold\t"<<bigPeaksThreshold<<endl;
	
	cerr<<"b\t"<<b<<endl;
	cerr<<"fitWithNeighbors\t"<<fitWithNeighbors<<endl;
	cerr<<"fitGDTIOnly\t"<<fitGDTIOnly<<endl;
	cerr<<"useFAIndicator\t"<<useFAIndicator<<endl;
	cerr<<"useGDTIPostProcessing\t"<<useGDTIPostProcessing<<endl;
	cerr<<"applyTensorSplitting\t"<<applyTensorSplitting<<endl;
	cerr<<"gradientDirsFname"<<gradientDirsFname<<endl;
	cerr<<"ODFDirFname\t"<<ODFDirFname<<endl;
	cerr<<"regularizeAlphaField\t"<<regularizeAlphaField<<endl;
	cerr<<"iterativeBasisUpdate\t"<<iterativeBasisUpdate<<endl;
	cerr<<"iterateDiffProf\t"<<iterateDiffProf<<endl;
	cerr<<"SNR\t"<<SNR<<endl;
	cerr<<"numSamplings\t"<<numSamplings<<endl;
	cerr<<"solverType\t"<<solverType<<endl;
}

int MultiTensorFieldExperimentParameters::loadData(void){
	loaded=false;
	//=====================check input======================
	if(groundTruthType==MTFGTT_GTGenerator){
		if(!fileExists(gtFname)){
			cerr<<"Error: Ground truth file '"<<gtFname<<"' not found."<<endl;
			return -1;
		}
	}else if(groundTruthType!=MTFGTT_None){
		//--verify that ground truth files exist--
		if(!fileExists(nTensFname)){
			cerr<<"Error: Ground truth file '"<<nTensFname<<"' not found."<<endl;
			return -1;
		}
		if(!fileExists(sizeCompartmentFname)){
			cerr<<"Error: Ground truth file '"<<sizeCompartmentFname<<"' not found."<<endl;
			return -1;
		}
		if(!fileExists(fPDDFname)){
			cerr<<"Error: Ground truth file '"<<fPDDFname<<"' not found."<<endl;
			return -1;
		}
	}

	if(inputType==MTFIT_DWMRI_FILES){
		//--verify that ground truth signal files exist
		if(!fileExists(string(rawDataBaseDir)+dwListFname)){
			cerr<<"Error: Input file '"<<string(rawDataBaseDir)+dwListFname<<"' not found."<<endl;
			return -1;
		}
	}


	if(!fileExists(string(rawDataBaseDir)+gradientDirsFname)){
		cerr<<"Error: Gradient directions file (scheme) '"<<string(rawDataBaseDir)+gradientDirsFname<<"' not found."<<endl;
		return -1;
	}
	if(!fileExists(ODFDirFname)){
		cerr<<"Error: ODF sampling directions file '"<<ODFDirFname<<"' not found."<<endl;
		return -1;
	}
	//======================================================
	//=====================initialization===================
	//======================================================
	DELETE_ARRAY(ODFDirections);
	loadArrayFromFile(ODFDirFname,ODFDirections,numODFDirections);
	if(numODFDirections%3){
		cerr<<"Error: directions array length is not multiple of 3."<<endl;
		return -1;
	}
	numODFDirections/=3;
	//----------load scheme--------------
	int *s0Indices=NULL;
	int numS0;
	DELETE_ARRAY(gradients);
	loadOrientations(string(rawDataBaseDir)+gradientDirsFname,gradients, numGradients, s0Indices, numS0);
	for(int i=0;i<numGradients;++i){
		double *g=&gradients[3*i];
		double nn=sqrt(dotProduct(g,g,3));
		for(int j=0;j<3;++j){
			g[j]/=nn;
		}
	}
	//-------load ground truth-----
	if(groundTruthType!=MTFGTT_None){
		DELETE_INSTANCE(GT);
		GT=new MultiTensorField();
		if(groundTruthType==MTFGTT_GTGenerator){
			GT->loadFromTxt(gtFname);
		}else{
			GT->loadFromNifti(nTensFname, sizeCompartmentFname, fPDDFname);
		}
		//----save crossing angles---
		/*if(GT->getNumSlices()==1){
			int nrows=GT->getNumRows();
			int ncols=GT->getNumCols();
			FILE *F=fopen("crossingAngles.txt","w");
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j){
					MultiTensor *vox=GT->getVoxelAt(0,i,j);
					double ca=vox->getMinAngleDegrees();
					fprintf(F,"%0.3lf\t", ca);
				}
				fprintf(F,"\n");
			}
			fclose(F);
		}*/
	}
	//---------------------------
	int nrows=-1;
	int ncols=-1;
	int nslices=-1;
	if(inputType==MTFIT_DWMRI_FILES){//DWMRI images (a separate file per gradient)
		vector<string> dwNames;
		readFileNamesFromFile(string(rawDataBaseDir)+dwListFname, dwNames);
		for(unsigned i=0;i<dwNames.size();++i){
			dwNames[i]=rawDataBaseDir+dwNames[i];
		}
		DELETE_ARRAY(s0Volume);
		DELETE_ARRAY(dwVolume);
		loadDWMRIFiles(dwNames, s0Indices, numS0, s0Volume, dwVolume, nrows, ncols, nslices);
	}else if(inputType==MTFIT_NIFTI){//nifti
		cerr<<"Loading from NIFTI...";
		int len;
		DELETE_ARRAY(s0Volume);
		DELETE_ARRAY(dwVolume);
		loadDWMRIFromNifti(string(rawDataBaseDir)+dwFname, s0Indices, numS0, s0Volume, dwVolume, nrows, ncols, nslices, len);
		cerr<<"done."<<endl;
		this->numGradients=len;
	}else if(groundTruthType!=MTFGTT_None){
		nrows=GT->getNumRows();
		ncols=GT->getNumCols();
		nslices=GT->getNumSlices();
	}
	
	if(reconstructed==NULL){
		reconstructed=new MultiTensorField;
	}

	int nvoxels=nrows*ncols*nslices;
	reconstructed->allocate(nslices, nrows, ncols);
	//----------load DBF directions--------------
	DELETE_ARRAY(DBFDirections);
	loadDiffusionDirections(basisDirsFname, DBFDirections, numDBFDirections);
	loaded=true;
	return 0;
}

void MultiTensorFieldExperimentParameters::dellocate(void){
	if(GT!=NULL){
		delete GT;
		GT=NULL;
	}
	if(reconstructed!=NULL){
		delete reconstructed;
		reconstructed=NULL;
	}
}

string getFileNameOnly(const string &s){
	size_t pos=s.find_first_of(".");
	return s.substr(0,pos);
}

void saveResults(double *A, int n, int m, vector<string> rowHeaders, vector<string> columnHeaders, const string &path, const string &fname){
	FILE *F=fopen(((path+PATH_SEPARATOR)+fname).c_str(),"w");
	fprintf(F,"0");
	for(int j=0;j<m;++j){
		fprintf(F,"\t%s", columnHeaders[j].c_str());
	}
	fprintf(F,"\n");
	double minVal=1e10;
	int sel=-1;
	for(int i=0;i<n;++i){
		fprintf(F,"%s", rowHeaders[i].c_str());
		for(int j=0;j<m;++j){
			fprintf(F,"\t%0.4lf", A[i*m+j]);
			if(A[i*m+j]<minVal){
				minVal=MIN(minVal, A[i*m+j]);
				sel=i*m+j;
			}
			
		}
		fprintf(F,"\n");
	}
	fclose(F);
	F=fopen(((path+PATH_SEPARATOR)+string("best_"+fname)).c_str(), "w");
	fprintf(F,"Min: %lf\n", minVal);
	fprintf(F,"thr: %s\n",rowHeaders[sel/m].c_str());
	fprintf(F,"ns: %s\n",columnHeaders[sel%m].c_str());
	fclose(F);
}

void generateErrorSummary(MultiTensorField &GT, MultiTensorField &reconstructed, double *ODFDirections, int numODFDirections, vector<double> &results, bool printResults, bool fullReport){
	vector<double> errorPositiveCompartmentCount;
	vector<double> errorNegativeCompartmentCount;
	vector<double> errorMissingWMVoxels;
	vector<double> errorExtraWMVoxels;
	vector<double> errorAngularPrecision;
	vector<double> errorODFAccuracy;
	evaluatePositiveCompartmentCount(	GT, reconstructed, errorPositiveCompartmentCount);
	evaluateNegativeCompartmentCount(	GT, reconstructed, errorNegativeCompartmentCount);
	evaluateMissingWMVoxels			(	GT, reconstructed, errorMissingWMVoxels);
	evaluateExtraWMVoxels			(	GT, reconstructed, errorExtraWMVoxels);
	evaluateAngularPrecision(	GT, reconstructed, errorAngularPrecision);
	evaluateODFAccuracy		(	GT, reconstructed, ODFDirections, numODFDirections, errorODFAccuracy);
	
	double errorPositiveCompartmentCountMean=getMean(errorPositiveCompartmentCount);
	double errorNegativeCompartmentCountMean=getMean(errorNegativeCompartmentCount);
	double errorMissingWMVoxelsMean=getMean(errorMissingWMVoxels);
	double errorExtraWMVoxelsMean=getMean(errorExtraWMVoxels);
	double errorAngularPrecisionMean=getMean(errorAngularPrecision);
	double errorODFAccuracyMean=getMean(errorODFAccuracy);

	double errorPositiveCompartmentCountSD=getStdev(errorPositiveCompartmentCount);
	double errorNegativeCompartmentCountSD=getStdev(errorNegativeCompartmentCount);
	double errorMissingWMVoxelsSD=getStdev(errorMissingWMVoxels);
	double errorExtraWMVoxelsSD=getStdev(errorExtraWMVoxels);
	double errorAngularPrecisionSD=getStdev(errorAngularPrecision);
	double errorODFAccuracySD=getStdev(errorODFAccuracy);
	if(printResults){
		cerr<<"Samples: "<<errorAngularPrecision.size()<<endl;
		cerr<<"-Comp. count : "<<errorNegativeCompartmentCountMean<<" ("<<errorNegativeCompartmentCountSD<<")"<<endl;
		cerr<<"+Comp. count : "<<errorPositiveCompartmentCountMean<<" ("<<errorPositiveCompartmentCountSD<<")"<<endl;	
		cerr<<"Missing vox : "<<errorMissingWMVoxelsMean<<" ("<<errorMissingWMVoxelsSD<<")"<<endl;
		cerr<<"Extra vox   : "<<errorExtraWMVoxelsMean<<" ("<<errorExtraWMVoxelsSD<<")"<<endl;	
		cerr<<"Angular error: "<<errorAngularPrecisionMean<<" ("<<errorAngularPrecisionSD<<")"<<endl;
		cerr<<"ODF error: "<<errorODFAccuracyMean<<" ("<<errorODFAccuracySD<<")"<<endl;
		cout<<errorNegativeCompartmentCountMean<<"\t"<<errorPositiveCompartmentCountMean<<"\t"<<errorAngularPrecisionMean<<"\n"<<endl;
	}
	if(fullReport){
		/*Code to analyse these data in R
			fname="fullReport_angular.txt";
			title="Angular, SNR=30";
			data<-read.table(fname);
			means<-colMeans(data);
			stdev<-sqrt(apply(data, 2, var));
			smeans<-means+stdev;
			imeans<-means-stdev;
			plot(smeans, xlab="Crossing angle", ylab="Error", main=title, col=2);
			lines(smeans, col=2);
			lines(means);
			lines(imeans,col=3);
			abline(h=(seq(0,20,2.5)), col="lightgray", lty="dotted");
			abline(v=(seq(0,100,10)), col="lightgray", lty="dotted");
		*/
		MultiTensor *voxGT=GT.getVoxels();
		map<int, vector<double> > E_angular;
		map<int, vector<double> > E_np;
		map<int, vector<double> > E_nm;
		map<int, int> sampleCount;
		int maxSamples=0;
		for(unsigned i=0;i<errorAngularPrecision.size();++i){
			double angle=voxGT[i].getMinAngleDegrees();
			int iAngle=int(angle);
			if(fabs(angle-(iAngle+1))<fabs(angle-iAngle)){
				++iAngle;
			}
			sampleCount[iAngle]++;
			maxSamples=MAX(maxSamples, sampleCount[iAngle]);
			E_angular[iAngle].push_back(errorAngularPrecision[i]);
			E_np[iAngle].push_back(errorPositiveCompartmentCount[i]);
			E_nm[iAngle].push_back(errorNegativeCompartmentCount[i]);
		}
		FILE *F=fopen("fullReport_angular.txt","w");
		for(int i=0;i<maxSamples;++i){
			for(map<int, vector<double> >::iterator it=E_angular.begin();it!=E_angular.end();++it){
				if(i<it->second.size()){
					fprintf(F, "%E\t",it->second[i]);
				}else{
					fprintf(F, "NA\t");
				}
			}
			fprintf(F, "\n");
		}
		fclose(F);

		F=fopen("fullReport_n+.txt","w");
		for(int i=0;i<maxSamples;++i){
			for(map<int, vector<double> >::iterator it=E_np.begin();it!=E_np.end();++it){
				if(i<it->second.size()){
					fprintf(F, "%E\t",it->second[i]);
				}else{
					fprintf(F, "NA\t");
				}
			}
			fprintf(F, "\n");
		}
		fclose(F);

		F=fopen("fullReport_n-.txt","w");
		for(int i=0;i<maxSamples;++i){
			for(map<int, vector<double> >::iterator it=E_nm.begin();it!=E_nm.end();++it){
				if(i<it->second.size()){
					fprintf(F, "%E\t",it->second[i]);
				}else{
					fprintf(F, "NA\t");
				}
			}
			fprintf(F, "\n");
		}
		fclose(F);
	}
	results.clear();
	results.push_back(errorNegativeCompartmentCountMean);
	results.push_back(errorPositiveCompartmentCountMean);
	results.push_back(errorAngularPrecisionMean);
	results.push_back(errorODFAccuracyMean);

	results.push_back(errorNegativeCompartmentCountSD);
	results.push_back(errorPositiveCompartmentCountSD);
	results.push_back(errorAngularPrecisionSD);
	results.push_back(errorODFAccuracySD);
}


int selectiveDenoiseDWVolume(double *dwVolume, int nrows, int ncols, int nslices, int numGradients, unsigned char *mask, double *neighFeature, double neighSimThr){
	int nvoxels=nrows*ncols*nslices;
	double *img=new double[nvoxels];
	double *denoised=new double[nvoxels];
	int dropped=0;
	for(int k=0;k<numGradients;++k){
		//--collect data from gradient k--
		int v=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					img[v]=dwVolume[v*numGradients+k];
				}
			}
		}
		//--denoise--
		dropped+=selectiveAverageVolumeDenoising(img, nrows, ncols, nslices, denoised, mask, neighFeature, neighSimThr);
		//--put the results back to the input memory--
		v=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					dwVolume[v*numGradients+k]=denoised[v];
				}
			}
		}
	}
	delete[] img;
	delete[] denoised;
	return dropped;

}

int denoiseDWVolume(double *dwVolume, int nrows, int ncols, int nslices, int numGradients, double lambda, double *denoisingParams, unsigned char *mask){
	int nvoxels=nrows*ncols*nslices;
	double *img=new double[nvoxels];
	double *denoised=new double[nvoxels];
	for(int k=0;k<numGradients;++k){
		//--collect data from gradient k--
		int v=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					img[v]=dwVolume[v*numGradients+k];
				}
			}
		}
		//--denoise--
		//int retVal=volumeDenoising_TGV(img, nrows, ncols, nslices, lambda, denoisingParams, denoised);
		//int retVal=robustVolumeDenoising(img, nrows, ncols, nslices, lambda, EDT_GEMAN_MCCLURE, denoisingParams, denoised);
		//int retVal=robustVolumeDenoisingOR(img, nrows, ncols, nslices, lambda, EDT_GEMAN_MCCLURE, denoisingParams, denoised);
		int retVal=averageVolumeDenoising(img, nrows, ncols, nslices, lambda, denoisingParams, denoised, mask);
		//int retVal=medianVolumeDenoising(img, nrows, ncols, nslices, lambda, denoisingParams, denoised);
		//--put the results back to the input memory--
		v=0;
		for(int s=0;s<nslices;++s){
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					dwVolume[v*numGradients+k]=denoised[v];
				}
			}
		}
	}
	delete[] img;
	delete[] denoised;
	return 0;
}

int denoiseDWVolume_align(double *dwVolume, int nrows, int ncols, int nslices, int numGradients, double *gradientOrientations, double *&aligningErrors, int &nSamples, int &sampleLength){
	int nvoxels=nrows*ncols*nslices;
	double *denoised=new double[nvoxels*numGradients];
	double *signalList=new double[27*3*2*numGradients];//neighSize*3D*2copies(symmetric)*directions
	vector<double *> vAE;
	double ae[27];
	memcpy(denoised, dwVolume, sizeof(double)*nvoxels*numGradients);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int v=s*nrows*ncols+r*ncols+c;
				double *dwOriginal=&dwVolume[v*numGradients];
				cerr<<"["<<s<<", "<<r<<", "<<c<<"]"<<endl;
				//-----
				int nSignals=1;//leave space for central signal
				for(int ds=-1;ds<=1;++ds){
					int ss=s+ds;
					if(!IN_RANGE(ss, 0, nslices)){
						continue;
					}
					for(int dr=-1;dr<=1;++dr){
						int rr=r+dr;
						if(!IN_RANGE(rr, 0, nrows)){
							continue;
						}
						for(int dc=-1;dc<=1;++dc){
							int cc=c+dc;
							if(!IN_RANGE(cc, 0, ncols)){
								continue;
							}

							bool computingCentral=false;
							int prevPosition;
							if((s==ss) && (r==rr) && (c==cc)){
								computingCentral=true;
								prevPosition=nSignals;
								nSignals=0;
							}
							//----pick signal at voxel (ss,rr,cc)----
							int voxPos=ss*nrows*ncols + rr*ncols + cc;
							double *dw=&dwVolume[voxPos*numGradients];
							double *signal=&signalList[3*numGradients*(2*nSignals)];
							double *signalSym=&signalList[3*numGradients*(2*nSignals+1)];
							for(int i=0;i<numGradients;++i){
								signal[3*i  ]=dw[i]*gradientOrientations[3*i];//x
								signal[3*i+1]=dw[i]*gradientOrientations[3*i+1];//y
								signal[3*i+2]=dw[i]*gradientOrientations[3*i+2];//z
								signalSym[3*i  ]=-dw[i]*gradientOrientations[3*i];//x
								signalSym[3*i+1]=-dw[i]*gradientOrientations[3*i+1];//y
								signalSym[3*i+2]=-dw[i]*gradientOrientations[3*i+2];//z
							}
							if(computingCentral){
								nSignals=prevPosition;
							}else{
								++nSignals;
							}
							
						}
					}
				}
				//----align local shapes----
				icp(NULL, signalList, nSignals, 2*numGradients, ae);
				if(nSignals==27){
					double *newAE=new double[26];
					memcpy(newAE, ae, sizeof(double)*26);
					vAE.push_back(newAE);
				}
				//----average(first attempt)----
				//double *dw=&dwVolume[v*numGradients];
				double *dwDenoised=&denoised[v*numGradients];
				double aecp[27];
				memcpy(aecp, ae, sizeof(double)*nSignals);
				sort(aecp, aecp+nSignals);
				double thr=aecp[nSignals/2];
				for(int i=0;i<numGradients;++i){
					double sum=0;
					double *currentSignal=signalList;
					int nAveraged=0;
					for(int j=0;j<nSignals;++j, currentSignal+=3*numGradients*2){
						if((j>0) && (ae[j-1]>thr)){
							continue;
						}
						double rad=SQR(currentSignal[3*i])+SQR(currentSignal[3*i+1])+SQR(currentSignal[3*i+2]);//the norm of the ith point of signal j
						rad=sqrt(rad);
						sum+=rad;
						++nAveraged;
					}
					sum/=nAveraged;
					dwDenoised[i]=sum;
				}
			}
		}
	}
	memcpy(dwVolume, denoised, sizeof(double)*nvoxels*numGradients);
	nSamples=vAE.size();
	sampleLength=26;
	aligningErrors=new double[vAE.size()*sampleLength];
	for(unsigned i=0;i<vAE.size();++i){
		memcpy(&aligningErrors[i*sampleLength], vAE[i], sizeof(double)*sampleLength);
		delete[] vAE[i];
	}
	delete[] denoised;
	delete[] signalList;
	return 0;
}


void testDenoisingMethod(MultiTensorFieldExperimentParameters &params){
	vector<double> results;
	if(!(params.loaded)){
		cerr<<"Error: parameters were not loaded. Call params.loadData()."<<endl;
		return ;
	}
	MultiTensor *voxGT=params.GT->getVoxels();
	int nrows=params.GT->getNumRows();
	int ncols=params.GT->getNumCols();
	int nslices=params.GT->getNumSlices();
	int nvoxels=nrows*ncols*nslices;
	
	//----------generate signals-----------
	double *dwVolume=new double[nvoxels*params.numGradients];
	double *s0Volume=new double[nvoxels];
	double *gt=new double[nrows*ncols*nslices*params.numGradients];
	double *denoised=new double[nrows*ncols*nslices*params.numGradients];
	if(params.inputType==MTFIT_GTGenerator){
		for(int v=0;v<nvoxels;++v){
			s0Volume[v]=1;
			double *S=&dwVolume[v*params.numGradients];
			memset(S, 0, sizeof(double)*params.numGradients);

			double *Sgt=&gt[v*params.numGradients];
			memset(Sgt, 0, sizeof(double)*params.numGradients);
			double sigma=0;
			if(params.SNR>0){
				sigma=1.0/params.SNR;
			}
			voxGT[v].acquireWithScheme(params.b, params.gradients, params.numGradients, 0, Sgt);
			voxGT[v].acquireWithScheme(params.b, params.gradients, params.numGradients, sigma,S);
		}
	}else{
		memcpy(dwVolume, params.dwVolume, sizeof(double)*nvoxels*params.numGradients);
		memcpy(s0Volume, params.s0Volume, sizeof(double)*nvoxels);
	}
	//----------test denoising--------
	const int nsteps=20;
	double minAlpha[2]={0.001,0.001};
	double maxAlpha[2]={0.1,0.1};
	double dAlpha[2]={(maxAlpha[0]-minAlpha[0])/(nsteps-1),(maxAlpha[1]-minAlpha[1])/(nsteps-1)};
	double alpha[2]={minAlpha[0], minAlpha[1]};

	double noisyRMSE=0;
	for(int k=nrows*ncols*nslices*params.numGradients-1;k>=0;--k){
		noisyRMSE+=SQR(gt[k]-dwVolume[k]);
	}
	noisyRMSE/=(nrows*ncols*nslices*params.numGradients);
	noisyRMSE=sqrt(noisyRMSE);
	cerr<<"Noisy:"<<noisyRMSE<<endl;
	alpha[0]=minAlpha[0];
	FILE *F=fopen("denoisingStats.txt", "w");
	fprintf(F, "Noisy: %lf\n", noisyRMSE);
	for(int i=0;i<nsteps;++i,alpha[0]+=dAlpha[0]){
		alpha[1]=minAlpha[1];
		for(int j=0;j<nsteps;++j,alpha[1]+=dAlpha[1]){
			memcpy(denoised, dwVolume, sizeof(double)*nrows*ncols*nslices*params.numGradients);
			denoiseDWVolume(denoised, nrows, ncols, nslices, params.numGradients, 1.0, alpha);
			double rmse=0;
			for(int k=nrows*ncols*nslices*params.numGradients-1;k>=0;--k){
				rmse+=SQR(gt[k]-denoised[k]);
			}
			rmse/=(nrows*ncols*nslices*params.numGradients);
			rmse=sqrt(rmse);
			cerr<<"Alpha[0]:"<<alpha[0]<<".\tAlpha[1]:"<<alpha[1]<<".\tRMSE:"<<rmse<<endl;
			fprintf(F, "%lf\t", rmse);
		}
		fprintf(F, "\n");
	}
	fclose(F);
	delete[] gt;
	delete[] denoised;
	//filterTGV_L2(img, r, c, lambda, params[0], params[1], params[2], params[3], params[4], f, NULL);
	
	
}


void saveMask(unsigned char *mask, int nslices, int nrows, int ncols, unsigned char filterMask, unsigned char filterVal, const char *fname){
	int nvoxels=nslices*nrows*ncols;
	unsigned char *saved=new unsigned char[nvoxels];
	int nonzero=0;
	for(int i=0;i<nvoxels;++i){
		if((mask[i]&filterMask)==filterVal){
			saved[i]=1;
			++nonzero;
		}else{
			saved[i]=0;
		}
	}
	save3DNifti(fname, saved, nslices, nrows, ncols);
	cerr<<"Saved mask with "<<nonzero<<" non-zero voxels"<<endl;
}
vector<double> testMultiTensorField(MultiTensorFieldExperimentParameters &params, RegularizationCallbackFunction callbackFunction, void *callbackData){
	//params.printConfig();
	vector<double> results;
	if(!(params.loaded)){
		cerr<<"Error: parameters were not loaded. Call params.loadData()."<<endl;
		return results;
	}
	MultiTensor *voxGT=NULL;
	if(params.groundTruthType!=MTFGTT_None){
		voxGT=params.GT->getVoxels();
	}
		
	int nrows=params.reconstructed->getNumRows();
	int ncols=params.reconstructed->getNumCols();
	int nslices=params.reconstructed->getNumSlices();
	int nvoxels=nrows*ncols*nslices;
	int maxSparsity=0;
	
	//----------generate signals-----------
	double *dwVolume=NULL;
	double *s0Volume=NULL;
	double *referenceSignal=new double[params.numGradients];
	if(params.inputType==MTFIT_GTGenerator){
		dwVolume=new double[nvoxels*params.numGradients];
		s0Volume=new double[nvoxels];
		double *gtErr=params.GT->getError();
		for(int v=0;v<nvoxels;++v){
			s0Volume[v]=1;
			double *S=&dwVolume[v*params.numGradients];
			memset(S, 0, sizeof(double)*params.numGradients);
			double sigma=0;
			if(params.SNR>0){
				sigma=1.0/params.SNR;
			}
			for(int i=0;i<params.numSamplings;++i){
				voxGT[v].acquireWithScheme(params.b, params.gradients, params.numGradients, sigma,referenceSignal);
				for(int j=0;j<params.numGradients;++j){
					S[j]+=referenceSignal[j];
				}

			}
			for(int j=0;j<params.numGradients;++j){
				S[j]/=params.numSamplings;
			}

			voxGT[v].acquireWithScheme(params.b, params.gradients, params.numGradients, 0,referenceSignal);
			double refErr=0;
			for(int i=0;i<params.numGradients;++i){
				refErr+=SQR(S[i]-referenceSignal[i]);
			}
			refErr=sqrt(refErr/params.numGradients);
			gtErr[v]=refErr;
		}
		//saveDWINifti(string("synthetic.nii"), s0Volume, dwVolume, nslices, nrows, ncols, params.numGradients);
	}else{
		dwVolume=params.dwVolume;
		s0Volume=params.s0Volume;
	}
	cerr<<"Running DBF with "<<params.numGradients<<" gradients."<<endl;
	delete[] referenceSignal;
	//---------estimate diffusivity profile-----


	GDTI H(2, params.b, params.gradients, params.numGradients);
	double longDiffusion=params.longDiffusion;
	double transDiffusion=params.transDiffusion;
	unsigned char *mask_WM=new unsigned char[nslices*nrows*ncols];
	double *GDTI_eVal=new double[3*nvoxels];
	double *GDTI_pdd=new double[3*nvoxels];
	double *fittedTensors=new double[H.getNumCoefficients()*nvoxels];
	if(params.WM_useS0HistogramThresholding){
		Histogram hist(s0Volume,nslices*nrows*ncols,100);
		if(hist.good()){
			params.WM_S0Threshold=hist.getFirstMin();
		}else{
			params.WM_S0Threshold=-1;
		}
	}
	
	H.createMask(s0Volume, dwVolume, nslices, nrows, ncols, fittedTensors, GDTI_eVal, GDTI_pdd, mask_WM, params.WM_FAThreshold, params.WM_S0Threshold, params.WM_AVSignalThreshold);
	unsigned char WMFilter=FA_BIT|S0_BIT|AVSIGNAL_BIT;
	unsigned char WMFilterValue=WMFilter;
	//-----------use LCI--------------
	double *LCI=NULL;
	int ndims;
	int *dims=NULL;
	int *vox=NULL;
	int *layout=NULL;
	int *layout_orientations=NULL;
	double *transform=NULL;
	read_mrtrix_image("challenge_training_SNR30_LCI_filtered.mif", LCI, ndims, dims, vox, layout, layout_orientations, transform);
	delete[] vox;
	delete[] layout;
	delete[] layout_orientations;
	delete[] transform;
	delete[] dims;

#ifdef USE_ALRAM_MASK	
	double *alramMask=NULL;
	int a,b,c;
	loadVolumeFromNifti(ALRAM_MASK_NAME, alramMask, a,b,c);
	for(int i=nslices*nrows*ncols-1;i>=0;--i){
		if((alramMask[i]>0) && (alramMask[i]<2)){
			mask_WM[i]=WMFilterValue|(mask_WM[i]&(FA_HIST_BIT|3));
		}else{
			mask_WM[i]=0;
			alramMask[i]=0;
		}
	}
#endif	
	saveMask(mask_WM, nslices, nrows, ncols, WMFilter, WMFilterValue, "wm_mask.nii");
	//---compute fa field---
	double *current_eval=GDTI_eVal;
	double *FAField=new double[nvoxels];
	for(int v=0;v<nvoxels;++v, current_eval+=3){
		if(!isNumber(current_eval[0])){
			FAField[v]=QNAN64;
		}else{
			FAField[v]=computeFractionalAnisotropy(current_eval);
		}
	}
	//----------------------


	DBF dbfInstance(params.gradients, params.numGradients, params.DBFDirections, params.numDBFDirections);
	dbfInstance.setSolverType(params.solverType);
	if(params.nonparametric){
#ifndef USE_GSL
		cerr<<"Warning: Spherical Deconvolution unsupported. Compile with GSL."<<endl;
		return results;
#else
		double *Phi=dbfInstance.getDiffusionBasis();
		CSDeconv deconv;
		const int lmax=8;
		const int nsh=45;
		double coefs[nsh];
		int retVal=deconv.estimateResponseFunction(s0Volume, dwVolume, nslices, nrows, ncols, mask_WM, 3, 1, GDTI_pdd, params.gradients, params.numGradients, lmax, coefs);
		//retVal=0;
		if(retVal==0){
			cerr<<"Warning: no elongated tensors found. Switching to FA-Histogram criterion."<<endl;
			retVal=deconv.estimateResponseFunction(s0Volume, dwVolume, nslices, nrows, ncols, mask_WM, FA_HIST_BIT, FA_HIST_BIT, GDTI_pdd, params.gradients, params.numGradients, lmax, coefs);
			if(retVal==0){
				cerr<<"Error: no high-FA tensors found. Aborting."<<endl;
				return results;
			}
			//compute diffusivities anyway
			double avProfile[3];
			retVal=H.computeAverageProfile(GDTI_eVal, nvoxels, mask_WM, FA_HIST_BIT, FA_HIST_BIT, avProfile);
			longDiffusion=avProfile[2];
			transDiffusion=0.5*(avProfile[0]+avProfile[1]);
			cerr<<"Estimated response function with "<<retVal<<" samples (FA-Histogram)"<<endl;
			cerr<<"Estimated diffusivity profile with "<<retVal<<" samples (FA-Histogram): ["<<longDiffusion<<", "<<transDiffusion<<"]"<<endl;
		}else{
			//compute diffusivities anyway
			double avProfile[3];
			retVal=H.computeAverageProfile(GDTI_eVal, nvoxels, mask_WM, 3,1, avProfile);//try with elongated tensors
			longDiffusion=avProfile[2];
			transDiffusion=0.5*(avProfile[0]+avProfile[1]);
			cerr<<"Estimated response function with "<<retVal<<" samples (elongated tensors)"<<endl;
			cerr<<"Estimated diffusivity profile with "<<retVal<<" samples (elongated tensors): ["<<longDiffusion<<", "<<transDiffusion<<"]"<<endl;
		}
		//---------
		vector<double> responseCoefs(1+lmax/2);
		cerr<<"Response coeffs:"<<endl;
		for(int i=0;i<=lmax;i+=2){
			int idx=SphericalHarmonics::index(i, 0);
			responseCoefs[i/2]=coefs[idx];
			cerr<<responseCoefs[i/2]<<" ";
		}
		cerr<<endl;
		//---------

		cerr<<"Using "<<retVal<<" signals for response function estimation."<<endl;
		FILE *FSH=fopen("response.txt", "w");
		for(int i=0;i<nsh;++i){
			fprintf(FSH, "%0.15lf ", coefs[i]);
		}
		fclose(FSH);
		double *rotatedGradients=new double[3*params.numGradients];
		double *amplitudes=new double[params.numGradients];
		double zaxis[]={0,0,1};
		double rotation[9];
		for(int j=0;j<params.numDBFDirections;++j){
			//rotate the scheme according to each DBF direction
			fromToRotation(&params.DBFDirections[3*j], zaxis, rotation);
			for(int i=0;i<params.numGradients;++i){
				multMatrixVector(rotation, &params.gradients[3*i],3,3,&rotatedGradients[3*i]);
			}
			SHEvaluator evaluator(rotatedGradients, params.numGradients,lmax);
			evaluator.evaluateFunction_amplitudes(coefs, nsh, lmax, amplitudes);
			for(int i=0;i<params.numGradients;++i){
				Phi[i*params.numDBFDirections+j]=amplitudes[i];
			}
		}
		delete[] rotatedGradients;
		delete[] amplitudes;
#endif
	}else{
		if(params.estimateDiffusivities){
			double avProfile[3];
			int retVal=H.computeAverageProfile(GDTI_eVal, nvoxels, mask_WM, 3,1, avProfile);//try with elongated tensors
			//retVal=0;
			if(retVal==0){
				cerr<<"Warning: no elongated tensors found. Switching to FA-Histogram criterion."<<endl;
				retVal=H.computeAverageProfile(GDTI_eVal, nvoxels, mask_WM, FA_HIST_BIT, FA_HIST_BIT, avProfile);//use high fa-index tensors
				if(retVal==0){
					cerr<<"Error: no high-FA tensors found. Aborting."<<endl;
					return results;
				}
				longDiffusion=avProfile[2];
				transDiffusion=0.5*(avProfile[0]+avProfile[1]);
				cerr<<"Estimated diffusivity profile with "<<retVal<<" samples (FA-Histogram): ["<<longDiffusion<<", "<<transDiffusion<<"]"<<endl;
				saveMask(mask_WM, nslices, nrows, ncols, FA_HIST_BIT, FA_HIST_BIT, "response_mask.nii");
			}else{
				longDiffusion=avProfile[2];
				transDiffusion=0.5*(avProfile[0]+avProfile[1]);
				cerr<<"Estimated diffusivity profile with "<<retVal<<" samples (elongated tensors): ["<<longDiffusion<<", "<<transDiffusion<<"]"<<endl;
				saveMask(mask_WM, nslices, nrows, ncols, 3, 1, "response_mask.nii");
			}
			
		}
#ifdef THIN_PROFILE
		transDiffusion*=0.5;
		longDiffusion*=2;	
#endif
		dbfInstance.reComputeDiffusionBasisFunctions(params.b, longDiffusion,transDiffusion);
	}
	

	double tau=0.5/sqrt(12.0);
	double sigma=0.5/sqrt(12.0);
	double theta=1;

	double param[5]={params.alpha0, params.alpha1, tau, sigma, theta};
	
	if(params.spatialSmoothing){
		cerr<<"Smoothing...";
		//-----aligning-based smoothing----
		/*double *aligningErrors=NULL; 
		int nSamples=0;
		int sampleLength=0;
		#ifdef USE_GSL
			CSDeconv deconv;
			deconv.denoiseDWVolume_align(s0Volume, dwVolume, nrows, ncols, nslices, params.numGradients, params.gradients, aligningErrors, nSamples, sampleLength);
			//denoiseDWVolume_align(dwVolume, nrows, ncols, nslices, params.numGradients, params.gradients, aligningErrors, nSamples, sampleLength);
		#else
			denoiseDWVolume_align(dwVolume, nrows, ncols, nslices, params.numGradients, params.gradients, aligningErrors, nSamples, sampleLength);
		#endif
		
		FILE *AEF=fopen("alignmentErrors.txt", "w");
		for(int i=0;i<nSamples;++i){
			for(int j=0;j<sampleLength;++j){

				fprintf(AEF, "%0.15lf\t", aligningErrors[i*sampleLength+j]);
			}
			fprintf(AEF, "\n");
		}
		fclose(AEF);
		delete[] aligningErrors;
		cerr<<"done."<<endl;*/
		//---------------------------------
		unsigned char *denoisingMask=new unsigned char[nvoxels];
		for(int i=0;i<nvoxels;++i){
			if((mask_WM!=NULL) && ((mask_WM[i]&WMFilter)!=WMFilterValue)){
					denoisingMask[i]=0;
			}else{
				denoisingMask[i]=1;
			}
		}
		//save3DNifti(string("WMmask.nii"), denoisingMask, nslices, nrows, ncols);
		denoiseDWVolume(dwVolume, nrows, ncols, nslices, params.numGradients, params.lambda, param, denoisingMask);
		//int droppedByFA=selectiveDenoiseDWVolume(dwVolume, nrows, ncols, nslices, params.numGradients, denoisingMask, FAField, 2.2);
		//cerr<<"Dropped by FA:"<<droppedByFA<<endl;

		delete[] denoisingMask;
		saveDWINifti(string("denoised.nii"), s0Volume, dwVolume, nslices, nrows, ncols, params.numGradients);
		double *denoised=new double[nrows*ncols*nslices*params.numGradients];
		double *lineProc=new double[nrows*ncols*nslices*6];
		//robust3DVectorFieldDenoisingOR(dwVolume, nrows,ncols, nslices, params.numGradients, params.lambda, EDT_TUKEY_BIWEIGHT, param, denoised, lineProc);
		//memcpy(dwVolume, denoised, sizeof(double)*nrows*ncols*nslices*params.numGradients);
		delete[] denoised;
		//showDWVolumeSlices("DW Denoised", dwVolume, nrows, ncols, nslices, params.numGradients);
		//showDWVolumeSlices("LP", lineProc, nrows, ncols, nslices, 6);
	}
	
	MultiTensor *voxE=params.reconstructed->getVoxels();
	double *rec_error=params.reconstructed->getError();

	if(params.fitGDTIOnly){
		for(int v=0;v<nvoxels;++v){
			if(params.groundTruthType!=MTFGTT_None){
				if(voxGT[v].getNumCompartments()<1){
					continue;
				}
			}else if(s0Volume[v]<params.WM_S0Threshold){
				continue;
			}
			voxE[v].dellocate();
			voxE[v].allocate(1);
			voxE[v].setDiffusivities(0, GDTI_eVal[3*v+0], GDTI_eVal[3*v+1], GDTI_eVal[3*v+2]);
			voxE[v].setVolumeFraction(0,1);
			voxE[v].setRotationMatrixFromPDD(0,&GDTI_pdd[3*v]);
		}
		if(params.inputType==MTFIT_GTGenerator){
			delete[] s0Volume;
			delete[] dwVolume;
		}
		
		
		delete[] GDTI_eVal;
		delete[] GDTI_pdd;
		if(params.groundTruthType!=MTFGTT_None){
			generateErrorSummary(*params.GT, *params.reconstructed, params.ODFDirections, params.numODFDirections, results, true, true);
		}
		delete[] mask_WM;
		return results;
	}else if(params.iterativeBasisUpdate){
		double *Phi=NULL;
		vector<set<int> > neighborhoods;
		buildNeighborhood(params.DBFDirections, params.numDBFDirections, params.clusteringNeighSize, neighborhoods);

		double *signal=new double[params.numGradients];
		tic();
		for(int v=0;v<nvoxels;++v){
			cerr<<v<<endl;
			REPORT_PROGRESS(v,nvoxels,cerr);
			/*if(params.groundTruthType!=MTFGTT_None){
				if(voxGT[v].getNumCompartments()<1){
					continue;
				}
			}else if(s0Volume[v]<params.maskThreshold){
				continue;
			}*/
			/*if(!(mask_WM[v])){
				continue;
			}*/
			if((mask_WM!=NULL) && ((mask_WM[v]&WMFilter)!=WMFilterValue)){
				voxE[v].setAlpha(NULL,0);
				continue;
			}
			memcpy(signal, &dwVolume[v*params.numGradients], sizeof(double)*params.numGradients);
			for(int i=0;i<params.numGradients;++i){
				signal[i]/=s0Volume[v];
			}
			voxE[v].fitDBFToSignal(signal,H,params.DBFDirections,params.numDBFDirections,params.b, longDiffusion, transDiffusion,params.iterateDiffProf);
			//voxE[v].fitDBFToSignal_baseline(S,H,params.DBFDirections,Phi, params.numDBFDirections, longDiffusion, transDiffusion,neighborhoods,params.bigPeaksThreshold);
			/*double lastDiffProf[3]={transDiffusion, transDiffusion, longDiffusion};
			double diffProffError=1;
			const int maxDiffIter=1;
			int diffIter=0;
			while((1e-6<diffProffError) && diffIter<maxDiffIter){
				++diffIter;
				voxE[v].fitDBFToSignal(signal,H,params.DBFDirections,params.numDBFDirections,params.b, lastDiffProf[2], lastDiffProf[0],params.iterateDiffProf);
				double *diffProf=voxE[v].getDiffusivities(0);
				diffProffError=MAX(fabs(diffProf[0]-lastDiffProf[0]), fabs(diffProf[1]-lastDiffProf[1]));
				diffProffError=MAX(diffProffError, fabs(diffProf[2]-lastDiffProf[2]));
				memcpy(lastDiffProf, diffProf, sizeof(double)*3);
			}*/
		}
		toc();
		delete[] signal;
		if(Phi!=NULL){
			delete[] Phi;
		}
		if(params.groundTruthType!=MTFGTT_None){
			generateErrorSummary(*params.GT, *params.reconstructed, params.ODFDirections, params.numODFDirections, results, true, true);
			vector<double > errorAngularPrecision;
			evaluateAngularPrecision(	*params.GT, *params.reconstructed, errorAngularPrecision);
			double maxErr=0;
			for(unsigned v=0;v<errorAngularPrecision.size();++v){
				rec_error[v]=errorAngularPrecision[v];
				maxErr=MAX(maxErr, rec_error[v]);
			}
			sort(errorAngularPrecision.begin(), errorAngularPrecision.end());
			cerr<<"Minimum angular error:\t"<<errorAngularPrecision[0]<<endl;
			cerr<<"Maximum angular error:\t"<<errorAngularPrecision[errorAngularPrecision.size()-1]<<endl;
		}
		delete[] mask_WM;
		return results;
	}
	//----------initialize DBF solver----
	double *RES_pdds=new double[3*(params.numDBFDirections+1)];
	double *RES_amount=new double[params.numDBFDirections+1];
	
	//======================================================
	//=====================run test=========================
	//======================================================
	double *Phi=dbfInstance.getDiffusionBasis();
	/*double *pseudoInverse=new double[params.numGradients*params.numDBFDirections];
	int retval=computePseudoInverseRMO(Phi, params.numGradients, params.numDBFDirections, pseudoInverse);*/


	double *Sx=new double[params.numGradients];
	//double *alphaVolume=new double[nvoxels*(params.numDBFDirections+1)];
	double *Phix=new double[params.numGradients*(params.numDBFDirections+1)];
	for(int i=0;i<params.numGradients;++i){
		memcpy(&Phix[i*(params.numDBFDirections+1)], &Phi[i*params.numDBFDirections], sizeof(double)*params.numDBFDirections);
		Phix[i*(params.numDBFDirections+1) + params.numDBFDirections]=0;
	}
	double newDir[3]={1,1,1};
	dbfInstance.computeDiffusionFunction(newDir, transDiffusion, transDiffusion, transDiffusion, Phix+params.numDBFDirections, params.numDBFDirections+1);
	int dropped=0;
	double mostNegativeError=0;
	int totalVoxels=nslices*nrows*ncols;
	FILE *F=fopen("diff_stats.txt","w");
	int ceCnt=0;
	
	
	if(mask_WM!=NULL){
		int nonEmptyVoxels=0;
		for(int i=nslices*nrows*ncols-1;i>=0;--i){
			if((mask_WM[i]&WMFilter)==WMFilterValue){
				++nonEmptyVoxels;
			}
			
		}
		cerr<<"Non-empty voxels:"<<nonEmptyVoxels<<endl;
#ifdef USE_ALRAM_MASK
		int pp=0,nn=0,pn=0,np=0;
		for(int i=nslices*nrows*ncols-1;i>=0;--i){
			if((mask_WM[i]&WMFilter)==WMFilterValue){
				++nonEmptyVoxels;
				if(alramMask[i]>0){
					++pp;
					mask_WM[i]=WMFilterValue;
				}else{
					mask_WM[i]=0;
					++pn;
				}
			}else{
				if(alramMask[i]>0){
					mask_WM[i]=WMFilterValue;
					++np;
				}else{
					mask_WM[i]=0;
					++nn;
				}
			}
			
		}
		cerr<<"PP: "<<pp<<". NN: "<<nn<<". PN: "<<pn<<". NP: "<<np<<endl;
#endif
	}
	double *currentAlpha=new double[params.numDBFDirections+1];
	tic();
	vector<double> verr;
	double *rec=new double[MAX(params.numGradients, params.numDBFDirections)];
	double *errMask=new double[nvoxels];
	for(int s=0;s<nslices;++s){
		for(int c=0;c<ncols;++c){
			for(int r=0;r<nrows;++r){
				int vox=s*(nrows*ncols)+r*ncols+c;
				errMask[vox]=0;
				REPORT_PROGRESS(vox,totalVoxels,cerr);
				/*if(params.groundTruthType!=MTFGTT_None){
					if(voxGT[vox].getNumCompartments()==0){
						voxE[vox].setAlpha(NULL,0);
						continue;
					}
				}else */
				/*if(LCI[vox]>LCI_THR){
					voxE[vox].setAlpha(NULL,0);
					continue;
				}*/
				if((mask_WM!=NULL) && ((mask_WM[vox]&WMFilter)!=WMFilterValue)){
					voxE[vox].setAlpha(NULL,0);
					continue;
				}
				//gt_error[vox]=params.GT->getVoxels()[vox].getMinAngleDegrees();
				double *evalVox=&GDTI_eVal[3*vox];
				double faVox=computeFractionalAnisotropy(evalVox, (evalVox[0]+evalVox[1]+evalVox[2])/3.0);
				//double *currentAlpha=NULL;
				//currentAlpha=&alphaVolume[vox*params.numDBFDirections];
				memcpy(Sx, &dwVolume[vox*params.numGradients], sizeof(double)*params.numGradients);
				for(int i=0;i<params.numGradients;++i){
					Sx[i]/=s0Volume[vox];
				}
				//===============reconstruct multi-tensor=================
				if(params.fitWithNeighbors){
					int ncount=1;
					int bestNeigh=-1;
					for(int k=0;k<MTFE_NEIGH_SIZE;++k){
						int ss=s+MTFE_dSlice[k];
						int rr=r+MTFE_dRow[k];
						int cc=c+MTFE_dCol[k];
						if(IN_RANGE(ss,0,nslices) && IN_RANGE(rr,0,nrows) && IN_RANGE(cc,0,ncols)){
							int neighPos=ss*(nrows*ncols)+rr*ncols+cc;
							double *evalNeigh=&GDTI_eVal[3*neighPos];
							double faNeigh=computeFractionalAnisotropy(evalNeigh, (evalNeigh[0]+evalNeigh[1]+evalNeigh[2])/3.0);
							double *Sneigh=&dwVolume[neighPos*params.numGradients];
							if(params.useFAIndicator && (fabs(faVox-faNeigh)>0.1)){
								params.reconstructed->setDualLattice(vox, neighPos, fabs(faVox-faNeigh));
								++dropped;
							}else{
								linCombVector<double>(Sx, 1.0, Sneigh, params.numGradients, Sx);
								++ncount;
							}
						}
					}
					multVectorScalar<double>(Sx,1.0/double(ncount), params.numGradients,Sx);
				}
				double errorNNLS=0;
				nnls(Phi, params.numGradients, params.numDBFDirections, Sx, currentAlpha, &errorNNLS);
				//***experimental: OLS**
				/*multMatrixVector(pseudoInverse, Sx, params.numDBFDirections, params.numGradients,currentAlpha);
				//---compute error---
				multMatrixVector(Phi, currentAlpha, params.numGradients, params.numDBFDirections, rec);
				for(int i=0;i<params.numGradients;++i){
					errorNNLS+=SQR(rec[i]-Sx[i]);
				}*/
				//**********************
				verr.push_back(errorNNLS);
				errMask[vox]=1;
				//---------------------
				//nnls(Phix, params.numGradients, params.numDBFDirections+1, Sx, currentAlpha, &errorNNLS);
				voxE[vox].setAlpha(currentAlpha, params.numDBFDirections);
			}
		}
	}
	fclose(F);
	delete[] rec;
	save3DNifti("errMask.nii", errMask, nslices, nrows, ncols);
	double meanErr=0;
	for(unsigned i=0;i<verr.size();++i){
		meanErr+=verr[i];
	}
	meanErr/=verr.size();
	double stdevErr=0;
	for(unsigned i=0;i<verr.size();++i){
		stdevErr+=SQR(verr[i]-meanErr);
	}
	stdevErr/=(verr.size()-1);
	cerr<<"Voxels considered: "<<verr.size()<<endl;
	cerr<<"Mean error: "<<meanErr<<". Stdev: "<<stdevErr<<endl;

	if(params.saveODFField){
		double lambda[3]={transDiffusion, transDiffusion, longDiffusion};
		cerr<<"Saving ODF Field..."<<endl;
		FILE *fodf=fopen((string(params.reconstructionFname)+".odf").c_str(), "wb");
		fwrite(&nslices, sizeof(int), 1, fodf);
		fwrite(&nrows, sizeof(int), 1, fodf);
		fwrite(&ncols, sizeof(int), 1, fodf);
		fwrite(&params.numODFDirections, sizeof(int), 1, fodf);		
		params.reconstructed->saveODFFieldFromAlpha(params.ODFDirections, params.numODFDirections, params.DBFDirections, params.numDBFDirections,lambda, fodf);
		fclose(fodf);
		cerr<<"done."<<endl;
	}
	cerr<<"Dropped by FA:"<<dropped<<endl;
	cerr<<"Most negative error difference: "<<mostNegativeError<<endl;
	delete[] Sx;
	delete[] Phix;
	//delete[] pseudoInverse;
	//---regularize---
	/*if(params.regularizeAlphaField){
		int retVal=regularizeDBF_alram(alphaVolume, *params.reconstructed, longDiffusion, transDiffusion, fittedTensors,dbfInstance, dwVolume, s0Volume, nslices, nrows, ncols, callbackFunction, callbackData);
	}*/
	//---build MultiTensor instance from the solution---
	vector<set<int> > neighborhoods;
	cerr<<"Grouping alpha coefficients...";
	buildNeighborhood(params.DBFDirections, params.numDBFDirections, params.clusteringNeighSize, neighborhoods);
	int nGrouped=0;
	int furtherDropped=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int v=s*(nrows*ncols)+r*ncols+c;
				if((mask_WM!=NULL) && ((mask_WM[v]&WMFilter)!=WMFilterValue)){
					continue;
				}
				if(voxE[v].getAlpha()==NULL){
					continue;
				}
				/*if(LCI[v]>LCI_THR){
					voxE[v].allocate(1);
					voxE[v].setRotationMatrixFromPDD(0,&GDTI_pdd[3*v]);
					voxE[v].setDiffusivities(0,transDiffusion, transDiffusion, longDiffusion);
					voxE[v].setVolumeFraction(0,1.0);
					continue;
				}*/
				
				//double *currentAlpha=NULL;
				//currentAlpha=&alphaVolume[v*params.numDBFDirections];
				memcpy(currentAlpha, voxE[v].getAlpha(), sizeof(double)*params.numDBFDirections);
				groupCoefficients(currentAlpha, params.DBFDirections, params.numDBFDirections, neighborhoods, params.bigPeaksThreshold, transDiffusion, longDiffusion, voxE[v]);
				if(voxE[v].getNumCompartments()>0){
					++nGrouped;
				}else{
					++furtherDropped;
				}
				
			}
		}
	}
	cerr<<"done."<<endl;
	cerr<<"Evaluating...";
	//cerr<<"Grouped: "<<nGrouped<<endl;
	//cerr<<"Dropped (from grouping): "<<furtherDropped<<endl;
			
#ifdef RUN_GRID_SEARCH
	double meanResults[5][thrSteps*neighSizeSteps];
	double stdevResults[5][thrSteps*neighSizeSteps];
	params.bigPeaksThreshold=thrStart;
	vector<string> rowHeaders;
	vector<string> columnHeaders;
	for(int ithr=0;ithr<thrSteps;++ithr, params.bigPeaksThreshold+=thrDelta){
		ostringstream os;
		os<<setprecision(2)<<params.bigPeaksThreshold;
		rowHeaders.push_back(os.str());
		params.clusteringNeighSize=neighSizeStart;
		for(int ins=0;ins<neighSizeSteps;++ins, params.clusteringNeighSize+=neighSizeDelta){
			cerr<<"Test "<<ithr*neighSizeSteps+ins<<"/"<<thrSteps*neighSizeSteps<<endl;
			if(ithr==0){
				ostringstream os2;
				os2<<setprecision(2)<<params.clusteringNeighSize;
				columnHeaders.push_back(os2.str());
			}
			
			//--------build neighborhood-----
#endif
#ifdef DEPRECATED
			//-------prepare diffusion directions-----
			buildNeighborhood(params.DBFDirections, params.numDBFDirections, params.clusteringNeighSize, neighborhoods);
			double *dirx=new double[3*(params.numDBFDirections+1)];
			memcpy(dirx, params.DBFDirections, sizeof(double)*3*params.numDBFDirections);
			//----------------------------------------
			if(params.useGDTIPostProcessing){
				double splitCount=0;
				for(int v=0;v<nvoxels;++v)if(voxGT[v].getNumCompartments()>0){
					double *currentAlpha=&alphaVolume[v*params.numDBFDirections];
					voxE[v].setAlpha(currentAlpha,params.numDBFDirections);

					vector<set<int> > groups;
					groupCoefficientsGDTI(H, longDiffusion, transDiffusion, currentAlpha, dirx, params.numDBFDirections, neighborhoods, voxE[v], groups);

					voxE[v].dropSmallPeaks(params.bigPeaksThreshold);
					//----
					rec_error[v]=0;
					
					for(int i=0;i<voxE[v].getNumCompartments();++i){
						double *lambda=voxE[v].getDiffusivities();
						//double fa=computeFractionalAnisotropy(lambda);
						double opc=lambda[1]/lambda[0];
						//rec_error[v]=MAX(rec_error[v],opc);
						if(1.1*lambda[0]<lambda[1]){
							rec_error[v]=1;
							if(params.applyTensorSplitting){
								voxE[v].split(H,params.DBFDirections, dbfInstance.getDiffusionBasis());
								++splitCount;
								break;
							}
						}
					}
				}
				splitCount=splitCount;
			}else{
				//for(int v=0;v<nvoxels;++v){
				for(int s=0;s<nslices;++s)for(int r=0;r<nrows;++r)for(int c=0;c<ncols;++c){
					int v=s*(nrows*ncols)+r*ncols+c;
					if(params.groundTruthType!=MTFGTT_None){
						if(voxGT[v].getNumCompartments()<1){
							continue;
						}
					}else if(s0Volume[v]<params.WM_S0Threshold){
						continue;
					}
					double *currentAlpha=NULL;
					currentAlpha=&alphaVolume[v*params.numDBFDirections];
					groupCoefficients(currentAlpha, params.DBFDirections, params.numDBFDirections, neighborhoods, params.bigPeaksThreshold, transDiffusion, longDiffusion, voxE[v]);
				}
			}


			if(params.groundTruthType!=MTFGTT_None){
				double *S=new double[params.numGradients];
				double *gtErr=params.GT->getError();
				for(int v=0;v<nvoxels;++v){
					voxE[v].acquireWithScheme(params.b, params.gradients, params.numGradients, 0,S);
					double rmse=0;
					for(int i=0;i<params.numGradients;++i){
						rmse+=SQR(S[i]-dwVolume[v*params.numGradients+i]);
					}
					rmse=sqrt(rmse/params.numGradients);
				}
				delete[] S;
				delete[] dirx;
			}
#endif			
			//-----print errors---
#ifdef RUN_GRID_SEARCH
			generateErrorSummary(*params.GT, *params.reconstructed, params.ODFDirections, params.numODFDirections, results, false);
			meanResults[0][ithr*neighSizeSteps+ins]=results[0];
			meanResults[1][ithr*neighSizeSteps+ins]=results[1];
			meanResults[2][ithr*neighSizeSteps+ins]=results[0]+results[1];
			meanResults[3][ithr*neighSizeSteps+ins]=results[2];
			meanResults[4][ithr*neighSizeSteps+ins]=results[3];
			stdevResults[0][ithr*neighSizeSteps+ins]=results[4];
			stdevResults[1][ithr*neighSizeSteps+ins]=results[5];
			stdevResults[2][ithr*neighSizeSteps+ins]=results[4]+results[5];
			stdevResults[3][ithr*neighSizeSteps+ins]=results[6];
			stdevResults[4][ithr*neighSizeSteps+ins]=results[7];
		}
	}
	string gtFname=getFileNameOnly(params.gtFname);
	ostringstream os[5];
	os[0]<<gtFname<<"_SNR"<<setfill('0')<<setw(4)<<params.SNR<<"_n+.txt";
	os[1]<<gtFname<<"_SNR"<<setfill('0')<<setw(4)<<params.SNR<<"_n-.txt";
	os[2]<<gtFname<<"_SNR"<<setfill('0')<<setw(4)<<params.SNR<<"_n.txt";
	os[3]<<gtFname<<"_SNR"<<setfill('0')<<setw(4)<<params.SNR<<"_Angular.txt";
	os[4]<<gtFname<<"_SNR"<<setfill('0')<<setw(4)<<params.SNR<<"_ODF.txt";
	for(int i=0;i<5;++i){
		saveResults(meanResults[i], thrSteps, neighSizeSteps, rowHeaders, columnHeaders, params.outputDir, string("mean_")+os[i].str());
		saveResults(stdevResults[i], thrSteps, neighSizeSteps, rowHeaders, columnHeaders, params.outputDir, string("stdev_")+os[i].str());
	}
#else
			if(params.groundTruthType!=MTFGTT_None){
				generateErrorSummary(*params.GT, *params.reconstructed, params.ODFDirections, params.numODFDirections, results, true, true);
			}
			
#endif
	//--------------------
	cerr<<"done."<<endl;
	toc();
	
	delete[] mask_WM;
	delete[] GDTI_eVal;
	delete[] GDTI_pdd;
	delete[] fittedTensors;
	//delete[] alphaVolume;
	delete[] currentAlpha;
	delete[] RES_pdds;
	delete[] RES_amount;
	if(params.inputType==MTFIT_GTGenerator){
		delete[] dwVolume;
		delete[] s0Volume;
	}
	return results;
}

void common_evaluate(MultiTensorField &GT, MultiTensorField &recovered, FILE *FNegative, FILE *FPositive, FILE *FAngular){
	int nr=GT.getNumRows();
	int nc=GT.getNumCols();
	int ns=GT.getNumSlices();
	int nvox=nr*nc*ns;
	vector<double > errorAngularPrecision;
	vector<double > errorNegativeCC;
	vector<double > errorPositiveCC;
	if(FNegative!=NULL){
		evaluateNegativeCompartmentCount(GT, recovered, errorNegativeCC);
		for(unsigned v=0;v<errorNegativeCC.size();++v){
			fprintf(FNegative, "%lf\t",errorNegativeCC[v]);
		}
		fprintf(FNegative, "\n");
	}
	if(FPositive!=NULL){
		evaluatePositiveCompartmentCount(GT, recovered, errorPositiveCC);
		for(unsigned v=0;v<errorPositiveCC.size();++v){
			fprintf(FPositive, "%lf\t",errorPositiveCC[v]);
		}
		fprintf(FPositive, "\n");
	}
	if(FAngular!=NULL){
		evaluateAngularPrecision(GT, recovered, errorAngularPrecision);
		for(unsigned v=0;v<errorAngularPrecision.size();++v){
			fprintf(FAngular, "%lf\t",errorAngularPrecision[v]);
		}
		fprintf(FAngular, "\n");
	}
}

void common_evaluate(MultiTensorField &GT, MultiTensorField &recovered){
	int nr=GT.getNumRows();
	int nc=GT.getNumCols();
	int ns=GT.getNumSlices();
	int nvox=nr*nc*ns;
	double CM[5*5];
	MultiTensor *voxGT=GT.getVoxels();
	MultiTensor *voxE=recovered.getVoxels();
	double *rec_error=recovered.getError();
	memset(CM, 0, sizeof(CM));
	int maxNAlpha=0;
	double maxIntra=-1;
	double minInter=180;
	FILE *Fintra=fopen("intraStats.txt","w");
	FILE *Finter=fopen("interStats.txt","w");
	for(int v=0;v<nvox;++v){
		int a=voxGT[v].getNumCompartments();
		int b=voxE[v].getNumAlpha();
		//int b=voxE[v].numLeavesContentionTree(1500);
		rec_error[v]=b;
		maxNAlpha=MAX(maxNAlpha, b);
		a=MIN(a,4);
		b=MIN(b,4);
		CM[a*4+b]++;

		double opcMaxIntra=voxE[v].getMaxIntraAngleDegrees();
		double opcMinInter=voxE[v].getMinAngleDegrees();
		fprintf(Fintra, "%E\n", opcMaxIntra);
		fprintf(Finter, "%E\n", opcMinInter);
		maxIntra=MAX(maxIntra, opcMaxIntra);
		minInter=MIN(minInter, opcMinInter);
		//rec_error[v]=opcMaxIntra;
	}
	fclose(Fintra);
	fclose(Finter);
	cerr<<"Max #alpha="<<maxNAlpha<<endl;
	cerr<<"Min inter: "<<minInter<<endl;
	cerr<<"Max intra: "<<maxIntra<<endl;

	int *alphaCount=new int[maxNAlpha+1];
	memset(alphaCount, 0, sizeof(int)*(maxNAlpha+1));
	for(int v=0;v<nvox;++v){
		int b=voxE[v].getNumAlpha();
		//int b=voxE[v].numLeavesContentionTree(1500);
		alphaCount[b]++;
	}
	for(int i=1;i<=maxNAlpha;++i){
		cerr<<alphaCount[i]<<endl;
	}
	delete[] alphaCount;

	double *ODFDirections=NULL;
	int numODFDirections=0;
	vector<double> results;
	generateErrorSummary(GT, recovered, ODFDirections, numODFDirections, results, true, true);
	vector<double > errorAngularPrecision;
	evaluateAngularPrecision(GT, recovered, errorAngularPrecision);
	double maxErr=0;
	
	/*for(unsigned v=0;v<errorAngularPrecision.size();++v){
		rec_error[v]=errorAngularPrecision[v];
		maxErr=MAX(maxErr, rec_error[v]);
	}*/
	for(int i=1;i<=4;++i){
		for(int j=1;j<=4;++j){
			cerr<<CM[i*4+j]<<"\t";
		}
		cerr<<endl;
	}
	FILE *Fangles=fopen("angle_beta.txt","w");
	double maxDiff=0;
	int cnt=0;
	int total=0;
	for(int v=0;v<nvox;++v){
		double *basisDirs=voxE[v].getDirections();
		double *beta=voxE[v].getAlpha();
		int numDirs=voxE[v].getNumAlpha();
		int numCmp=voxGT[v].getNumCompartments();
		double pdd[3];
		voxGT[v].getPDD(0,pdd);
		for(int i=0;i<numDirs;++i){
			double angle_i=getAbsAngleDegrees(pdd,&basisDirs[3*i],3);
			fprintf(Fangles, "%lf\t%lf\n",angle_i, beta[i]);
			//--conjecture 1 verification
			
			for(int j=i+1;j<numDirs;++j){
				++total;
				double angle_j=getAbsAngleDegrees(pdd,&basisDirs[3*j],3);
				/*if(angle_j<angle_i){
					if(beta[j]<beta[i]){
						cerr<<"Counter example: ("<<angle_i<<","<<angle_j<<")-->["<<beta[i]<<","<<beta[j]<<"]"<<endl;
						maxDiff=MAX(maxDiff,beta[i]-beta[j]);
						++cnt;
					}
				}else if(angle_j>angle_i){
					if(beta[j]>beta[i]){
						cerr<<"Counter example: ("<<angle_i<<","<<angle_j<<")-->["<<beta[i]<<","<<beta[j]<<"]"<<endl;
						maxDiff=MAX(maxDiff,beta[i]-beta[j]);
						++cnt;
					}
				}*/
				if((beta[i]<=0) && (beta[j]>0)){
					if(angle_i<angle_j){
						cerr<<"Counter example: ("<<angle_i<<","<<angle_j<<")-->["<<beta[i]<<","<<beta[j]<<"]"<<endl;
						maxDiff=MAX(maxDiff,beta[i]-beta[j]);
						++cnt;
					}
				}else if((beta[j]<=0) && (beta[i]>0)){
					if(angle_j<angle_i){
						cerr<<"Counter example: ("<<angle_i<<","<<angle_j<<")-->["<<beta[i]<<","<<beta[j]<<"]"<<endl;
						maxDiff=MAX(maxDiff,beta[i]-beta[j]);
						++cnt;
					}
				}
			}
		}
	}
	cerr<<"Count: "<<cnt<<"/"<<total<<endl;
	cerr<<"Max diff: "<<maxDiff<<endl;
	fclose(Fangles);
}


void MultiTensorFieldExperimentParameters::configFromFile(const char *fname){
	ConfigField vars[]={
		"inputType", &inputType, ConfigField::CFFT_Int,
		"groundTruthType", &groundTruthType, ConfigField::CFFT_Int,
		"outputDir", outputDir, ConfigField::CFFT_String,
		"gtFname", gtFname, ConfigField::CFFT_String,
		"solutionFname", solutionFname, ConfigField::CFFT_String,
		"solutionListFname", solutionListFname, ConfigField::CFFT_String,
		"gradientDirsFname", gradientDirsFname, ConfigField::CFFT_String,
		"basisDirsFname", basisDirsFname, ConfigField::CFFT_String,
		"ODFDirFname", ODFDirFname, ConfigField::CFFT_String,
		"b", &b, ConfigField::CFFT_Double,
		"denoisingTest", &denoisingTest, ConfigField::CFFT_Bool, 
		"fitWithNeighbors", &fitWithNeighbors, ConfigField::CFFT_Bool,
		"fitGDTIOnly", &fitGDTIOnly, ConfigField::CFFT_Bool,
		"useFAIndicator", &useFAIndicator, ConfigField::CFFT_Bool,
		"useGDTIPostProcessing", &useGDTIPostProcessing, ConfigField::CFFT_Bool,
		"applyTensorSplitting", &applyTensorSplitting, ConfigField::CFFT_Bool,
		"iterativeBasisUpdate", &iterativeBasisUpdate, ConfigField::CFFT_Bool,
		"iterateDiffProf", &iterateDiffProf, ConfigField::CFFT_Bool,
		"evaluate", &evaluate, ConfigField::CFFT_Bool,
		"evaluateList", &evaluateList, ConfigField::CFFT_Bool,
		"nonparametric", &nonparametric, ConfigField::CFFT_Bool,
		"clusteringNeighSize", &clusteringNeighSize, ConfigField::CFFT_Int,
		"numSamplings", &numSamplings, ConfigField::CFFT_Int,
		"bigPeaksThreshold", &bigPeaksThreshold, ConfigField::CFFT_Double,
		"regularizeAlphaField", &regularizeAlphaField, ConfigField::CFFT_Bool,
		"SNR", &SNR, ConfigField::CFFT_Int,
		"solverType", &solverType, ConfigField::CFFT_Int,
		"nTensFname", nTensFname, ConfigField::CFFT_String,
		"sizeCompartmentFname", sizeCompartmentFname, ConfigField::CFFT_String,
		"fPDDFname", fPDDFname, ConfigField::CFFT_String,
		"dwListFname", dwListFname, ConfigField::CFFT_String,
		"rawDataBaseDir", rawDataBaseDir, ConfigField::CFFT_String,
		"dwFname", dwFname, ConfigField::CFFT_String,
		"diffUnits", diffUnits, ConfigField::CFFT_String,
		"lambda", &lambda, ConfigField::CFFT_Double,
		"alpha0", &alpha0, ConfigField::CFFT_Double,
		"alpha1", &alpha1, ConfigField::CFFT_Double,
		"spatialSmoothing", &spatialSmoothing, ConfigField::CFFT_Bool,
		"WM_useS0HistogramThresholding", &WM_useS0HistogramThresholding, ConfigField::CFFT_Bool,
		"WM_S0Threshold", &WM_S0Threshold, ConfigField::CFFT_Double,
		"useS0MedianFilter", &useS0MedianFilter, ConfigField::CFFT_Bool,
		"WM_FAThreshold", &WM_FAThreshold, ConfigField::CFFT_Double,
		"WM_AVSignalThreshold", &WM_AVSignalThreshold, ConfigField::CFFT_Double,
		"WM_erosionSteps", &WM_erosionSteps, ConfigField::CFFT_Int,
		"loadMask", &loadMask, ConfigField::CFFT_Bool,
		"saveMask", &saveMask, ConfigField::CFFT_Bool,
		"saveODFField", &saveODFField, ConfigField::CFFT_Bool,
		"WM_maskFname", WM_maskFname, ConfigField::CFFT_String,
		"estimateDiffusivities", &estimateDiffusivities, ConfigField::CFFT_Bool,
		"longDiffusion", &longDiffusion, ConfigField::CFFT_Double,
		"transDiffusion", &transDiffusion, ConfigField::CFFT_Double,
		"reconstructionFname", reconstructionFname, ConfigField::CFFT_String
	};
	int nvars=sizeof(vars)/sizeof(ConfigField);
	ConfigField *cfvars=new ConfigField[nvars];
	memcpy(cfvars, vars, sizeof(ConfigField)*nvars);
	ConfigManager cm(cfvars, nvars);
	cm.loadFromFile(fname);
	cm.showConfig();
	delete[] cfvars;
	if(strcmp(diffUnits, "s/m^2")==0){
		b*=1e6;
	}

}
