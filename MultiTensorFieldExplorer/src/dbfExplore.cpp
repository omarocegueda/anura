#ifdef USE_QT
	#include "multitensorfieldexplorer.h"
	#include <QtGui/QApplication>	
#endif
#include <iostream>
#include "GDTI.h"
#include "DBF.h"
#include <vector>
#include <fstream>
#include <string>
#include "linearalgebra.h" 
#include "statisticsutils.h"
#include "nifti1_io.h"
#include "utilities.h"
#include "dtiutils.h"
#include "cv.h"
#include "highgui.h"
#include "lars.h"
#include "geometryutils.h"
#include "linearalgebra.h"
#include "nnls.h"
#include "nnsls.h"
#include "nnsls_pgs.h"
#include "emd_hat.hpp"
#include "emd_hat_signatures_interface.hpp"
#include "MultiTensorFieldExperimentParameters.h"
#include "DirectoryListing.h"
#include "expm.h"
using namespace std;


//#define TEST_INDIVIDUAL_MULTITENSORS



int nzCount(double *v, int n){
	int cnt=0;
	for(int i=0;i<n;++i)if(v[i]>1e-9){
		++cnt;
	}
	return cnt;
}

double sumVector(double *v, int n){
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=v[i];
	}
	return sum;
}

//-------------Earth mover's distance code-------------
double BasisDists[130][130];
void computeBasisDists(double *pdds, int m){
	for(int i=0;i<m;++i){
		double *pddi=&pdds[3*i];
		double nrm=sqrt(dotProduct(pddi,pddi,3));
		for(int j=0;j<3;++j){
			pddi[j]/=nrm;
		}
		for(int j=0;j<i;++j){
			double *pddj=&pdds[3*j];
			double p=dotProduct(pddi, pddj, 3);
			p=MAX(p,-1);
			p=MIN(p,1);
			double d=acos(p);
			BasisDists[i][j]=BasisDists[j][i]=d;
		}
		BasisDists[i][i]=0;
	}
}

double cost_mat_dist_double(feature_tt *F1, feature_tt *F2) { return BasisDists[*F1][*F2]; } // for emd_hat_signatures_interface

double shanonEntropy(double *h, int n, double sum){
	double entropy=0;
	for(int i=0;i<n;++i){
		entropy-=(h[i]/sum)*log((h[i])/sum);
	}
	return entropy;
}

double emdTensor(int *iAlpha, double *alpha, int mAlpha, 
			 int *iBeta, double *beta, int mBeta){
	signature_tt<double> A;
    signature_tt<double> B;
    A.n= mAlpha;
    B.n= mBeta;
	A.Features=iAlpha;
	B.Features=iBeta;
    A.Weights=alpha;
    B.Weights=beta;
	double d=emd_hat_signature_interface<double>(&A, &B, cost_mat_dist_double,-1);
	//----------rubner correction------
	double maxCost=0;
	for(int i=0;i<mAlpha;++i){
		for(int j=0;j<mBeta;++j){
			maxCost=MAX(maxCost, BasisDists[i][j]);
		}
	}
	double sumA=0;
	double sumB=0;
	for(int i=0;i<mAlpha;++i){
		sumA+=alpha[i];
	}
	for(int i=0;i<mBeta;++i){
		sumB+=beta[i];
	}
	double minSum=MIN(sumA, sumB);
	double maxSum=MAX(sumA, sumB);
	d=(d-(maxSum-minSum)*maxCost)/minSum + fabs(sumA-sumB)*maxCost;
	//---------Shanon entropy----------
	double seA=shanonEntropy(alpha, mAlpha, sumA);
	double seB=shanonEntropy(beta, mBeta, sumB);
	d+=maxSum*fabs(seA-seB);
	//---------------------------------
	return d;    
}
//-------------------------------------------------------






#ifdef USE_QT
//-----------------------------Qt main-------------------
int main(int argc, char *argv[]){
	QApplication a(argc, argv);
	MultiTensorFieldExplorer w;
	w.show();
	return a.exec();
}
#else
//-----------------------------non-Qt main-------------------

void runTestFromTxt(const string &gtFname){
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	params.bigPeaksThreshold=0.3;
	params.clusteringNeighSize=29;
	params.fitWithNeighbors=true;
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	strcpy(params.gtFname, gtFname.c_str());
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");
	params.regularizeAlphaField=false;
	params.SNR=10;
	params.solverType=DBFS_NNLS;

	vector<string> dbfDirsNames=ReadFileNamesFromDirectory(DEFAULT_DBF_FILE_NAME);
	for(unsigned i=0;i<dbfDirsNames.size();++i){
		strcpy(params.basisDirsFname, dbfDirsNames[i].c_str());
		cerr<<"DBFs: "<<params.basisDirsFname<<endl;
		size_t pos=string(params.basisDirsFname).find_first_of(".");
		string directoryName=params.basisDirsFname;
		if(pos!=string::npos){
			directoryName=string(params.basisDirsFname).substr(0,pos);
		}
		strcpy(params.outputDir, directoryName.c_str());
		DirUtil::CreateDir(directoryName);
		params.dellocate();
		params.loadData();
		if(!(params.loaded)){
			cerr<<"Error: could not load data. Aborting."<<endl;
			continue;
		}
		for(int SNR=10;SNR<=10;SNR+=5){
			params.SNR=SNR;
			cerr<<"\tSNR: "<<params.SNR<<endl;
			time_t seed=1331241348;
			srand(seed);
			testMultiTensorField(params, NULL, NULL);
		}
	}
}

vector<double> runTestFromNifti(	const string &baseDir,
						const string &GT_nTensFname, 
						const string &GT_sizeCompartmentFname, 
						const string &GT_fPDDFname,
						const string &REC_dwListFname,
						const string &REC_s0Fname,
						bool save,
						const string &Sol_nTensFname, 
						const string &Sol_sizeCompartmentFname, 
						const string &Sol_fPDDFname
					  ){
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	params.bigPeaksThreshold=0.3;
	params.clusteringNeighSize=16;
	params.fitWithNeighbors=true;
	params.regularizeAlphaField=false;
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");

	//---ground truth and input data---
	strcpy(params.nTensFname, (baseDir+GT_nTensFname).c_str());
	strcpy(params.sizeCompartmentFname, (baseDir+GT_sizeCompartmentFname).c_str());
	strcpy(params.fPDDFname, (baseDir+GT_fPDDFname).c_str());
	strcpy(params.rawDataBaseDir, "TestingData\\");
	params.groundTruthType=MTFGTT_NIFTI;
	
	//-----------------------------------
	if(getExtension(REC_dwListFname)==".nii"){
		params.inputType=MTFIT_NIFTI;
		strcpy(params.dwFname, REC_dwListFname.c_str());

	}else{
		params.inputType=MTFIT_DWMRI_FILES;
		strcpy(params.dwListFname, REC_dwListFname.c_str());

	}
	
	params.solverType=DBFS_NNLS;
	
	params.loadData();
	vector<double> retVal=testMultiTensorField(params, NULL, NULL);
	if(save){
		params.reconstructed->saveAsNifti(Sol_nTensFname, Sol_sizeCompartmentFname,Sol_fPDDFname);
	}
	
	return retVal;
}
void averageSignals(const string &baseDir, vector<string> listFnames, const string &destFname){
	for(unsigned i=0;i<listFnames.size();++i){
		listFnames[i]=baseDir+changeExtension(listFnames[i],".nii");
	}
	nifti_image *nii=nifti_image_read(listFnames[0].c_str(), 1);
	float *niiData=(float *)nii->data;
	int nvoxels=nii->nx*nii->ny*nii->nz*nii->nt;
	for(unsigned i=1;i<listFnames.size();++i){
		nifti_image *current=nifti_image_read(listFnames[i].c_str(), 1);
		float *currentData=(float *)current->data;
		for(int i=0;i<nvoxels;++i){
			niiData[i]+=currentData[i];
		}
	}
	for(int i=0;i<nvoxels;++i){
		niiData[i]/=double(listFnames.size());
	}
	nifti_set_filenames(nii,(baseDir+destFname).c_str(),0,1);
	nifti_image_write(nii);
	nifti_image_free(nii);
}

void buildNiftiFromDWMRI(const string &baseDir, const string &listFnames, const string &fname){
	int s0Indices[1]={0};
	int numS0=1;
	int nr;
	int nc;
	int ns;
	double *s0Volume=NULL;
	double *dwVolume=NULL;
	vector<string> dwNames;
	readFileNamesFromFile(baseDir+listFnames, dwNames);
	for(unsigned i=0;i<dwNames.size();++i){
		dwNames[i]=baseDir+dwNames[i];
	}
	loadDWMRIFiles(dwNames, s0Indices, numS0, s0Volume, dwVolume, nr, nc, ns);
	int signalLength=dwNames.size()-numS0;
	int dims[5]={4, nc, nr, ns,signalLength};
	nifti_image *nii=nifti_make_new_nim(dims,DT_FLOAT32,true);
	float *niiData=(float *)nii->data;
	for(int s=0;s<ns;++s){
		for(int r=0;r<nr;++r){
			for(int c=0;c<nc;++c){
				double *currentDW=&dwVolume[(s*nr*nc+r*nc+c)*signalLength];
				for(int k=0;k<signalLength;++k){
					niiData[c + nc*((nr-1-r) + nr*(s + ns*k))]=currentDW[k];
				}
				
			}
		}
	}
	nifti_set_filenames(nii,(baseDir+fname).c_str(),0,1);
	nifti_image_write(nii);
	nifti_image_free(nii);
	delete[] s0Volume;
	delete[] dwVolume;
}

void convert(vector<string> listFnames){
	//----convert----
	string baseDir="TestingData\\";
	for(unsigned i=0;i<listFnames.size();++i){
		string newName=changeExtension(listFnames[i], ".nii");
		buildNiftiFromDWMRI(baseDir,listFnames[i], newName);
	}
	//---------------
}

void generatePseudoGroundTruth_TestingSet(void){
	vector<string> listFnames;
	//testing data
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_05_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_10_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_15_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_20_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_25_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_30_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_35_nsamples_048_b1500.txt");
	listFnames.push_back("ListOfFiles_Testing_SF_SNR_40_nsamples_048_b1500.txt");
	
	vector<string> listS0Names;
	//testing data
	listS0Names.push_back("S0_Testing_SF_SNR_05_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_10_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_15_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_20_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_25_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_30_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_35_nsamples_048_b1500.nii");
	listS0Names.push_back("S0_Testing_SF_SNR_40_nsamples_048_b1500.nii");


	convert(listFnames);
	averageSignals("TestingData\\",listFnames,"average.nii");
	listFnames.clear();
	listFnames.push_back("average.nii");


	
	FILE *F=fopen("results.txt", "w");
	string rowNames[8]={"5","10","15","20","25","30","35","40"};
	fprintf(F,"0\tn-\tn+\tAngular\tODF\n");
	for(unsigned i=0;i<listFnames.size();++i){
		vector<double> results=runTestFromNifti("TestingData\\",
												"GT26Neigh_nTens.nii",
												"GT26Neigh_sizeCompartment.nii",
												"GT26Neigh_fPDD.nii",
												listFnames[i],
												listS0Names[i],
												true,
												"GT26Neigh_40_05_nTens.nii",
												"GT26Neigh_40_05_sizeCompartment.nii",
												"GT26Neigh_40_05_fPDD.nii"
												);
		fprintf(F,"%s", rowNames[i].c_str());
		for(unsigned j=0;j<results.size();++j){
			fprintf(F,"\t%0.5lf", results[j]);
		}
		fprintf(F,"\n");
	}
	fclose(F);
}

void generatePseudoGroundTruth_TrainingSet(const string &gtFname, const string &schemeFname){
	const double b=1500;
	//----------load scheme--------------
	double *gradients=NULL;
	int numGradients;
	int *s0Indices=NULL;
	int numS0;
	loadOrientations(schemeFname,gradients, numGradients, s0Indices, numS0);
	for(int i=0;i<numGradients;++i){
		double *g=&gradients[3*i];
		double nn=sqrt(dotProduct(g,g,3));
		for(int j=0;j<3;++j){
			g[j]/=nn;
		}
	}
	delete[] s0Indices;
	//-------load ground truth-----
	MultiTensorField GT;
	GT.loadFromTxt(gtFname);
	MultiTensor *voxGT=GT.getVoxels();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	int nvoxels=nrows*ncols*nslices;
	double *dwVolume=new double[nvoxels*numGradients];
	double *dwCopy=new double[nvoxels*numGradients];
	memset(dwVolume, 0, sizeof(double)*nvoxels*numGradients);
	double *s=new double[numGradients];
	int nvolumes=1;
	for(int SNR=40;SNR>=5;SNR-=5, ++nvolumes){
		for(int v=0;v<nvoxels;++v){
			double *currentSignal=&dwVolume[v*numGradients];
			voxGT[v].acquireWithScheme(b,gradients, numGradients, 1.0/double(SNR), s);
			linCombVector<double>(currentSignal, 1, s, numGradients, currentSignal);
		}
		//save current volume
		multVectorScalar(dwVolume, 1.0/double(nvolumes), nvoxels*numGradients,dwCopy);
		ostringstream os;
		os<<"average_"<<SNR<<".nii";
		save4DNifti(os.str(), dwCopy, nslices, nrows, ncols, numGradients);

	}
	delete[] s;
	delete[] gradients;
	delete[] s0Indices;
	delete[] dwCopy;
}


void singleTest_GTGenerator(const string &gtFname){
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	params.bigPeaksThreshold=0.3;
	params.clusteringNeighSize=16;
	params.fitWithNeighbors=true;
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	strcpy(params.gtFname, gtFname.c_str());
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");
	params.regularizeAlphaField=false;
	params.SNR=0;
	params.solverType=DBFS_NNLS;
	params.inputType=MTFIT_GTGenerator;
	params.groundTruthType=MTFGTT_GTGenerator;
	
	strcpy(params.dwFname, "average_5.nii");

	params.loadData();
	testMultiTensorField(params, NULL, NULL);

	strcpy(params.dwFname, "average_40.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);

	strcpy(params.dwFname, "average_35.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);

	strcpy(params.dwFname, "average_30.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);

	strcpy(params.dwFname, "average_25.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	
	strcpy(params.dwFname, "average_20.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	
	strcpy(params.dwFname, "average_15.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	
	strcpy(params.dwFname, "average_10.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	
	strcpy(params.dwFname, "average_5.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
}


void singleTest_TestSet(void){
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");
	params.bigPeaksThreshold=0.3;
	params.clusteringNeighSize=16;
	params.fitWithNeighbors=true;
	params.regularizeAlphaField=false;
	params.solverType=DBFS_NNLS;
	params.inputType=MTFIT_NIFTI;
	strcpy(params.rawDataBaseDir, ".\\");

	//ground truth
	params.groundTruthType=MTFGTT_NIFTI;
	strcpy(params.fPDDFname, "GT_40_05_fPDD.nii");
	strcpy(params.nTensFname, "GT_40_05_nTens.nii");
	strcpy(params.sizeCompartmentFname, "GT_40_05_sizeCompartment.nii");
	//
	


	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_05_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR05.txt");
	
	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_10_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR10.txt");

	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_15_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR15.txt");

	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_20_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR20.txt");

	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_25_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR25.txt");
	
	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_30_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR30.txt");
	
	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_35_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR35.txt");
	
	strcpy(params.dwFname, "TestingData\\ListOfFiles_Testing_SF_SNR_40_nsamples_048_b1500.nii");
	params.loadData();
	testMultiTensorField(params, NULL, NULL);
	params.reconstructed->saveToTxt("reconstructed_SNR40.txt");
}

void unstructuredTest(){
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");
	params.bigPeaksThreshold=0.2;
	params.clusteringNeighSize=6;
	params.fitWithNeighbors=false;
	params.regularizeAlphaField=false;
	params.solverType=DBFS_NNLS;
	params.inputType=MTFIT_GTGenerator;
	strcpy(params.rawDataBaseDir, ".\\");

	//ground truth
	params.groundTruthType=MTFGTT_GTGenerator;
	strcpy(params.gtFname, "Training_IV.txt");
	//
	params.SNR=10;
	params.loadData();
	strcpy(params.outputDir, ".");
#ifdef TEST_INDIVIDUAL_MULTITENSORS
	MultiTensor *voxE=params.reconstructed->getVoxels();
	MultiTensor *voxGT=params.GT->getVoxels();
	int nrows=params.GT->getNumRows();
	int ncols=params.GT->getNumCols();
	int nslices=params.GT->getNumSlices();
	int nvoxels=nrows*ncols*nslices;

	double sigma=0;
	if(params.SNR>0){
		sigma=1.0/params.SNR;
	}

	double *S=new double[params.numGradients];
	for(int v=0;v<nvoxels;++v){
		voxGT[v].acquireWithScheme(params.b, params.gradients, params.numGradients, sigma,S);
		voxE[v].fitMultiTensor(1,S,params.gradients,params.numGradients, params.b, params.DBFDirections, params.numDBFDirections);
	}
	delete[] S;
	//---evaluate---
	vector<double> errorPositiveCompartmentCount;
	vector<double> errorNegativeCompartmentCount;
	vector<double> errorAngularPrecision;
	vector<double> errorODFAccuracy;
	evaluatePositiveCompartmentCount(	*params.GT, *params.reconstructed, errorPositiveCompartmentCount);
	evaluateNegativeCompartmentCount(	*params.GT, *params.reconstructed, errorNegativeCompartmentCount);
	evaluateAngularPrecision(	*params.GT, *params.reconstructed, errorAngularPrecision);
	evaluateODFAccuracy		(	*params.GT, *params.reconstructed, params.ODFDirections, params.numODFDirections, errorODFAccuracy);
	//-----print errors---
	double errorPositiveCompartmentCountMean=getMean(errorPositiveCompartmentCount);
	double errorNegativeCompartmentCountMean=getMean(errorNegativeCompartmentCount);
	double errorAngularPrecisionMean=getMean(errorAngularPrecision);
	double errorODFAccuracyMean=getMean(errorODFAccuracy);
	double errorPositiveCompartmentCountSD=getStdev(errorPositiveCompartmentCount);
	double errorNegativeCompartmentCountSD=getStdev(errorNegativeCompartmentCount);
	double errorAngularPrecisionSD=getStdev(errorAngularPrecision);
	double errorODFAccuracySD=getStdev(errorODFAccuracy);
	//-----------------
#else	
	/*double minThr=0.1;
	double maxThr=0.3;
	int thrSteps=5;
	double deltaThr=(maxThr-minThr)/(thrSteps-1);
	int minNeigh=6;
	int maxNeigh=75;
	int neighSteps=maxNeigh-minNeigh+1;
	params.bigPeaksThreshold=minThr;
	for(int t=0;t<thrSteps;++t, params.bigPeaksThreshold+=deltaThr){
		params.clusteringNeighSize=minNeigh;
		for(int n=0;n<neighSteps;++n, ++neighSize){
			
			
		}
	}
	*/

	for(params.SNR=5;params.SNR<=40;params.SNR+=5){
		cerr<<"SNR "<<params.SNR<<endl;
		testMultiTensorField(params, NULL, NULL);
	}
	
#endif
}

void getCommandLineParameters(int argc, char *argv[], MultiTensorFieldExperimentParameters &params){
	for(int i=0;i<argc;++i){
		if((string(argv[i])=="-SNR") && ((i+1)<argc)){
			istringstream is(argv[i+1]);
			double val;
			if(is>>val){
				params.SNR=val;
			}
		}else if((string(argv[i])=="-BPT") && ((i+1)<argc)){
			istringstream is(argv[i+1]);
			double val;
			if(is>>val){
				params.bigPeaksThreshold=val;
			}
		}else if((string(argv[i])=="-neighSize") && ((i+1)<argc)){
			istringstream is(argv[i+1]);
			int val;
			if(is>>val){
				params.clusteringNeighSize=val;
			}
		}else if((string(argv[i])=="-b") && ((i+1)<argc)){
			istringstream is(argv[i+1]);
			double val;
			if(is>>val){
				params.b=val;
			}
		}else if(string(argv[i])=="-fitWithNeighbors"){
			params.fitWithNeighbors=true;
		}else if(string(argv[i])=="-fitGDTIOnly"){
			params.fitGDTIOnly=true;
		}else if(string(argv[i])=="-regularize"){
			params.regularizeAlphaField=true;
		}else if(string(argv[i])=="-useFAIndicator"){
			params.useFAIndicator=true;
		}else if(string(argv[i])=="-useGDTIPostProcessing"){
			params.useGDTIPostProcessing=true;
		}else if(string(argv[i])=="-applyTensorSplitting"){
			params.applyTensorSplitting=true;
		}else if(string(argv[i])=="-IDU"){
			params.iterativeBasisUpdate=true;
		}else if((string(argv[i])=="-gt") && ((i+1)<argc)){
			strcpy(params.gtFname, argv[i+1]);
		}else if(string(argv[i])=="-iterateDiffProf"){
			params.iterateDiffProf=true;
		}else if((string(argv[i])=="-eval") &&((i+1)<argc)){
			params.evaluate=true;
			strcpy(params.solutionFname, argv[i+1]);
		}else if((string(argv[i])=="-evalList") &&((i+1)<argc)){
			params.evaluateList=true;
			strcpy(params.solutionListFname, argv[i+1]);
		}else if(string(argv[i])=="-smooth"){
			params.spatialSmoothing=true;
		}else if((string(argv[i])=="-lambda") && ((i+1)<argc)){
			params.lambda=true;
			istringstream is(argv[i+1]);
			double val;
			if(is>>val){
				params.lambda=val;
			}
		}else if((string(argv[i])=="-denoisingTest") && ((i)<argc)){
			params.denoisingTest=true;
		}
	}
}

void runGUIPipeline(int argc, char *argv[]){
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;
	if(argc>1){
		if(fileExists(string(argv[1]))){
			params.configFromFile(argv[1]);
		}
	}else{
		params.configFromFile("dbfconfig.ini");
	}
	getCommandLineParameters(argc, argv, params);
	params.loadData();
	if(params.evaluate){
		params.reconstructed->loadFromTxt(params.solutionFname);
		common_evaluate(*params.GT, *params.reconstructed);
	}else if(params.evaluateList){
		vector<string> names;
		readFileNamesFromFile(params.solutionListFname, names);
		FILE *FNegative=fopen("fullNegativeStats.txt","w");
		FILE *FPositive=fopen("fullPositiveStats.txt","w");
		FILE *FAngular=fopen("fullAngularStats.txt","w");
		for(unsigned i=0;i<names.size();++i){
			params.reconstructed->loadFromTxt(names[i]);
			common_evaluate(*params.GT, *params.reconstructed, FNegative, FPositive, FAngular);
		}
		fclose(FNegative);
		fclose(FPositive);
		fclose(FAngular);
	}else if(params.denoisingTest){
		testDenoisingMethod(params);
	}else{
		testMultiTensorField(params, NULL, NULL);
		params.reconstructed->saveToTxt(params.reconstructionFname);
	}
	
	
}

void testExponential(void){
	double A[9]={1,1,0,0,0,2,0,0,-1};
	multVectorScalar<double>(A,0.0024,9,A);
	double E[9];
	expm(A,3,E);
	for(int i=0;i<3;++i){
		for(int j=0;j<3;++j){
			cerr<<E[i*3+j]<<"\t";
		}
		cerr<<endl;
	}
}

int main(int argc, char *argv[]){
	runGUIPipeline(argc, argv);
	//string gtFname="Training_3D_SF.txt";
	//if(argc>1){
	//	gtFname=argv[1];
	//}
	//singleTest_GTGenerator(gtFname);
	//singleTest_TestSet();
	//generatePseudoGroundTruth_TestingSet();
	//unstructuredTest();
	return 0;
}

#endif



