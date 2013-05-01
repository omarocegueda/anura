#include "DBF.h"
#include <vector>
#include <string>
#include "utilities.h"
#include "dtiutils.h"
#include "nifti1_io.h"
#include "linearalgebra.h"
#include "geometryutils.h"
#include "nnls.h"
#include "nnsls.h"
#include "lars.h"
#include "nnsls_pgs.h"
#include "GDTI.h"
#include <cv.h>
#include <highgui.h>
#include "dtisynthetic.h"
#include "emd_hat.hpp"
#include "emd_hat_signatures_interface.hpp"
//#define USE_EMD
//#define USE_JOOG_REG
using namespace std;

DBF::DBF(double _b, double _longitudinalDiffusion, double _transversalDiffusion, double *_gradients, int _numGradients, double *_basisDirections, int _numBasisDirections){
	b=_b;
	longitudinalDiffusion=_longitudinalDiffusion;
	transversalDiffusion=_transversalDiffusion;
	createDiffusionBasisFunctions(_gradients, _numGradients, _basisDirections, _numBasisDirections);
	solverType=DBFS_NNLS;
}

DBF::DBF(double *_gradients, int _numGradients, double *_basisDirections, int _numBasisDirections){
	numGradients=_numGradients;
	numBasisFunctions=_numBasisDirections;
	diffusionBasis=new double[numBasisFunctions*numGradients];
	diffusionDirections=new double[numBasisFunctions*3];
	gradients=new double[numGradients*3];

	memcpy(diffusionDirections, _basisDirections, sizeof(double)*numBasisFunctions*3);
	memcpy(gradients, _gradients, sizeof(double)*numGradients*3);
	memset(diffusionBasis, 0, sizeof(double)*numBasisFunctions*numGradients);
	solverType=DBFS_NNLS;
}

void DBF::setDiffusionBasis(double *Phi){
	memcpy(diffusionBasis, Phi, sizeof(double)*numBasisFunctions*numGradients);
}

DBF::~DBF(){
	DELETE_ARRAY(diffusionBasis);
	DELETE_ARRAY(diffusionDirections);
	DELETE_ARRAY(gradients);
}
void DBF::reComputeDiffusionBasisFunctions(double _b, double _longitudinalDiffusion, double _transversalDiffusion){
	b=_b;
	longitudinalDiffusion=_longitudinalDiffusion;
	transversalDiffusion=_transversalDiffusion;
	double e0[3]={1,0,0};
	double D[9]={
		longitudinalDiffusion, 0, 0,
		0, transversalDiffusion, 0,
		0, 0, transversalDiffusion
	};
	for(int i=0;i<numBasisFunctions;++i){
		double *dir=&diffusionDirections[3*i];
		double T[9];
		fromToRotation(e0, dir, T);
		double Ti[9]={//Ti=D*T'
			T[0]*longitudinalDiffusion, T[3]*longitudinalDiffusion, T[6]*longitudinalDiffusion,
			T[1]*transversalDiffusion, T[4]*transversalDiffusion, T[7]*transversalDiffusion,
			T[2]*transversalDiffusion, T[5]*transversalDiffusion, T[8]*transversalDiffusion
		};
		multMatrixMatrix<double>(T,Ti,3,Ti);
		for(int j=0;j<numGradients;++j){
			int dbfPos=j*numBasisFunctions+i;
			double *g=&gradients[3*j];
			double eval=evaluateQuadraticForm(Ti, g, 3);
			diffusionBasis[dbfPos]=exp(-b*eval);
		}
	}
}

void DBF::computeDiffusionFunction(double *dir, double _lambdaMin, double _lambdaMiddle, double _lambdaLong, double *phi, int idxInc){
	::computeDiffusionFunction(dir, _lambdaMin, _lambdaMiddle, _lambdaLong, b, gradients, numGradients, phi, idxInc);
}

void DBF::createDiffusionBasisFunctions(double *orientations, int numOrientations, double *basisDirections, int numBasisDirections){
	numGradients=numOrientations;
	numBasisFunctions=numBasisDirections;
	diffusionBasis=new double[numBasisFunctions*numGradients];
	diffusionDirections=new double[numBasisDirections*3];
	gradients=new double[numOrientations*3];
	memcpy(diffusionDirections, basisDirections, sizeof(double)*numBasisDirections*3);
	memcpy(gradients, orientations, sizeof(double)*numGradients*3);
	reComputeDiffusionBasisFunctions(b, longitudinalDiffusion, transversalDiffusion);
}

void DBF::setSolverType(DBFSolverType _solverType){
	solverType=_solverType;
}

double DBF::solve(double *Si, double *alphas, double lprop, bool NNOLS_fit, double *s0){
	double *Phi;
	if(s0==NULL){
		Phi=diffusionBasis;
	}else{
		Phi=new double[numGradients*numBasisFunctions];
		memcpy(Phi, diffusionBasis, sizeof(double)*numGradients*numBasisFunctions);
		multVectorScalar(Phi, *s0, numGradients*numBasisFunctions, Phi);
	}
		
	int len=numBasisFunctions*numGradients;
	double retVal;
	switch(solverType){
		case DBFS_NNLS:{
			double errorNNLS=0;
			nnls(Phi, numGradients, numBasisFunctions, Si, alphas, &errorNNLS);
			retVal=errorNNLS;
		}
		break;
		case DBFS_LARS:{
			double errorNNLS=0;
			//nnlars_addaptiveScale(Phi, numGradients, numBasisFunctions, Si, alphas, lprop*500, lprop, &errorNNLS);
			nnls(Phi, numGradients, numBasisFunctions, Si, alphas, &errorNNLS);
			if((lprop>0) && (lprop<1)){
				double *betas=new double[numBasisFunctions];
				double sa=0;
				for(int i=0;i<numBasisFunctions;++i){
					sa+=fabs(alphas[i]);
				}
				double errorNNLARS=0;
				nnlars(Phi, numGradients, numBasisFunctions, Si, betas,sa*lprop, &errorNNLARS,false);
				if(NNOLS_fit){
					int *I=new int[numBasisFunctions];
					int mm=0;
					for(int i=0;i<numBasisFunctions;++i)if(betas[i]>5e-9){
						I[mm]=i;
						++mm;
					}
					nnls_subspace(Phi, numGradients, numBasisFunctions, Si, betas, I, mm, &errorNNLARS);
					delete[] I;
				}
				memcpy(alphas, betas, sizeof(double)*numBasisFunctions);
				delete[] betas;
				retVal=errorNNLARS;
			}else{
				retVal=errorNNLS;
			}
			
		}
		break;
		case DBFS_NNSLS:{
			double errorNNLS=0;
			nnls(Phi, numGradients, numBasisFunctions, Si, alphas, &errorNNLS);
			double fullEnergyError=SQR(errorNNLS)*numGradients;
			double zeroEnergyError=0;
			for(int i=0;i<numGradients;++i){
				zeroEnergyError+=SQR(Si[i]);
			}
			double sa=0;
			for(int i=0;i<numBasisFunctions;++i){
				sa+=fabs(alphas[i]);
			}
			double gainRate=(zeroEnergyError-fullEnergyError)/sa;

			double errorNNSLS=0;
			nnsls(Phi, numGradients, numBasisFunctions, Si, alphas, SQR(lprop*gainRate), &errorNNSLS);
			if(NNOLS_fit){
				int *I=new int[numBasisFunctions];
				int mm=0;
				for(int i=0;i<numBasisFunctions;++i)if(alphas[i]>EPSILON){
					I[mm]=i;
					++mm;
				}
				nnls_subspace(Phi, numGradients, numBasisFunctions, Si, alphas, I, mm, &errorNNSLS);
				delete[] I;
			}
			retVal=errorNNSLS;
		}
		break;
		case DBFS_PGS:{
			double errorNNLS=0;
			nnls(Phi, numGradients, numBasisFunctions, Si, alphas, &errorNNLS);
			double errorPGS=0;
			nnsls_pgs(Phi, numGradients, numBasisFunctions, Si, alphas,lprop, &errorPGS);
			retVal=errorPGS;
		}
		break;
	}
	if(s0!=NULL){
		delete[] Phi;
	}
	return retVal;
}

int DBF::getNumBasisFunctions(void){
	return numBasisFunctions;
}

double *DBF::getDiffusionBasis(void){
	return diffusionBasis;
}

double *DBF::getDiffusionDirections(void){
	return diffusionDirections;
}

double *DBF::getGradients(void){
	return gradients;
}

int DBF::getNumGradients(void){
	return numGradients;
}

DBF_output::DBF_output(){
	pdds=NULL;
	sizeCompartments=NULL;
	alphas=NULL;
	error=NULL;
	nzCount=NULL;
}

DBF_output::~DBF_output(){
	DELETE_ARRAY(pdds);
	DELETE_ARRAY(sizeCompartments);
	DELETE_ARRAY(alphas);
}


void DBF_params::setDefault(void){
#ifdef _WIN32
	/*orientationsFileName="D:\\local_repository\\research\\fiberCup\\1500\\3x3x3_diffusion_directions.txt";
	dbfDirectionsFileName="D:\\downloads\\dbf_tracto_fibrecupData\\DBF_129orientations.dat";
	string fileNamesFileName="D:\\downloads\\dbf_tracto_fibrecupData\\B0_DW_Files_2Repeats_3x3x3_dwi_b1500.dat";
	string rawDataBaseDir="D:\\downloads\\dbf_tracto_fibrecupData\\rawData\\";
    string maskFileName="D:\\downloads\\dbf_tracto_fibrecupData\\output\\outputDT\\DT_maskFA_0_fibreCup2Rep.nii";
	b=1.5e9;
	transversalDiffusion=0.000000001393979; 
	longitudinalDiffusion=0.000000001974300;
	*/
	orientationsFileName="D:\\cimat\\experiments\\PGS_evaluation\\SenalesConGT\\CHOP_5B0_64DW_1Repeat_scheme.txt";
	dbfDirectionsFileName="D:\\cimat\\experiments\\PGS_evaluation\\DBF_129orientations.dat";
	string fileNamesFileName="D:\\cimat\\experiments\\PGS_evaluation\\SenalesConGT\\filesSNR10.txt";
	string rawDataBaseDir="D:\\cimat\\experiments\\PGS_evaluation\\SenalesConGT\\";
    string maskFileName="D:\\cimat\\experiments\\PGS_evaluation\\SenalesConGT\\mask.nii";
	b=1e3;
	//transversalDiffusion=1e-3; 
	//longitudinalDiffusion=2.2e-4;
#else
    orientationsFileName="3x3x3_diffusion_directions.txt";
	dbfDirectionsFileName="DBF_129orientations.dat";
	string fileNamesFileName="/Users/omar/dbf_tracto_fibrecupData/B0_DW_Files_2Repeats_3x3x3_dwi_b1500.dat";
	string rawDataBaseDir="/Users/omar/dbf_tracto_fibrecupData/rawData/";
    string maskFileName="/Users/omar/dbf_tracto_fibrecupData/outputDT/DT_maskFA_0_fibreCup2Rep.nii";
#endif
    
    
	readFileNamesFromFile(fileNamesFileName, dwNames);
	for(unsigned i=0;i<dwNames.size();++i){
		dwNames[i]=rawDataBaseDir+dwNames[i];
	}
	
	
	nifti_image *nii_mask=nifti_image_read(maskFileName.c_str(), 1);
	nr=nii_mask->ny;
	nc=nii_mask->nx;
	ns=nii_mask->nz;
	int len=nr*nc*ns;
	mask=new unsigned char[len];
	memset(mask,  0, sizeof(unsigned char)*len);
	float *nii_mask_data=(float *)nii_mask->data;
	int nVoxels=0;
	for(int i=0;i<len;++i){
		if(nii_mask_data[i]>0){
			mask[i]=1;
			++nVoxels;
		}
		
	}
	nifti_image_free(nii_mask);
	/*string MDFileName="D:\\downloads\\dbf_tracto_fibrecupData\\output\\outputDT\\DT_MD_fibreCup2Rep.nii";
	nifti_image *nii_md=nifti_image_read(MDFileName.c_str(), 1);*/
	lProp=0.1;
}

DBF_params::DBF_params(){
	mask=NULL;
}

DBF_params::~DBF_params(){
	if(mask!=NULL){
		delete[] mask;
	}
}

void loadDiffusionDirections(const std::string fname, double *&directions, int &numDirections){
	FILE *F=fopen(fname.c_str(), "r");
	vector<double> v;
	while(!feof(F)){
		double x,y,z;
		if(fscanf(F,"%lf%lf%lf", &x, &y, &z)==3){
			v.push_back(x);
			v.push_back(y);
			v.push_back(z);
		}
	}
	numDirections=v.size()/3;
	directions=new double[v.size()];
	for(unsigned i=0;i<v.size();++i){
		directions[i]=v[i];
	}
}

void choosePDDs(DBF_output &output, double *basisDirections){
	//---select best directions---
	int &ns=output.ns;
	int &nr=output.nr;
	int &nc=output.nc;
	double *alphas=output.alphas;
	double *pdds=output.pdds;
	int &maxDirections=output.maxDirections;
	int &numBasisDirections=output.numBasisDirections;

	memset(output.pdds,0,sizeof(double)*3*maxDirections*nr*nc*ns);
	memset(output.sizeCompartments,0,sizeof(double)*maxDirections*nr*nc*ns);

	double *sizeCompartments=output.sizeCompartments;
	for(int ps=0;ps<ns;++ps){
		for(int pc=0;pc<nc;++pc){
			for(int pr=0;pr<nr;++pr){
				double *currentAlphas=&alphas[numBasisDirections*(ps*(nr*nc) + pr*nc + pc)];
				double *currentPDDS=&pdds[3*maxDirections*(ps*(nr*nc) + pr*nc + pc)];
				double *currentSizeCompartments=&sizeCompartments[maxDirections*(ps*(nr*nc) + pr*nc + pc)];
				set<pair<double, int> > selected;
				for(int i=0;i<numBasisDirections;++i)if(currentAlphas[i]>EPSILON){
					selected.insert(make_pair(-currentAlphas[i], i));
				}
				int cnt=0;
				for(set<pair<double, int> >::iterator it=selected.begin();(it!=selected.end()) && (cnt<maxDirections);++it){
					int idx=it->second;
					memcpy(&currentPDDS[3*cnt], &basisDirections[3*idx], 3*sizeof(double));
					currentSizeCompartments[cnt]=-(it->first);
					++cnt;
				}
			}
		}
	}
	
}
int fitDBF(const DBF_params &params, DBF_output &output){
	double *orientations=NULL;
	int numOrientations=0;
	int *s0Indices=NULL;
	int numS0=0;
	double *basisDirections=NULL;
	int numBasisDirections=0;

	int nr, nc, ns;
	double *dwVolume=NULL;
	double *s0Volume=NULL;

	loadOrientations(params.orientationsFileName,orientations, numOrientations, s0Indices, numS0);
	
    loadDiffusionDirections(params.dbfDirectionsFileName, basisDirections, numBasisDirections);

	vector<set<int> > neighborhoods;
	buildNeighborhood(basisDirections, numBasisDirections, 15, neighborhoods);
	
    loadDWMRIFiles(params.dwNames, s0Indices, numS0, s0Volume, dwVolume, nr, nc, ns);
	//--------estimate diffusivity parameters--------
	int len=nr*nc*ns;
	//unsigned char *mask=new unsigned char[len];
	//getMaximumConnectedComponentMask(s0Volume, nr, nc, ns, mask);
	GDTI H(2, params.b, orientations, numOrientations);
	double longDiffusion;
	double transDiffusion;
	double *eigenvalues=new double[3*len];
	unsigned char *mask=new unsigned char[len];
	H.createMask(s0Volume, dwVolume, ns, nr, nc, NULL, eigenvalues, NULL, mask, -1, -1,-1);
	double avProfile[3];
	H.computeAverageProfile(eigenvalues, len, mask, 3, 1, avProfile);
	longDiffusion=avProfile[2];
	transDiffusion=0.5*(avProfile[0]+avProfile[1]);
	delete[] mask;
	delete[] eigenvalues;
	//-----------------------------------------------
	DBF dbfInstance(params.b, longDiffusion, transDiffusion,orientations, numOrientations, basisDirections, numBasisDirections);
    

	double s0=0;
	double *dwSignal=new double[numOrientations];
	double totalError=0;
	output.maxDirections=3;
	output.numBasisDirections=numBasisDirections;
	output.nzCount=new int[nr*nc*ns];
	output.error=new double[nr*nc*ns];
	output.alphas=new double[nr*nc*ns*numBasisDirections];
	output.pdds=new double[nr*nc*ns*output.maxDirections*3];
	output.sizeCompartments=new double[nr*nc*ns*output.maxDirections];
	output.ns=ns;
	output.nr=nr;
	output.nc=nc;
	memset(output.pdds, 0, sizeof(double)*nr*nc*ns*output.maxDirections*3);
	memset(output.alphas, 0, sizeof(double)*nr*nc*ns*output.maxDirections);
	memset(output.error, 0, sizeof(double)*nr*nc*ns);
	memset(output.nzCount, 0, sizeof(int)*nr*nc*ns);
	double *Phi=new double[numOrientations*numBasisDirections];
	//-----sum the slices---
	double *sumImage=new double[nr*nc];
	memset(sumImage, 0, sizeof(double)*nr*nc);
	double maxVal=-1e10;
	double minVal=1e10;
	for(int pc=0;pc<nc;++pc){
		for(int pr=0;pr<nr;++pr){
			double *currentPDDS=&output.pdds[3*output.maxDirections*(1*(nr*nc) + pr*nc + pc)];
			double *alphas=&output.alphas[numBasisDirections*(1*(nr*nc) + pr*nc + pc)];
			int pos=1*(nr*nc)+(nr-1-pr)*nc+pc;
			if(params.mask[pos]){
				getDWSignalAtVoxel(s0Volume, dwVolume, nr, nc, ns, numOrientations, pr, pc, 1, s0, dwSignal);
				sumImage[pr*nc+pc]=0;
				for(int i=0;i<numOrientations;++i){
					sumImage[pr*nc+pc]+=dwSignal[i];
				}
				sumImage[pr*nc+pc]/=s0;
				if(sumImage[pr*nc+pc]<minVal){
					minVal=sumImage[pr*nc+pc];
				}
				if(maxVal<sumImage[pr*nc+pc]){
					maxVal=sumImage[pr*nc+pc];
				}
			}
		}
	}
	double diff=maxVal-minVal;
	double maxSum=maxVal;
	double minSum=minVal;
	cv::Mat img;
	img.create(nr,nc,CV_8UC1);
	unsigned char *img_data=(unsigned char *)img.data;
	memset(img_data, 0, sizeof(unsigned char)*nr*nc);
	for(int i=0;i<nr;++i){
		for(int j=0;j<nc;++j)if(sumImage[i*nc+j]>0){
			img_data[i*nc+j]=(unsigned char)(255*((sumImage[i*nc+j]-minVal)/diff));
		}
	}
	cv::imshow("sum", img);
	

	//----------------------
	
#ifdef RUN_LOSS_ANALYSIS
	vector<vector<double> > errors;
	vector<vector<double> > sparsity;
	double lprop=-1;
	for(double lprop=0.05;lprop-1e-9<0.95;lprop+=0.05){
		vector<double> currentErrors;
		vector<double> currentSparsity;
#endif
		dbfInstance.setSolverType(DBFS_LARS);
		for(int ps=0;ps<ns;++ps){
			cerr<<"Slice "<<ps<<endl;

			maxVal=-1e10;
			minVal=1e10;
			for(int pc=0;pc<nc;++pc){
				for(int pr=0;pr<nr;++pr){
					double *currentPDDS=&output.pdds[3*output.maxDirections*(ps*(nr*nc) + pr*nc + pc)];
					double *alphas=&output.alphas[numBasisDirections*(ps*(nr*nc) + pr*nc + pc)];
					int pos=ps*(nr*nc)+(nr-1-pr)*nc+pc;
					if(params.mask[pos]){
						getDWSignalAtVoxel(s0Volume, dwVolume, nr, nc, ns, numOrientations, pr, pc, ps, s0, dwSignal);
						
#ifdef RUN_LOSS_ANALYSIS
						double nnlsError=dbfInstance.solve(dwSignal, alphas, -1, false);
						double nnlarsError=dbfInstance.solve(dwSignal, alphas, lprop, true);
						currentErrors.push_back((nnlarsError-nnlsError)/nnlsError);
#else
						//------test solver----
						/*multVectorScalar(dwSignal, 1.0/s0, numOrientations, dwSignal);
						double sumSignal=0;
						for(int i=0;i<numOrientations;++i){
							sumSignal+=dwSignal[i];
						}
						//double nnslsError=dbfInstance.solve(dwSignal, alphas,params.lProp, true);
						double lprop=(sumSignal-minSum)/(maxSum-minSum);
						lprop=MIN(lprop,0.5);
						lprop=MAX(lprop,0.05);
						double nnslsError=dbfInstance.solve(dwSignal, alphas,lprop, true);
						

						multMatrixVector(dbfInstance.getDiffusionBasis(), alphas, numOrientations, numBasisDirections, dwSignal);
						double sumRecovered=0;
						sumImage[pr*nc+pc]=0;
						for(int i=0;i<numOrientations;++i){
							sumRecovered+=dwSignal[i];
							sumImage[pr*nc+pc]+=dwSignal[i];
						}
						if(sumImage[pr*nc+pc]<minVal){
							minVal=sumImage[pr*nc+pc];
						}
						if(maxVal<sumImage[pr*nc+pc]){
							maxVal=sumImage[pr*nc+pc];
						}

						output.error[ps*(nr*nc) + pr*nc + pc]=fabs(sumSignal-sumRecovered)/sumSignal;
						//output.error[ps*(nr*nc) + pr*nc + pc]=nnslsError;
						for(int i=0;i<numBasisDirections;++i)if(alphas[i]>EPSILON){
							output.nzCount[ps*(nr*nc) + pr*nc + pc]++;
						}
						*/
						//------test iterative-------
						memcpy(Phi, dbfInstance.getDiffusionBasis(), sizeof(double)*numOrientations*numBasisDirections);
						multVectorScalar(Phi,s0,numOrientations*numBasisDirections, Phi);
						double nnlsError=-1;
						nnls(Phi, numOrientations, numBasisDirections, dwSignal, alphas, &nnlsError);
						double iterativeError=nnlsError;
						//nnsls_pgs(Phi, numOrientations,numBasisDirections, dwSignal, alphas, -700, &iterativeError);
						output.error[ps*(nr*nc) + pr*nc + pc]=iterativeError;
						for(int i=0;i<numBasisDirections;++i)if(alphas[i]>EPSILON){
							output.nzCount[ps*(nr*nc) + pr*nc + pc]++;
						}
						//---------------------------

						vector<set<int> > groups;
						groupCoefficients(alphas, numBasisDirections, 1e-2, neighborhoods, groups);
						int numClusters=MIN(groups.size(), 3);
						for(int i=0;i<numClusters;++i){
							double centroid[3];
							double amountDiff=0;
							computeCentroid(basisDirections, numBasisDirections, alphas, groups[i], centroid, amountDiff);
							amountDiff=amountDiff;
						}
						
						
						
#endif
					}
				}
			}
			//---
			diff=maxVal-minVal;
			for(int i=0;i<nr;++i){
				for(int j=0;j<nc;++j)if(sumImage[i*nc+j]>0){
					img_data[i*nc+j]=(unsigned char)(255*((sumImage[i*nc+j]-minVal)/diff));
				}
			}
			ostringstream imgName;
			imgName<<ps;
			cv::imshow(imgName.str().c_str(), img);
		}
#ifdef RUN_LOSS_ANALYSIS
		errors.push_back(currentErrors);
	}
	FILE *F=fopen("d:\\loss_stats.txt", "w");
	/*for(double lprop=0.05;lprop-1e-9<0.95;lprop+=0.05){
		fprintf(F, "%0.7lf\t", lprop);
	}*/
	for(unsigned i=0;i<errors[0].size();++i){
		for(unsigned j=0;j<errors.size();++j){
			double loss=errors[j][i];
			fprintf(F, "%0.7lf\t", loss);
		}
		fprintf(F, "\n");
	}
	fclose(F);
#endif
	
	
	choosePDDs(output, basisDirections);
	//normalizeToRange(output.error,ns*nr*nc,0,1);
	delete[] sumImage;
	delete[] orientations;
	delete[] s0Indices;
	delete[] dwVolume;
	delete[] s0Volume;
	delete[] dwSignal;
	delete[] basisDirections;
	delete[] Phi;
	return 0;
}

//it assumes that output already contains an initial DBF
int regularizeDBF(const DBF_params &params, double lambda, double mu, DBF_output &output, RegularizationCallbackFunction callback, void*callbackParam){
	double *orientations=NULL;
	int numOrientations=0;
	int *s0Indices=NULL;
	int numS0=0;
	double *basisDirections=NULL;
	int numBasisDirections=0;

	int nr, nc, ns;
	double *dwVolume=NULL;
	double *s0Volume=NULL;
	double *alpha=output.alphas;
	double longDiffusion=-1;
	double transDiffusion=-1;
	cerr<<"Not supported yet: needs to estimate longitudinal and transversal diffusion"<<endl;
	return -1;

	loadOrientations(params.orientationsFileName,orientations, numOrientations, s0Indices, numS0);
	loadDiffusionDirections(params.dbfDirectionsFileName, basisDirections, numBasisDirections);
	loadDWMRIFiles(params.dwNames, s0Indices, numS0, s0Volume, dwVolume, nr, nc, ns);
	regularizeDBF(alpha, lambda, mu, 
				  orientations, numOrientations,
				  basisDirections, numBasisDirections,
				  dwVolume, s0Volume, ns, nr, nc, 
				  params.b, longDiffusion, transDiffusion,
				  callback, callbackParam);
	DELETE_ARRAY(orientations);
	DELETE_ARRAY(s0Indices);
	DELETE_ARRAY(basisDirections);
	DELETE_ARRAY(dwVolume);
	DELETE_ARRAY(s0Volume);
	return 0;
}

int regularizeDBF(double *alpha, double lambda, double mu, 
				  double *orientations, int numOrientations,
				  double *basisDirections, int numBasisDirections,
				  double *dwVolume, double *s0Volume, int ns, int nr, int nc, 
				  double bParam, double longDiffusion, double transDiffusion,
				  RegularizationCallbackFunction callback, void*callbackParam){
	DBF dbfInstance(bParam, longDiffusion, transDiffusion,orientations, numOrientations, basisDirections, numBasisDirections);

	double *Phi=dbfInstance.getDiffusionBasis();
	double *PP=new double[numBasisDirections*numBasisDirections];
	double *Ar=new double[numBasisDirections*numBasisDirections];
	double *br=new double[numBasisDirections];


	//compute Phi^T*Phi + lambda*V^TV
	for(int i=0;i<numBasisDirections;++i){
		for(int j=i;j<numBasisDirections;++j){
			double *phi_i=&Phi[i];
			double *phi_j=&Phi[j];
			double &sum=PP[i*numBasisDirections+j];
			sum=-lambda/numBasisDirections;
			for(int k=0;k<numOrientations;++k, phi_i+=numBasisDirections, phi_j+=numBasisDirections){
				sum+=(*phi_i)*(*phi_j);
			}
			if(i==j){
				sum+=lambda;
			}
			PP[j*numBasisDirections+i]=sum;
		}
	}

	double s0=0;
	double *dwSignal=new double[numOrientations];
	//compute Phi^T*s
	int numVoxels=nr*nc*ns;
	double **b=new double*[numVoxels];//one vector per voxel
	for(int ps=0;ps<ns;++ps){
		for(int pr=0;pr<nr;++pr){
			for(int pc=0;pc<nc;++pc){
				int pos=ps*(nr*nc)+(nr-1-pr)*nc+pc;
				//if(params.mask[pos]){
					getDWSignalAtVoxel(s0Volume, dwVolume, nr, nc, ns, numOrientations, pr, pc, ps, s0, dwSignal);
					if(s0>EPSILON){
						multVectorScalar(dwSignal, 1.0/s0, numOrientations, dwSignal);
					}
					double *&bCurrent=b[ps*(nr*nc)+pr*nc+pc];
					bCurrent=new double[numBasisDirections];
					multVectorMatrix(dwSignal, Phi, numOrientations, numBasisDirections,bCurrent);
				/*}else{
					b[ps*(nr*nc)+pr*nc+pc]=NULL;
				}*/
			}
		}
	}
	double *Wrs=new double[numBasisDirections];
	for(int i=0;i<numBasisDirections;++i){
		Wrs[i]=1;
	}
	for(int iter=0;iter<100;++iter){
		cerr<<"Iteration "<<iter+1<<"/"<<100<<endl;
		for(int ps=0;ps<ns;++ps){
			for(int pr=0;pr<nr;++pr){
				for(int pc=0;pc<nc;++pc){
					int pos=ps*(nr*nc)+(nr-1-pr)*nc+pc;
					//if(params.mask[pos]){
						memcpy(br, b[ps*(nr*nc)+pr*nc+pc], sizeof(double)*numBasisDirections);
						memcpy(Ar, PP, sizeof(double)*numBasisDirections*numBasisDirections);
						double *alpha_r=&alpha[numBasisDirections*(ps*(nr*nc)+pr*nc+pc)];
						int neighborCount=0;
						for(int k=0;k<NUM_NEIGHBORS;++k){
							int ss=ps+dSlice[k];
							int rr=pr+dRow[k];
							int cc=pc+dCol[k];
							if(IN_RANGE(ss, 0, ns) && IN_RANGE(rr, 0, nr) && IN_RANGE(cc, 0, nc)){
								double *alpha_s=&alpha[numBasisDirections*(ps*(nr*nc)+pr*nc+pc)];
								++neighborCount;
								for(int j=0;j<numBasisDirections;++j){
									br[j]+=mu*alpha_s[j];
								}
							}
						}
						double *A=Ar;
						for(int k=0;k<numBasisDirections;++k, A+=numBasisDirections+1){
							(*A)+=neighborCount*mu;
						}
						//solve with nonnegativity constraints
						double errorNNLS;
						nnls(Ar,numBasisDirections, numBasisDirections, br,alpha_r,&errorNNLS);
					//}
				}
			}
		}
		if(callback!=NULL){
			callback(callbackParam);
		}
	}
	delete[] orientations;
	delete[] dwVolume;
	delete[] s0Volume;
	delete[] dwSignal;
	delete[] basisDirections;
	delete[] PP;
	delete[] Ar;
	delete[] br;
	delete[] Wrs;

	return 0;

}

void computeBasisDists(double *pdds, int m, double *D){
	for(int i=0;i<m;++i){
		D[i*m+i]=0;
		double *pddi=&pdds[3*i];
		for(int j=0;j<i;++j){
			double *pddj=&pdds[3*j];
			double p=fabs(dotProduct(pddi, pddj, 3));
			p=MAX(p,-1);
			p=MIN(p,1);
			double d=acos(p);
			D[i*m+j]=D[j*m+i]=d;
		}
		
	}
}

double emd_distance_alphas(double *a, double *b, int n, double *D){
	int na=0;
	int nb=0;
	vector<int> idxa;
	vector<int> idxb;
	for(int i=0;i<n;++i){
		if(a[i]>EPS_DBF){
			idxa.push_back(i);
			++na;
		}
	}
	for(int i=0;i<n;++i){
		if(b[i]>EPS_DBF){
			idxb.push_back(i);
			++nb;
		}
	}
	int sz=na+nb;
	vector<double>P(sz,0);
	vector<double>Q(sz,0);
	int pa=0;
	int pb=0;
	for(int i=0;i<n;++i){
		if(a[i]>EPS_DBF){
			P[pa]=a[i];
			++pa;
		}
		if(b[i]>EPS_DBF){
			Q[na+pb]=b[i];
			++pb;
		}
	}
	vector<vector<double> > subD(sz,vector<double>(sz,0));
	for(int i=0;i<na;++i){
		for(int j=0;j<nb;++j){
			double d=D[idxa[i]*n + idxb[j]];
			subD[i][na+j]=d;
			subD[na+j][i]=d;
		}
	}
	double res=emd_hat<double,NO_FLOW>()(P,Q,subD, -1);
	return res;
}

//------------alram's regularization--------
void weightProyectionOnDT(double *HRT_coeffs, int nvoxels, int numCoeffPerTensor, double *DBFDirections, int numDBFDirections, double *WPT){
	for(int v=0;v<nvoxels;++v){
		double *tens=&HRT_coeffs[v*numCoeffPerTensor];
		double T[9]={tens[0], tens[3], tens[4],
					tens[3], tens[1], tens[5],
					tens[4], tens[5], tens[2]};
		double trace=tens[0]+tens[1]+tens[2];
		double T2[9]={	trace-tens[0],	-tens[3],		-tens[4],
						-tens[3],		trace-tens[1],	-tens[5],
						-tens[4],		-tens[5],		trace-tens[2]};
		double maxV=-1e10;
		double minV=1e10;
		double *currentWPT=&WPT[v*numDBFDirections];
		for(int i=0;i<numDBFDirections;++i){
			double *dir=&DBFDirections[3*i];
			double eval=evaluateQuadraticForm(T2,dir, 3);
			currentWPT[i]=eval;
			maxV=MAX(maxV, eval);
			minV=MIN(minV, eval);
		}
		double diff=maxV-minV;
		if(diff>EPS_DBF){
			for(int i=0;i<numDBFDirections;++i){
				currentWPT[i]=(currentWPT[i]-minV)/(diff);
			}
		}
	}
}
void computeDWDiffSmooth(double *alphaVolume, int nslices, int nrows, int ncols, double *DBFDirections, int numDBFDirections, double *DWeigh){
	double cosTh=cos(5*M_PI/180);
	int nvoxels=nslices*nrows*ncols;
	int *noXYReg=new int[nvoxels*3];
	double *alphaRSm=new double[numDBFDirections];
	double *alphaSSm=new double[numDBFDirections];
	for(int i=nvoxels*3-1;i>=0;--i){
		noXYReg[i]=1;
	}
	double *D=new double[numDBFDirections*numDBFDirections];
	computeBasisDists(DBFDirections, numDBFDirections, D);
	double maxDWeight=-1e10;
	memset(DWeigh, 0, sizeof(double)*nvoxels*3);
	for(int py=0;py<nrows;++py){
		for(int px=0;px<ncols;++px){
			for(int pz=0;pz<nslices;++pz){
				int pos=pz*(nrows*ncols)+py*ncols+px;
				double *alphaR=&alphaVolume[pos*numDBFDirections];
				double *alphaS=NULL;
				int *xyReg=&noXYReg[pos*3];
				double *currentDWeight=&DWeigh[pos*3];
				for(int v=0;v<3;++v){
					switch(v){
						case 0:
							if(px==ncols-1){
								xyReg[0]=0;
							}else{
								int spos=pz*(nrows*ncols)+py*ncols+px+1;
								alphaS=&alphaVolume[numDBFDirections*spos];
							}
						break;
						case 1:
							if(py==nrows-1){
								xyReg[1]=0;
							}else{
								int spos=pz*(nrows*ncols)+(py+1)*ncols+px;
								alphaS=&alphaVolume[numDBFDirections*spos];
							}
						break;
						case 2:
							if(pz==nslices-1){
								xyReg[2]=0;
							}else{
								int spos=(pz+1)*(nrows*ncols)+py*ncols+px;
								alphaS=&alphaVolume[numDBFDirections*spos];
							}
						break;
					}

					if(alphaS==NULL){
						continue;
					}
#ifdef USE_EMD
					double d=emd_distance_alphas(alphaR, alphaS, numDBFDirections, D);
					currentDWeight[v]=d;
#else
					memset(alphaRSm, 0, sizeof(double)*numDBFDirections);
					memset(alphaSSm, 0, sizeof(double)*numDBFDirections);
					double *p=DBFDirections;
					
					for(int i=0;i<numDBFDirections;++i,p+=3){
						double *q=DBFDirections;
						int nConsidered=0;
						for(int j=0;j<numDBFDirections;++j, q+=3){
							double prod=fabs(dotProduct(p,q,3));
							if(prod>cosTh){
								alphaRSm[i]+=alphaR[j]*prod;
								alphaSSm[i]+=alphaS[j]*prod;
								++nConsidered;
							}
						}
						//cerr<<"Considered:"<<nConsidered<<endl;
					}
					double ed=euclideanDistanceSQR(alphaR, alphaSSm, numDBFDirections);
					/*double ed=0;
					for(int i=0;i<numDBFDirections;++i){
						ed+=(1.0-alphaSSm[i])*fabs(alphaR[i]-alphaS[i]);
					}*/
					currentDWeight[v]=ed;
#endif
					maxDWeight=MAX(maxDWeight, currentDWeight[v]);
				}//for v
			}//for pz
		}//for px
	}//for py
	delete[] D;
	if(fabs(maxDWeight)>EPS_DBF){
		int len=nvoxels*3;
		for(int i=0;i<len;++i){
			DWeigh[i]=(1-DWeigh[i]/maxDWeight)*noXYReg[i];
			//DWeigh[i]/=maxDWeight;
		}
	}
	delete[] alphaRSm;
	delete[] alphaSSm;
	delete[] noXYReg;
}

double robustSmoothDBF(MultiTensorField &reconstructed, double longDiffusion, double transDiffusion, double *dwVolume, double *s0Volume, int numGradients, int nItera, double lambda, double stp, 
					   double *DBF, double *DBFDirections, int numDBFDirections, int nIteraInterna,
					   double *alphaVolume0, double *DWeight, double *WPT, double lWPT, double lc, RegularizationCallbackFunction callback, void*callbackParam){
	int ns=reconstructed.getNumSlices();
	int nr=reconstructed.getNumRows();
	int nc=reconstructed.getNumCols(); 
	double *sumPhiSq=new double[numDBFDirections];
	double *A=new double[numGradients];
	vector<set<int> > neighborhoods;
	double *RES_pdds=NULL;
	double *RES_amount=NULL;
	int RES_count=0;
	if(callback!=NULL){
		buildNeighborhood(DBFDirections, numDBFDirections, 15, neighborhoods);
		RES_pdds=new double[3*numDBFDirections];
		RES_amount=new double[numDBFDirections];
		RES_count;
	}

	
	for(int i=0;i<numDBFDirections;++i){
		sumPhiSq[i]=0;
		double *currentDBF=&DBF[i];
		for(int j=0;j<numGradients;++j, currentDBF+=numDBFDirections){
			sumPhiSq[i]+=SQR(*currentDBF);
		}
	}
	int nvoxels=ns*nr*nc;
	double *currentS=dwVolume;
	double *sumAPhi=new double[nvoxels*numDBFDirections];
	double *sumPhiSumPhi=new double[numDBFDirections*numDBFDirections];
	double *currentSumAPhi=sumAPhi;
	for(int v=0;v<nvoxels;++v, currentS+=numGradients, currentSumAPhi+=numDBFDirections){
		double currentS0=s0Volume[v];
		multVectorScalar<double>(currentS,1.0/currentS0, numGradients, A);
		multVectorMatrix<double>(A,DBF,numGradients, numDBFDirections, currentSumAPhi); 
	}

	for(int i=0;i<numDBFDirections;++i){
		for(int j=i;j<numDBFDirections;++j){
			double *phi_i=&DBF[i];
			double *phi_j=&DBF[j];
			double &sum=sumPhiSumPhi[i*numDBFDirections+j];
			sum=0;
			for(int k=0;k<numGradients;++k, phi_i+=numDBFDirections, phi_j+=numDBFDirections){
				sum+=(*phi_i)*(*phi_j);
			}
			sumPhiSumPhi[j*numDBFDirections+i]=sum;
		}
	}
	multVectorScalar(WPT, lWPT, nvoxels*numDBFDirections,WPT);

	for(int iter=1;iter<=nItera;iter+=nIteraInterna){
		//=================callback==============
		if(callback!=NULL){
			MultiTensor *voxE=reconstructed.getVoxels();
			//---build MultiTensor instance from the solution---
			for(int v=0;v<nvoxels;++v){
				double *currentAlpha=&alphaVolume0[v*numDBFDirections];
				groupCoefficients(currentAlpha, DBFDirections, numDBFDirections, neighborhoods, RES_pdds, RES_amount, RES_count);
				//get big peaks
				getBigPeaks(0.2, RES_pdds, RES_amount, RES_count);
				voxE[v].dellocate();
				voxE[v].allocate(RES_count);
				for(int i=0;i<RES_count;++i){
					voxE[v].setDiffusivities(i, transDiffusion, transDiffusion, longDiffusion);
				}
				voxE[v].setVolumeFractions(RES_amount);
				for(int k=0;k<RES_count;++k){
					voxE[v].setRotationMatrixFromPDD(k,&RES_pdds[3*k]);
				}
			}
			callback(callbackParam);
		}
		//=======================================
		double dif=0;
		for(int py=0;py<nr;++py){
			for(int px=0;px<nc;++px){
				for(int pz=0;pz<ns;++pz){
					int pos=pz*(nr*nc)+py*nc+px;
					currentSumAPhi=&sumAPhi[pos*numDBFDirections];
					double *currentWPT=&WPT[pos*numDBFDirections];
					double *currentAlpha0=&alphaVolume0[pos*numDBFDirections];
					double meanA=getMean(currentAlpha0,numDBFDirections);
					for(int iterInterna=0;iterInterna<nIteraInterna;++iterInterna){
						for(int k=0;k<numDBFDirections;++k){
							double sumKAs=dotProduct(&sumPhiSumPhi[k*numDBFDirections],currentAlpha0, numDBFDirections);
							sumKAs-=currentAlpha0[k]*sumPhiSumPhi[k*numDBFDirections+k];
							double sumWA = 0; 
							double sumW  = 0;
							if(py>0){
								int npos=pz*(nr*nc)+(py-1)*nc+px;
								sumWA+=DWeight[3*npos+1]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*npos+1];
							}
							if(py<nr-1){
								int npos=pz*(nr*nc)+(py+1)*nc+px;
								sumWA+=DWeight[3*pos+1]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*pos+1];
							}
							if(px>0){
								int npos=pz*(nr*nc)+py*nc+px-1;
								sumWA+=DWeight[3*npos]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*npos];
							}
							if(px<nc-1){
								int npos=pz*(nr*nc)+py*nc+px+1;
								sumWA+=DWeight[3*pos]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*pos];
							}
							if(pz>0){
								int npos=(pz-1)*(nr*nc)+py*nc+px;
								sumWA+=DWeight[3*npos+2]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*npos+2];
							}
							if(pz<ns-1){
								int npos=(pz+1)*(nr*nc)+py*nc+px;
								sumWA+=DWeight[3*pos+2]*alphaVolume0[numDBFDirections*npos+k];
								sumW+=DWeight[3*pos+2];
							}
							
							double nVal=(currentSumAPhi[k]-sumKAs+lambda*sumWA-lc*meanA)/(sumPhiSq[k]+lambda*sumW+currentWPT[k]-lc);
							if(nVal<0){
								nVal=0;
							}
							dif+=SQR(currentAlpha0[k]-nVal);
							currentAlpha0[k]=nVal;
						}
					}

				}
			}
		}

		dif/=(nvoxels*numDBFDirections);
		if(dif<=stp){
			//cerr<<"Convergio en "<<iter<<endl;
			break;
		}
		//cerr<<"iter "<<iter<<": "<<dif<<endl;
		
	}

	delete[] sumPhiSumPhi;
	delete[] sumPhiSq;
	delete[] A;
	delete[] sumAPhi;
	if(callback!=NULL){
		delete[] RES_pdds;
		delete[] RES_amount;
	}
	return 0;
}



void computeRegularizationMatrix(double *DBFDirections, int numDBFDirections, double *QtQ){
	for(int i=0;i<numDBFDirections;++i){
		double *p=&DBFDirections[3*i];
		double maxVal=0;
		for(int j=0;j<numDBFDirections;++j){
			double *q=&DBFDirections[3*j];
			double prod=fabs(dotProduct(p,q,3));
			prod=pow(prod, 500.);
			/*if(prod<.9){
				prod=0;
			}*/
			QtQ[i*numDBFDirections+j]=prod;
			maxVal=MAX(maxVal, prod);
		}
		for(int j=0;j<numDBFDirections;++j){
			QtQ[i*numDBFDirections+j]/=maxVal;
		}
	}
	multMatrixMatrix(QtQ,QtQ,numDBFDirections,QtQ);
}

void computeGiniMatrix(int N, double *VtV){
	double offDiag=-1.0/N;
	int len=N*N;
	for(int i=0;i<len;++i){
		VtV[i]=offDiag;
	}
	for(int i=0;i<N;++i){
		VtV[i*(N+1)]+=1;
	}
	multMatrixMatrix<double>(VtV,VtV,N,VtV);
}

double robustSmoothDBF_joog(MultiTensorField &reconstructed, double longDiffusion, double transDiffusion, double *dwVolume, double *s0Volume, int numGradients, int nItera, double lambda, double stp, 
					   double *DBF, double *DBFDirections, int numDBFDirections, int nIteraInterna,
					   double *alphaVolume0, double *DWeight, double *WPT, double lWPT, double lc, RegularizationCallbackFunction callback, void*callbackParam){
	int ns=reconstructed.getNumSlices();
	int nr=reconstructed.getNumRows();
	int nc=reconstructed.getNumCols(); 
	double *sumPhiSq=new double[numDBFDirections];
	
	vector<set<int> > neighborhoods;
	double *RES_pdds=NULL;
	double *RES_amount=NULL;
	int RES_count=0;
	if(callback!=NULL){
		buildNeighborhood(DBFDirections, numDBFDirections, 15, neighborhoods);
		RES_pdds=new double[3*numDBFDirections];
		RES_amount=new double[numDBFDirections];
		RES_count;
	}

	
	for(int i=0;i<numDBFDirections;++i){
		sumPhiSq[i]=0;
		double *currentDBF=&DBF[i];
		for(int j=0;j<numGradients;++j, currentDBF+=numDBFDirections){
			sumPhiSq[i]+=SQR(*currentDBF);
		}
	}
	int nvoxels=ns*nr*nc;
	double *currentS=dwVolume;
	double *sumAPhi=new double[nvoxels*numDBFDirections];
	double *sumPhiSumPhi=new double[numDBFDirections*numDBFDirections];
	double *QtQ=new double[numDBFDirections*numDBFDirections];
	double *VtV=new double[numDBFDirections*numDBFDirections];
	computeRegularizationMatrix(DBFDirections, numDBFDirections, QtQ);
	//setIdentity(QtQ,numDBFDirections);
	
	computeGiniMatrix(numDBFDirections, VtV);
	double *currentSumAPhi=sumAPhi;

	double *snorm=new double[numGradients];
	for(int v=0;v<nvoxels;++v, currentS+=numGradients, currentSumAPhi+=numDBFDirections){
		double currentS0=s0Volume[v];
		multVectorScalar<double>(currentS,1.0/currentS0, numGradients, snorm);
		multVectorMatrix<double>(snorm,DBF,numGradients, numDBFDirections, currentSumAPhi); 
	}

	for(int i=0;i<numDBFDirections;++i){
		for(int j=i;j<numDBFDirections;++j){
			double *phi_i=&DBF[i];
			double *phi_j=&DBF[j];
			double &sum=sumPhiSumPhi[i*numDBFDirections+j];
			sum=0;
			for(int k=0;k<numGradients;++k, phi_i+=numDBFDirections, phi_j+=numDBFDirections){
				sum+=(*phi_i)*(*phi_j);
			}
			sumPhiSumPhi[j*numDBFDirections+i]=sum;
		}
	}

	double *A=new double[numDBFDirections*numDBFDirections];
	double *b=new double[numDBFDirections];
	for(int iter=1;iter<=nItera;iter+=nIteraInterna){
		//=================callback==============
		if(callback!=NULL){
			MultiTensor *voxE=reconstructed.getVoxels();
			//---build MultiTensor instance from the solution---
			for(int v=0;v<nvoxels;++v){
				double *currentAlpha=&alphaVolume0[v*numDBFDirections];
				groupCoefficients(currentAlpha, DBFDirections, numDBFDirections, neighborhoods, RES_pdds, RES_amount, RES_count);
				//get big peaks
				getBigPeaks(0.2, RES_pdds, RES_amount, RES_count);
				voxE[v].dellocate();
				voxE[v].allocate(RES_count);
				for(int i=0;i<RES_count;++i){
					voxE[v].setDiffusivities(i, transDiffusion, transDiffusion, longDiffusion);
				}
				voxE[v].setVolumeFractions(RES_amount);
				for(int k=0;k<RES_count;++k){
					voxE[v].setRotationMatrixFromPDD(k,&RES_pdds[3*k]);
				}
			}
			callback(callbackParam);
		}
		//=======================================
		double dif=0;
		for(int py=0;py<nr;++py){
			for(int px=0;px<nc;++px){
				for(int pz=0;pz<ns;++pz){
					int pos=pz*(nr*nc)+py*nc+px;
					currentSumAPhi=&sumAPhi[pos*numDBFDirections];
					double *currentWPT=&WPT[pos*numDBFDirections];
					double *currentAlpha0=&alphaVolume0[pos*numDBFDirections];
					double meanA=getMean(currentAlpha0,numDBFDirections);
					//----build A and b----
					memcpy(b,currentSumAPhi,sizeof(double)*numDBFDirections);
							double sumW  = 0;
							if(py>0){
								int npos=pz*(nr*nc)+(py-1)*nc+px;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*npos+1];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*npos+1];
							}
							if(py<nr-1){
								int npos=pz*(nr*nc)+(py+1)*nc+px;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*pos+1];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*pos+1];
							}
							if(px>0){
								int npos=pz*(nr*nc)+py*nc+px-1;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*npos];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*npos];
							}
							if(px<nc-1){
								int npos=pz*(nr*nc)+py*nc+px+1;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*pos];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*pos];
							}
							if(pz>0){
								int npos=(pz-1)*(nr*nc)+py*nc+px;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*npos+2];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*npos+2];
							}
							if(pz<ns-1){
								int npos=(pz+1)*(nr*nc)+py*nc+px;
								double *alpha_s=&alphaVolume0[numDBFDirections*npos];
								double factor=lambda*DWeight[3*pos+2];
								linCombVector<double>(alpha_s, factor, b, numDBFDirections, b);
								sumW+=DWeight[3*pos+2];
							}
					multMatrixVector(QtQ,b,numDBFDirections,numDBFDirections,b);
					for(int i=numDBFDirections*numDBFDirections-1;i>=0;--i){
						A[i]=sumPhiSumPhi[i]+lambda*sumW*QtQ[i]-lambda*VtV[i];
						if(i%(numDBFDirections+1)==0){
							A[i]+=lambda*currentWPT[i/(numDBFDirections+1)];
						}
					}
					//---------------------
					for(int iterInterna=0;iterInterna<nIteraInterna;++iterInterna){
						for(int k=0;k<numDBFDirections;++k){
							double sumKAs=dotProduct(&A[k*numDBFDirections],currentAlpha0, numDBFDirections);
							sumKAs-=currentAlpha0[k]*A[k*numDBFDirections+k];
							double nVal=(b[k]-sumKAs)/A[k*numDBFDirections+k];
							if(nVal<0){
								nVal=0;
							}
							dif+=SQR(currentAlpha0[k]-nVal);
							currentAlpha0[k]=nVal;
						}
					}
				}
			}
		}
		dif/=(nvoxels*numDBFDirections);
		if(dif<=stp){
			//cerr<<"Convergio en "<<iter<<endl;
			break;
		}
		//cerr<<"iter "<<iter<<": "<<dif<<endl;
		
	}

	delete[] sumPhiSumPhi;
	delete[] sumPhiSq;
	delete[] A;
	delete[] sumAPhi;
	if(callback!=NULL){
		delete[] RES_pdds;
		delete[] RES_amount;
	}
	delete[] QtQ;
	delete[] VtV;
	delete[] snorm;
	return 0;
}

int regularizeDBF_alram(double *alphaVolume, MultiTensorField &reconstructed, double longDiffusion, double transDiffusion, double *HRT_coeffs,
					DBF &dbfInstance, double *dwVolume, double *s0Volume, int ns, int nr, int nc, 
					RegularizationCallbackFunction callback, void*callbackParam){

	double *gradients=dbfInstance.getGradients(); 
	int numGradients=dbfInstance.getNumGradients();
	double *DBFDirections=dbfInstance.getDiffusionDirections(); 
	int numDBFDirections=dbfInstance.getNumBasisFunctions();
	double *DBF=dbfInstance.getDiffusionBasis();

	int nItera = 1000;
    //double lambda = 114.5e-2;
	double lambdaIni = 4.5e-2;
	double lambda=lambdaIni;
    double stp = 1e-8;
    double lWPT = lambda;
    double lc = lambda;
    int nIteraInterna = 4;
    int nTimesReg = 3;
	int nvoxels=ns*nr*nc;
	double *WPT=new double[numDBFDirections*nvoxels];
	double *DWeight=new double[3*nvoxels];
	weightProyectionOnDT(HRT_coeffs, nvoxels, 6, DBFDirections, numDBFDirections, WPT);
	computeDWDiffSmooth(alphaVolume, ns, nr, nc, DBFDirections, numDBFDirections, DWeight);
#ifdef USE_JOOG_REG
	cerr<<"Using joog regularization..."<<endl;
#endif
	cerr<<"Reg. step "<<1<<endl;
	
	
	
#ifdef USE_JOOG_REG
	robustSmoothDBF_joog(reconstructed, longDiffusion, transDiffusion, dwVolume, s0Volume, numGradients, nItera, 
					lambda,stp,DBF, DBFDirections, numDBFDirections, nIteraInterna, alphaVolume, DWeight, WPT, lWPT, lc, callback, callbackParam);
#else
	robustSmoothDBF(reconstructed, longDiffusion, transDiffusion, dwVolume, s0Volume, numGradients, nItera, 
		lambda,stp,DBF, DBFDirections, numDBFDirections, nIteraInterna, alphaVolume, DWeight, WPT, lWPT, lc, callback, callbackParam);
#endif

	for(int i=2;i<=nTimesReg;++i){
		cerr<<"Reg. step "<<i<<endl;
		lambda+=lambdaIni;
        lWPT = lambda;
        lc = lambda;
		computeDWDiffSmooth(alphaVolume, ns, nr, nc, DBFDirections, numDBFDirections, DWeight);
		
#ifdef USE_JOOG_REG
		robustSmoothDBF_joog(reconstructed, longDiffusion, transDiffusion, dwVolume, s0Volume, numGradients, nItera, 
					lambda,stp,DBF, DBFDirections, numDBFDirections, nIteraInterna, alphaVolume, DWeight, WPT, lWPT, lc, callback, callbackParam);
#else
		robustSmoothDBF(reconstructed, longDiffusion, transDiffusion, dwVolume, s0Volume, numGradients, nItera, 
							lambda,stp,DBF, DBFDirections, numDBFDirections, nIteraInterna, alphaVolume, DWeight, WPT, lWPT, lc, callback, callbackParam);
#endif
	}
	delete[] WPT;
	delete[] DWeight;
	return 0;
}
