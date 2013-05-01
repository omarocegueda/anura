#include "multitensorfieldexplorer.h"
#include <string>
#include <sys/stat.h>
#include <QErrorMessage>
#include <QFileDialog>
#include "DBF.h"
#include "GDTI.h"
#include "dtiutils.h"
#include "statisticsutils.h"
#include "geometryutils.h"
#include "nnls.h"
#include <time.h>
#include <iostream>
#include <sstream>
#include <cv.h>
#include <highgui.h>
#include "MultiTensorFieldExperimentParameters.h"
#include "expm.h"
#include "denoising.h"
#include "image_io.h"
#include "macros.h"
#include "mds.h"
#include "clustering.h"
#include "SparseMatrix.h"
#include "carpack.h"
#include "ecqmmf.h"
#include "eckqmmf.h"
#include "clustering.h"
#include "DirectoryListing.h"
#include <vector>
#include <queue>
#include <sstream>
#include "SphericalHarmonics.h"
#include "CSDeconv.h"
#include "DWMRISimulator.h"
#include <map>
#include "mrtrixio.h"
#include "bits.h"
using namespace std;



std::string chooseFileToRead(string title=""){
    QFileDialog dialog;
	QString s=dialog.getOpenFileName(NULL, title.c_str());
    return s.toLocal8Bit().data();
}

std::string chooseFileToWrite(string title=""){
    QFileDialog dialog;
	QString s=dialog.getSaveFileName(NULL, title.c_str());
    return s.toLocal8Bit().data();
}

int FileExists(const std::string& file_name) {
#if defined(_WIN32) || defined(_WIN64)
  struct _stat buffer;
  if (!_stat(file_name.c_str(), &buffer)) return 1;
#else
  struct stat buffer;
  if (!stat(file_name.c_str(), &buffer)) return 1;
#endif
  return 0;
}

void showSimpleErrorMessage(string msg){
	QErrorMessage errorMessage;
	errorMessage.showMessage(msg.c_str());
	errorMessage.exec();
}


MultiTensorFieldExplorer::MultiTensorFieldExplorer(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);
	connect(ui.updateGT_button, SIGNAL(clicked()), this,				SLOT(updateTensorViewGT()));
	connect(ui.updateRecovered_button, SIGNAL(clicked()), this,			SLOT(updateTensorViewRecovered()));
	connect(ui.runDBF, SIGNAL(clicked()), this,							SLOT(runDBF()));
	connect(ui.loadOnlyGT_button, SIGNAL(clicked()), this,				SLOT(loadOnlyGT()));
	connect(ui.loadOnlyRecovered_button, SIGNAL(clicked()), this,		SLOT(loadOnlyRecovered()));
	connect(ui.save_button, SIGNAL(clicked()), this,					SLOT(saveRecovered()));
	connect(ui.selectGT_button, SIGNAL(clicked()), this,				SLOT(selectVoxelFromGT()));
	connect(ui.selectRecovered_button, SIGNAL(clicked()), this,			SLOT(selectVoxelFromRecovered()));
	connect(ui.glViewerGT, SIGNAL(multiTensorClicked(MultiTensor *, int, int, int)), this,			SLOT(multiTensorSelectedGT(MultiTensor *, int, int, int)));
	connect(ui.glViewer, SIGNAL(multiTensorClicked(MultiTensor *, int, int, int)), this,			SLOT(multiTensorSelectedRecovered(MultiTensor *, int, int, int)));
	connect(ui.random_button, SIGNAL(clicked()), this,			SLOT(createRandomTensor()));
	connect(ui.alphaBehaviorStudy_button, SIGNAL(clicked()), this,			SLOT(alphaBehaviorStudy()));
	connect(ui.evaluate_button, SIGNAL(clicked()), this,			SLOT(evaluate()));
	connect(ui.compare_button, SIGNAL(clicked()), this,			SLOT(compare()));
	connect(ui.buildTrainingSet_button, SIGNAL(clicked()), this,			SLOT(buildTrainingSet()));
	connect(ui.testDenoising_button, SIGNAL(clicked()), this,			SLOT(testDenoising()));

	connect(ui.spectralClustering_button, SIGNAL(clicked()), this,			SLOT(runSpectralClustering()));

	connect(ui.mds_button, SIGNAL(clicked()), this,			SLOT(runMultiDimensionalScaling()));
	connect(ui.testMask_button, SIGNAL(clicked()), this,			SLOT(testMask()));
	connect(ui.gtFromPaths_button, SIGNAL(clicked()), this,			SLOT(gtFromPaths()));
	connect(ui.mtfToLCI, SIGNAL(clicked()), this,			SLOT(mtfToLCI()));
	connect(ui.LCIToMask, SIGNAL(clicked()), this,			SLOT(LCIToMask()));
	DBFDirections=NULL;
	numDBFDirections=0;
	if(fileExists(DEFAULT_DBF_FILE_NAME)){
		loadDiffusionDirections(DEFAULT_DBF_FILE_NAME, DBFDirections, numDBFDirections);
	}else{
		cerr<<"DBF directions file not found '"<<DEFAULT_DBF_FILE_NAME<<"'"<<endl;
	}
	

	ui.glViewer->tensorField.setDBFDirections(&DBFDirections);
	ui.glViewerGT->tensorField.setDBFDirections(&DBFDirections);



	//-----test CSDeconv--
	double response[]={203.224, -100.236, 32.2269, -7.1637, 1.37425};
	vector<double>responseCoefs(response, response+sizeof(response)/sizeof(double));

	double filter[]={1,1,1,0,0};
	vector<double>initFilter(filter, filter+sizeof(filter)/sizeof(double));

	/*CSDeconv deconv;
	double *gradientDirs=NULL;
	double *bs=NULL;
	int ngrads;
	double *constraintDirs=NULL;
	int nConstraints;
	bool csdReady=true;
	if(fileExists("constraint_dirs.txt")){
		deconv.readConstraintDirections("constraint_dirs.txt", constraintDirs, nConstraints);
	}else{
		cerr<<"CSD constraint directions file not found '"<<"constraint_dirs.txt"<<"'"<<endl;
		csdReady=false;
	}
	if(fileExists("gradient_list_048_b1500.txt")){
		deconv.readGradients("gradient_list_048_b1500.txt", gradientDirs, bs, ngrads);
	}else{
		cerr<<"CSD gradient list file not found '"<<"gradient_list_048_b1500.txt"<<"'"<<endl;
		csdReady=false;
	}
	

	const int lmax=8;
	if(csdReady){
		deconv.init(responseCoefs, initFilter, gradientDirs, ngrads, constraintDirs, nConstraints, lmax);
	}
	*/

	
}

MultiTensorFieldExplorer::~MultiTensorFieldExplorer()
{

}

void MultiTensorFieldExplorer::updateTensorViewRecovered(void){
	string s=ui.showSlicesRecovered_edit->text().toLocal8Bit().data();
	string p=ui.showPddsRecovered_edit->text().toLocal8Bit().data();
	ui.glViewer->setVisibleSlices(s);
	ui.glViewer->setVisibleTensors(p);
	MultiTensorFieldVisualizationType visType=getVisualizationType();
	ui.glViewer->tensorField.setShowGroupColors(ui.showGroupColors_check->isChecked());
	ui.glViewer->tensorField.setVisualizationType(visType);
	ui.glViewer->updateGL();
}

void MultiTensorFieldExplorer::updateTensorViewGT(void){
	string s=ui.showSlicesGT_edit->text().toLocal8Bit().data();
	string p=ui.showPddsGT_edit->text().toLocal8Bit().data();
	ui.glViewerGT->setVisibleSlices(s);
	ui.glViewerGT->setVisibleTensors(p);
	MultiTensorFieldVisualizationType visType=getVisualizationType();
	ui.glViewerGT->tensorField.setShowGroupColors(ui.showGroupColors_check->isChecked());
	ui.glViewerGT->tensorField.setVisualizationType(visType);
	ui.glViewerGT->updateGL();
	
}


MultiTensorFieldVisualizationType MultiTensorFieldExplorer::getVisualizationType(void){
	if(ui.orientation_radio->isChecked()){
		return MTFVT_Orientation;
	}else if(ui.sizeCompartment_radio->isChecked()){
		return MTFVT_Alpha;
	}else if(ui.error_radio->isChecked()){
		return MTFVT_Error;
	}else if(ui.sparcity_radio->isChecked()){
		return MTFVT_Sparcity;
	}else if(ui.arrows_radio->isChecked()){
		return MTFVT_Arrows;
	}else if(ui.apparentDiffusion_radio->isChecked()){
		return MTFVT_ApparentDiffusion;
	}else if(ui.clusterColors_radio->isChecked()){
		return MTFVT_ClusterColors;
	}else{
		return MTFVT_OrientationDiffusion;
	}
}

void computeAngleBounds(MultiTensor *voxels, int nVoxels, double &minAngle, double &maxAngle){
	minAngle=180;
	maxAngle=0;
	for(int v=0;v<nVoxels;++v){
		int mm=voxels[v].getNumCompartments();
		for(int i=0;i<mm;++i){
			for(int j=i+1;j<mm;++j){
				double p[3];
				double q[3];
				voxels[v].getPDD(i,p);
				voxels[v].getPDD(j,q);
				double angle=getAbsAngleDegrees(p,q,3);
				if(angle<minAngle){
					minAngle=angle;
					cerr<<"--------"<<endl;
					cerr<<"New min angle:"<<minAngle<<endl;
					for(int k=0;k<3;++k){
						cerr<<p[k]<<" ";
					}
					cerr<<endl;
					for(int k=0;k<3;++k){
						cerr<<q[k]<<" ";
					}
					cerr<<endl;
					cerr<<"--------"<<endl;
				}
				if(maxAngle<angle){
					maxAngle=angle;
					cerr<<"--------"<<endl;
					cerr<<"New max angle:"<<maxAngle<<endl;
					for(int k=0;k<3;++k){
						cerr<<p[k]<<" ";
					}
					cerr<<endl;
					for(int k=0;k<3;++k){
						cerr<<q[k]<<" ";
					}
					cerr<<endl;
					cerr<<"--------"<<endl;
				}
			}
		}
	}
}

void regularizationCallback(void *data){
	GLViewer *viewer=(GLViewer *)data;
	viewer->repaint();	
}

void MultiTensorFieldExplorer::loadOnlyGT(void){
	string fname=chooseFileToRead();
	if(fname.empty()){
		return;
	}
	ui.glViewerGT->tensorField.loadFromTxt(fname);
	MultiTensor *voxGT=ui.glViewerGT->tensorField.getVoxels();
	int nvoxels=ui.glViewerGT->tensorField.getNumSlices()*ui.glViewerGT->tensorField.getNumRows()*ui.glViewerGT->tensorField.getNumCols();
	map<int, int> M;
	bool firstTime=true;
	for(int i=0;i<nvoxels;++i, ++voxGT){
		int nc=voxGT->getNumCompartments();
		M[nc]++;
		if((nc>0)&&firstTime){
			firstTime=false;
			cerr<<"Diff. Profile: "<<voxGT->getDiffusivities()[0]<<", "<<voxGT->getDiffusivities()[1]<<", "<<voxGT->getDiffusivities()[2]<<endl;
		}
	}
	for(map<int, int>::iterator it=M.begin();it!=M.end();++it){
		cerr<<it->first<<" voxels: "<<it->second<<endl;
	}
	/*ui.glViewerGT->tensorField.saveAsNifti("challenge_gt_ntens.nii","challenge_gt_szcomp.nii", "challenge_gt_pdd.nii");
	MultiTensorField test;
	test.loadFromNifti("challenge_gt_ntens.nii","challenge_gt_szcomp.nii", "challenge_gt_pdd.nii");
	common_evaluate(ui.glViewerGT->tensorField, test);*/
}

void MultiTensorFieldExplorer::loadOnlyRecovered(void){
	string fname=chooseFileToRead();
	if(fname.empty()){
		return;
	}
	ui.glViewer->tensorField.loadFromTxt(fname);
	int numVoxels=ui.glViewer->tensorField.getNumSlices()*ui.glViewer->tensorField.getNumRows()*ui.glViewer->tensorField.getNumCols();
}

const bool sortByCrossingAngle(const MultiTensor *A, const MultiTensor *B){
	double a=A->getMinAngleDegrees();
	double b=B->getMinAngleDegrees();
	return a<b;
}


void MultiTensorFieldExplorer::multiTensorSelectedGT(MultiTensor *tensor, int slice, int row, int col){
	if(tensor==NULL){
		return;
	}
	if(tensor->getNumAlpha()>0){
		cv::Mat M;
		double *selectedDirs=tensor->getDirections();
		if(selectedDirs==NULL){
			selectedDirs=DBFDirections;
		}
		int w=1+512/tensor->getNumAlpha();
		if(ui.showGroupColors_check->isChecked()){
			showOrientationHistogram(tensor->getAlpha(), tensor->getGroups(), tensor->getNumAlpha(), selectedDirs, 100, tensor->getNumAlpha()*w, M);
		}else{
			showOrientationHistogram(tensor->getAlpha(), NULL, tensor->getNumAlpha(), selectedDirs, 100, tensor->getNumAlpha()*w, M);
		}
		
		cv::imshow("Alpha",M);
	}
	QString num;	
	ui.sliceGT_edit->setText(num.setNum(slice));
	ui.rowGT_edit->setText(num.setNum(row));
	ui.columnGT_edit->setText(num.setNum(col));
}

void MultiTensorFieldExplorer::multiTensorSelectedRecovered(MultiTensor *tensor, int slice, int row, int col){
	if(tensor==NULL){
		return;
	}
	if(tensor->getNumAlpha()>0){
		cv::Mat A;
		cv::Mat M;
		cv::Mat expM;
		double *selectedDirs=tensor->getDirections();
		if(selectedDirs==NULL){
			selectedDirs=DBFDirections;
		}
		int w=1+512/tensor->getNumAlpha();
		if(ui.showGroupColors_check->isChecked()){
			showOrientationHistogram(tensor->getAlpha(), tensor->getGroups(), tensor->getNumAlpha(), selectedDirs, 100, tensor->getNumAlpha()*w, A);
		}else{
			showOrientationHistogram(tensor->getAlpha(), NULL, tensor->getNumAlpha(), selectedDirs, 100, tensor->getNumAlpha()*w, A);
		}
		
		cv::imshow("Alpha",A);
		
		double *lambda=tensor->getDiffusivities(0);
		showOrientationSimilarityMatrix(tensor->getAlpha(), tensor->getNumAlpha(), selectedDirs,tensor->getNumAlpha()*4, tensor->getNumAlpha()*w, M, expM);
		cv::imshow("Sim. matrix",M);
		cv::imshow("Exp Sim. matrix",expM);
	}
	QString num;	
	ui.sliceRecovered_edit->setText(num.setNum(slice));
	ui.rowRecovered_edit->setText(num.setNum(row));
	ui.columnRecovered_edit->setText(num.setNum(col));
}


void MultiTensorFieldExplorer::getUIParameters(MultiTensorFieldExperimentParameters &params){
	params.SNR=ui.SNR_edit->text().toDouble();
	params.bigPeaksThreshold=ui.bigPeaksThr_edit->text().toDouble();
	params.clusteringNeighSize=ui.neighSize_edit->text().toInt();
	params.fitWithNeighbors=ui.fitWithNeighbors_check->isChecked();
	params.regularizeAlphaField=ui.regularizeIterative_check->isChecked();
	params.useFAIndicator=ui.useFAIndicator_check->isChecked();
	params.useGDTIPostProcessing=ui.useGDTIPostProc_check->isChecked();
	params.numSamplings=ui.sampling_edit->text().toInt();
	params.applyTensorSplitting=ui.applyTensorSplitting_check->isChecked();
	params.iterateDiffProf=ui.iterateDiffProf_check->isChecked();
	params.lambda=ui.lambda_edit->text().toDouble();
	params.alpha0=ui.alpha0_edit->text().toDouble();
	params.alpha1=ui.alpha1_edit->text().toDouble();
	params.spatialSmoothing=ui.spatialSmoothing_check->isChecked();
	
}


void computeDWMRIVariance(double *s0Volume, double *dwVolume, int nslices, int nrows, int ncols, int ngradients, double *var){
	int p=0;
	double *dwSignal=dwVolume;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c,++p, dwSignal+=ngradients){
				var[p]=0;
				if((!isNumber(s0Volume[p]))||(s0Volume[p]<1e-9)){
					continue;
				}
				double av=0;
				for(int i=0;i<ngradients;++i){
					av+=dwSignal[i]/s0Volume[p];
				}
				av/=ngradients;
				
				for(int i=0;i<ngradients;++i){
					var[p]+=SQR(av-dwSignal[i]);
				}
				var[p]/=(ngradients-1);
			}
		}
	}
}


void MultiTensorFieldExplorer::runDBF(void){
	string configFileName=chooseFileToRead();
	if(configFileName.empty()){
		return;
	}
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;
	getUIParameters(params);
	params.configFromFile(configFileName.c_str());

	params.reconstructed=&(ui.glViewer->tensorField);
	//---------
	params.loadData();
	int nvox=params.reconstructed->getNumRows()*params.reconstructed->getNumCols()*params.reconstructed->getNumSlices();
	//==========Experimental: dwi variance===============
	/*double *var=new double[nvox];
	computeDWMRIVariance(params.s0Volume, params.dwVolume, params.reconstructed->getNumSlices(), params.reconstructed->getNumRows(), params.reconstructed->getNumCols(), params.numGradients, var);
	write_mrtrix_image("var.mif", var, params.reconstructed->getNumSlices(), params.reconstructed->getNumRows(), params.reconstructed->getNumCols());
	delete[] var;*/
	//======================================================
	//----save vector field---
	/*int nrows=params.GT->getNumRows();
	int ncols=params.GT->getNumCols();
	FILE *F=fopen("vectorField.bin", "wb");
	fwrite(&nrows,sizeof(int), 1, F);
	fwrite(&ncols,sizeof(int), 1, F);
	double pdd[3];
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols;++j){
			MultiTensor *vox=params.GT->getVoxelAt(0,i,j);
			if(vox->getNumCompartments()>0){
				vox->getPDD(0, pdd);
			}else{
				pdd[0]=pdd[1]=0;
			}
			
			fwrite(pdd, sizeof(double), 2, F);
		}
	}
	fclose(F);
	//----save multi-vector field---
	F=fopen("multiVectorField.bin", "wb");
	int maxCompartments=3;
	fwrite(&nrows,sizeof(int), 1, F);
	fwrite(&ncols,sizeof(int), 1, F);
	fwrite(&maxCompartments,sizeof(int), 1, F);
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols;++j){
			for(int k=0;k<maxCompartments;++k){
				MultiTensor *vox=params.GT->getVoxelAt(0,i,j);
				if(k < vox->getNumCompartments()){
					vox->getPDD(0, pdd);
				}else{
					pdd[0]=pdd[1]=0;
				}
				fwrite(pdd, sizeof(double), 2, F);
			}
		}
	}
	fclose(F);*/
	//------------------------------
	testMultiTensorField(params, regularizationCallback, (void*)(ui.glViewer));
	
	double *LCI=new double[nvox];
	params.reconstructed->computeLocalCoherenceIndex(180, LCI);
	write_mrtrix_image("challenge_testing_SNR30_LCI.mif", LCI, params.reconstructed->getNumSlices(), params.reconstructed->getNumRows(), params.reconstructed->getNumCols());
	save3DNifti("challenge_testing_SNR30_LCI.nii", LCI, params.reconstructed->getNumSlices(), params.reconstructed->getNumRows(), params.reconstructed->getNumCols()); 
	delete[] LCI;
	
	

	params.reconstructed->saveToTxt(params.reconstructionFname);
	params.reconstructed=NULL;
	if(params.groundTruthType!=MTFGTT_None){
		ui.glViewerGT->tensorField.copyFrom(*params.GT);
		ui.glViewerGT->update();
	}
	ui.glViewer->update();
	
	
}
/*
void MultiTensorFieldExplorer::runEvaluationFromNifti(void){
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;
	params.b=1500;
	params.basisDirsFname=DEFAULT_DBF_FILE_NAME;
	params.bigPeaksThreshold=0.3;
	params.clusteringNeighSize=16;
	params.fitWithNeighbors=true;
	params.regularizeAlphaField=false;
	params.gradientDirsFname="gradient_list_048.txt";
	params.ODFDirFname="ODF_XYZ.txt";

	//---ground truth and input data---
	//params.nTensFname="TestingData\\nTens_Testing_SF_all_SNR_70_nsamples_048_method_NNLS_basisSize_129_b1500.nii";
	//params.sizeCompartmentFname="TestingData\\sizeCompartment_Testing_SF_all_SNR_70_nsamples_048_method_NNLS_basisSize_129_b1500.nii";
	//params.fPDDFname="TestingData\\fPDD_Testing_SF_all_SNR_70_nsamples_048_method_NNLS_basisSize_129_b1500.nii";
	params.nTensFname="TestingData\\GT26Neigh_nTens.nii";
	params.sizeCompartmentFname="TestingData\\GT26Neigh_sizeCompartment.nii";
	params.fPDDFname="TestingData\\GT26Neigh_fPDD.nii";

	params.rawDataBaseDir="TestingData\\";
	params.dwListFname="ListOfFiles_Testing_SF_SNR_10_nsamples_048_b1500.txt";
	params.s0Fname="S0_Testing_SF_SNR_10_nsamples_048_b1500.nii";
	//-----------------------------------
	
	//params.inputType=MTFIT_GTGenerator;
	//params.groundTruthType=MTFGTT_GTGenerator;

	params.inputType=MTFIT_DWMRI;
	params.groundTruthType=MTFGTT_NIFTI;

	params.solverType=DBFS_NNLS;
	params.reconstructed=&(ui.glViewer->tensorField);
	params.loadData();
	testMultiTensorField(params, regularizationCallback, (void*)(ui.glViewer));
	params.reconstructed=NULL;
	ui.glViewer->update();
}
*/

void MultiTensorFieldExplorer::selectVoxelFromGT(void){
	int slice=0, row=0, col=0;
	getSelectedVoxelPositionGT(slice, row, col);
	MultiTensorField newField;
	newField.allocate(1,1,1);
	MultiTensor *voxSel=newField.getVoxels();
	MultiTensor *selected=ui.glViewerGT->tensorField.getVoxelAt(slice, row, col);
	if(selected==NULL){
		return;
	}
	/*selected->setDiffusivities(0,0.0001,0.0001,0.0007);
	selected->setDiffusivities(1,0.0003,0.0003,0.0007);*/
	voxSel[0].copyFrom(*selected);
	ostringstream os;
	os<<"selectedVoxel_GT_"<<slice<<"_"<<row<<"_"<<col<<".txt";
	newField.saveToTxt(os.str());
}


void MultiTensorFieldExplorer::selectVoxelFromRecovered(void){
	int slice=0, row=0, col=0;
	getSelectedVoxelPositionRecovered(slice, row, col);
	MultiTensorField newField;
	newField.allocate(1,1,1);
	MultiTensor *voxSel=newField.getVoxels();
	MultiTensor *selected=ui.glViewer->tensorField.getVoxelAt(slice, row, col);
	if(selected==NULL){
		return;
	}
	//----
		if(selected->getNumAlpha()>0){
			cv::Mat A;
			cv::Mat M;
			cv::Mat expM;
			double *selectedDirs=selected->getDirections();
			if(selectedDirs==NULL){
				selectedDirs=DBFDirections;
			}
			int w=1+512/selected->getNumAlpha();
			if(ui.showGroupColors_check->isChecked()){
				showOrientationHistogram(selected->getAlpha(), selected->getGroups(), selected->getNumAlpha(), selectedDirs, 100, selected->getNumAlpha()*w, A);
			}else{
				showOrientationHistogram(selected->getAlpha(), NULL, selected->getNumAlpha(), selectedDirs, 100, selected->getNumAlpha()*w, A);
			}
			
			cv::imshow("Alpha",A);

			double *lambda=selected->getDiffusivities(0);
			showOrientationSimilarityMatrix(selected->getAlpha(), selected->getNumAlpha(), selectedDirs, selected->getNumAlpha()*4, selected->getNumAlpha()*w, M, expM);
			cv::imshow("Sim. matrix",M);
			cv::imshow("Exp Sim. matrix",expM);
		}
	//----
	voxSel[0].copyFrom(*selected);
	ostringstream os;
	os<<"selectedVoxel_Recovered_"<<slice<<"_"<<row<<"_"<<col<<".txt";
	newField.saveToTxt(os.str());
}

void MultiTensorFieldExplorer::getSelectedVoxelPositionGT(int &slice, int &row, int &col){
	slice=ui.sliceGT_edit->text().toInt();
	row=ui.rowGT_edit->text().toInt();
	col=ui.columnGT_edit->text().toInt();
}
void MultiTensorFieldExplorer::getSelectedVoxelPositionRecovered(int &slice, int &row, int &col){
	slice=ui.sliceRecovered_edit->text().toInt();
	row=ui.rowRecovered_edit->text().toInt();
	col=ui.columnRecovered_edit->text().toInt();
}

void MultiTensorFieldExplorer::saveRecovered(void){
	string fname=chooseFileToWrite();
	if(fname.empty()){
		return;
	}
	if(ui.volume_radio->isChecked()){
		ui.glViewer->tensorField.saveToTxt(fname);
	}else{
		int slice=ui.sliceRecovered_edit->text().toInt();
		ui.glViewer->tensorField.saveSliceToTxt(slice, fname);
	}
	
} 


void generateErrorSummary(MultiTensorField &GT, MultiTensorField &reconstructed, double *ODFDirections, int numODFDirections, vector<double> &results, bool printResults, bool fullReport);
void MultiTensorFieldExplorer::evaluate(void){
	MultiTensorField &recovered=ui.glViewer->tensorField;
	MultiTensorField &GT=ui.glViewerGT->tensorField;
	common_evaluate(GT, recovered);
	MultiTensor *voxGT=GT.getVoxels();
	MultiTensor *voxRec=recovered.getVoxels();
	int nvox=GT.getNumSlices()*GT.getNumRows()*GT.getNumCols();
	for(int i=0;i<nvox;++i, voxGT++, voxRec++){
		if(voxGT->getNumCompartments()==0){
			voxRec->dellocate();
		}
	}
	string fname=chooseFileToWrite("Save cropped MTF as (optional)...");
	if(fname.empty()){
		return;
	}
	recovered.saveToTxt(fname);
}

void MultiTensorFieldExplorer::compare(void){
	string fname=chooseFileToRead();
	if(fname.empty()){
		return;
	}
	MultiTensorField reference;
	reference.loadFromTxt(fname);
	MultiTensorField &recovered=ui.glViewer->tensorField;
	MultiTensorField &GT=ui.glViewerGT->tensorField;
	vector<double > errorReference;
	evaluateAngularPrecision(GT, reference, errorReference);
	vector<double > errorRecovered;
	evaluateAngularPrecision(GT, recovered, errorRecovered);
	MultiTensor *voxRecovered=recovered.getVoxels();
	MultiTensor *voxReference=reference.getVoxels();

	double *rec_error=recovered.getError();
	int improveCount=0;
	FILE *F=fopen("improvement_stats.txt","w");
	int strictImprovement=0;
	int strictWorsening=0;
	double strictImprovementAve=0;
	double strictWorseningAve=0;
	for(unsigned v=0;v<errorRecovered.size();++v){
		fprintf(F,"%0.15lf\n", errorReference[v]-errorRecovered[v]);
		if(errorRecovered[v]<errorReference[v]){
			rec_error[v]=0;
			++improveCount;
			if(voxRecovered[v].getNumCompartments()==voxReference[v].getNumCompartments()){
				++strictImprovement;
				strictImprovementAve+=errorReference[v]-errorRecovered[v];
			}
		}else{
			rec_error[v]=errorRecovered[v]-errorReference[v];
			if(voxRecovered[v].getNumCompartments()==voxReference[v].getNumCompartments()){
				++strictWorsening;
				strictWorseningAve-=errorReference[v]-errorRecovered[v];
			}
		}
	}
	fclose(F);
	strictImprovementAve/=strictImprovement;
	strictWorseningAve/=strictWorsening;
	cerr<<"Improve count:"<<improveCount<<endl;
	cerr<<"Strict improve count:"<<strictImprovement<<"\t("<<strictImprovementAve<<")"<<endl;
	cerr<<"Strict worsening count:"<<strictWorsening<<"\t("<<strictWorseningAve<<")"<<endl;

}

void MultiTensorFieldExplorer::buildTrainingSet(void){
	MultiTensor *voxGT=ui.glViewerGT->tensorField.getVoxels();
	MultiTensor *voxE=ui.glViewer->tensorField.getVoxels();
	int nr=ui.glViewerGT->tensorField.getNumRows();
	int nc=ui.glViewerGT->tensorField.getNumCols();
	int ns=ui.glViewerGT->tensorField.getNumSlices();
	int nvox=nr*nc*ns;
	FILE *F=fopen("trainingSet.txt","w");
	int trueAlphaCount[5]={0,0,0,0,0};
	int ncompCount[5]={0,0,0,0,0};
	int CM[5][5];
	memset(CM, 0, sizeof(CM));
	for(int v=0;v<nvox;++v){
		MultiTensor &V=voxGT[v];
		MultiTensor &E=voxE[v];
		int nc=V.getNumCompartments();
		CM[nc][E.getNumCompartments()]++;
		if(E.getNumAlpha()==V.getNumCompartments()){
			trueAlphaCount[nc]++;
		}
		ncompCount[nc]++;
		int nAlpha=E.getNumAlpha();
		double *alpha=E.getAlpha();
		double *directions=E.getDirections();
		fprintf(F,"%d\t%d",nc,nAlpha);
		for(int i=0;i<nAlpha;++i){
			fprintf(F,"\t%E", alpha[i]);
		}
		for(int i=1;i<nAlpha;++i){
			double angle=getAbsAngle(&directions[3*(i-1)], &directions[3*i],3);
			fprintf(F,"\t%E", angle);
		}
		fprintf(F,"\n");
	}
	fclose(F);
	cerr<<"True alpha count 1:\t"<<trueAlphaCount[1]<<endl;
	cerr<<"True alpha count 2:\t"<<trueAlphaCount[2]<<endl;
	cerr<<"True alpha count 3:\t"<<trueAlphaCount[3]<<endl;
	cerr<<"True alpha count 4:\t"<<trueAlphaCount[4]<<endl;
	cerr<<"1 Component:\t"<<ncompCount[1]<<endl;
	cerr<<"2 Component:\t"<<ncompCount[2]<<endl;
	cerr<<"3 Component:\t"<<ncompCount[3]<<endl;
	cerr<<"4 Component:\t"<<ncompCount[4]<<endl;
	for(int i=1;i<=4;++i){
		for(int j=1;j<=4;++j){
			cerr<<CM[i][j]<<"\t";
		}
		cerr<<endl;
	}

}





void MultiTensorFieldExplorer::testDenoising(void){
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;
	params.b=1500;

	strcpy(params.basisDirsFname, DEFAULT_DBF_FILE_NAME);
	//----UI params----
	strcpy(params.gradientDirsFname, "gradient_list_048.txt");
	//params.gtFname="Training_3D_SF.txt";
	strcpy(params.gtFname, "Testing_3D_SF.txt");
	strcpy(params.ODFDirFname, "ODF_XYZ.txt");
	params.inputType=MTFIT_GTGenerator;
	params.groundTruthType=MTFGTT_GTGenerator;
	params.solverType=DBFS_NNLS;
	params.iterativeBasisUpdate=true;
	getUIParameters(params);
	params.loadData();
	int nrows=params.GT->getNumRows();
	int ncols=params.GT->getNumCols();
	int nslices=params.GT->getNumSlices();
	int nvoxels=nrows*ncols*nslices;	
	//---prepare mosaics----
	int imagesWidth=sqrt(double (nslices*params.numGradients));
	int imagesHeight=(nslices*params.numGradients)/imagesWidth;
	cv::Mat globalInput;
	cv::Mat globalDenoised;
	cv::Mat globalLatticeH;
	cv::Mat globalLatticeV;
	cv::Mat globalOutliers;
	
	int width=(ncols+1)*imagesWidth+1;
	int height=(nrows+1)*imagesHeight+1;
	globalInput.create(height, width, CV_8UC1);
	globalDenoised.create(height, width, CV_8UC1);
	globalLatticeH.create(height, width, CV_8UC1);
	globalLatticeV.create(height, width, CV_8UC1);
	globalOutliers.create(height, width, CV_8UC1);
	unsigned char *gInputData=(unsigned char *)globalInput.data;
	unsigned char *gDenoisedData=(unsigned char *)globalDenoised.data;
	unsigned char *gLatticeData=(unsigned char *)globalDenoised.data;
	unsigned char *gOutliersData=(unsigned char *)globalOutliers.data;
	for(int i=0;i<height*width;++i){
		gInputData[i]=127;
		gDenoisedData[i]=127;
		gLatticeData[i]=0;
		gOutliersData[i]=0;
	}
	
	//---generate signals---
	double *dwVolume=new double[nvoxels*params.numGradients];
	double *s0Volume=new double[nvoxels];
	double *referenceSignal=new double[params.numGradients];
	MultiTensor *voxGT=params.GT->getVoxels();
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
	}
	
	//----denoise---------
	
	double *img=new double[nrows*ncols];
	double *denoised=new double[nrows*ncols];
	double *Z=new double[2*nrows*ncols];
	double *M=new double[nrows*ncols];
	cv::Mat inputImg;
	cv::Mat denoisedImg;
	cv::Mat latticeHImg;
	cv::Mat latticeVImg;
	cv::Mat outliersImg;
	inputImg.create(nrows,ncols,CV_8UC1);
	denoisedImg.create(nrows,ncols,CV_8UC1);
	latticeHImg.create(nrows,ncols,CV_8UC1);
	latticeVImg.create(nrows,ncols,CV_8UC1);
	outliersImg.create(nrows,ncols,CV_8UC1);
	unsigned char *inputImg_data=inputImg.data;
	unsigned char *denoisedImg_data=denoisedImg.data;
	unsigned char *latticeHImg_data=latticeHImg.data;
	unsigned char *latticeVImg_data=latticeVImg.data;
	unsigned char *outliersImg_data=outliersImg.data;

	int imageIndex=0;
	double lambda=ui.lambda_edit->text().toDouble();
	/*double alpha0 = 0.1*lambda;
	double alpha1 = 0.15*lambda;*/
	double alpha0=ui.alpha0_edit->text().toDouble();
	double alpha1=ui.alpha1_edit->text().toDouble();
	double tau=0.5/sqrt(12.0);
	double sigma=0.5/sqrt(12.0);
	double theta=1;

	double param[5]={alpha0, alpha1, tau, sigma, theta};
	
	//denoiseDWVolume(dwVolume, nrows, ncols, nslices, params.numGradients, lambda, &param);
	for(int k=0;k<params.numGradients;++k){
		int v=0;
		for(int s=0;s<nslices;++s, ++imageIndex){
			int imgPos_row=imageIndex/imagesWidth;
			int imgPos_col=imageIndex%imagesWidth;
			cv::Rect rect(1+imgPos_col*(ncols+1),1+imgPos_row*(nrows+1), ncols,nrows);
			cv::Mat input_roi=globalInput(rect);
			cv::Mat denoised_roi=globalDenoised(rect);
			cv::Mat latticeH_roi=globalLatticeH(rect);
			cv::Mat latticeV_roi=globalLatticeV(rect);
			cv::Mat outliers_roi=globalOutliers(rect);
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++v){
					img[i*ncols+j]=dwVolume[v*params.numGradients+k];
				}
			}
			showImg(img, nrows, ncols, inputImg_data);
			inputImg.copyTo(input_roi);
			//memcpy(denoised, img, sizeof(double)*nrows*ncols);
			memset(denoised, 0, sizeof(double)*nrows*ncols);
			//int retVal=robustDenoising(img, nrows, ncols, lambda, EDT_TUKEY_BIWEIGHT, &param, denoised, Z);
			//int retVal=robustDenoisingOR(img, nrows, ncols, lambda, EDT_TUKEY_BIWEIGHT, param, denoised, Z, M);
			//int retVal=robustDenoisingOR(img, nrows, ncols, lambda, EDT_TGV, param, denoised, Z, M);
			filterTGV_L2(img, nrows, ncols, lambda, param[0], param[1], param[2], param[3], param[4], denoised, NULL);
			showImg(denoised, nrows, ncols, denoisedImg_data);
			denoisedImg.copyTo(denoised_roi);
			showImg(M, nrows, ncols, outliersImg_data);
			outliersImg.copyTo(outliers_roi);
			showHorizontalLattice(Z,nrows,ncols,latticeHImg_data);
			showVerticalLattice(Z+nrows*ncols,nrows,ncols,latticeVImg_data);
			latticeHImg.copyTo(latticeH_roi);
			latticeVImg.copyTo(latticeV_roi);
			cv::imshow("DW Input",globalInput);
			cv::imshow("DW Denoised",globalDenoised);
			cv::imshow("DW H-Lattice",globalLatticeH);
			cv::imshow("DW V-Lattice",globalLatticeV);
			cv::imshow("DW Outliers",globalOutliers);
		}
		
		
		
	}
	cv::imshow("DW Input",globalInput);
	cv::imshow("DW Denoised",globalDenoised);
	cv::imshow("DW H-Lattice",globalLatticeH);
	cv::imshow("DW V-Lattice",globalLatticeV);
	cv::imshow("DW Outliers",globalOutliers);
	//----show images---
	//delete[] Z;
	delete[] denoised;
	delete[] img;
	delete[] dwVolume;
	delete[] s0Volume;
	delete[] referenceSignal;
}

void buildGeodesicDistances(double *dists, int n, int knearest){
	const double INF=1e100;
	if(n<knearest){
		knearest=n;
	}
	for(int i=0;i<n;++i){
		vector<pair<double, int> > v;
		for(int j=0;j<n;++j){
			v.push_back(make_pair(dists[i*n+j], j));
		}
		sort(v.begin(), v.end());
		for(int j=0;j<n;++j){
			dists[i*n+j]=INF;
		}
		for(int j=0;j<knearest;++j){
			int col=v[j].second;
			dists[i*n+col]=v[j].first;
		}
	}
	for(int k=0;k<n;++k){
		for(int i=0;i<n;++i){
			for(int j=0;j<n;++j){
				double opc=dists[i*n+k]+dists[k*n+j];
				if(opc<dists[i*n+j]){
					dists[i*n+j]=opc;
				}
			}
		}
	}
	
}

void buildSparseGraph(double *dists, int n, int knearest, SparseMatrix &M){
	if(n<knearest){
		knearest=n;
	}
	M.create(n,knearest);
	for(int i=0;i<n;++i){
		vector<pair<double, int> > v;
		for(int j=0;j<n;++j){
			v.push_back(make_pair(dists[i*n+j], j));
		}
		sort(v.begin(), v.end());
		Edge **edges=M.edges;
		for(int j=0;j<knearest;++j){
			edges[i][j].destination=v[j].second;
			edges[i][j].w=exp(-v[j].first);
		}
	}
}

void buildNormalizedCutsGraph(SparseMatrix &M){
	int n=M.n;
	double *d=new double[n];
	M.sumRowValues(d);
	for(int i=0;i<n;++i){
		d[i]=1.0/sqrt(d[i]);
	}
	M.multDiagLeftRight(d,d);
	delete[] d;
}

void runECKQMMF(SparseMatrix &kernel, int K, int maxIter, double lambda, double mu, double *&probs){
	double *s=new double[K];
	double *sk1=new double[kernel.n*K];
	double *sk2=new double[K];
	double *pbuff=new double[K*kernel.n];
	for(int iter=0;iter<maxIter;++iter){
		
		computeKernelIntegralFactors(probs, kernel, K, s, sk1, sk2);
		double error=iterateP_kernel(probs, pbuff, kernel, K, lambda, mu, s, sk1, sk2);
		cerr<<"Iteration "<<iter<<": "<<error<<endl;
	}
	delete[] pbuff;
	delete[] s;
	delete[] sk1;
	delete[] sk2;
}


void spectralSegmentation(SparseMatrix &S, double *data, int n, int dim, int K, int maxIter, double lambda, double mu, double *&means, double *&probs){
	int numVertices=S.n;
	double *v_data=new double[numVertices*K];
	double **v=new double*[numVertices];
	double **p=new double*[numVertices];
	double **data_list=new double*[numVertices];
	double **model_list=new double*[K];

	int **neighbors=new int*[numVertices];
	int* numNeighbors=new int[numVertices];
	for(int i=0;i<numVertices;++i){
		v[i]=&v_data[i*K];
		p[i]=&probs[i*K];
		numNeighbors[i]=S.degree[i];
		neighbors[i]=new int[numNeighbors[i]];
		for(int j=0;j<numNeighbors[i];++j){
			neighbors[i][j]=S.edges[i][j].destination;
		}
		data_list[i]=&data[i*dim];
	}

	for(int i=0;i<K;++i){
		model_list[i]=&means[i*dim];
	}
	//------------------------------------------------
	double *N=NULL;
	double *D=NULL;

	for(int iter=1;iter<=maxIter;++iter){
		cerr<<"Iteration "<<iter<<endl;
		computeGaussianNegLogLikelihood(data_list, model_list, numVertices, K, dim, v);
		for(int vertex=0;vertex<numVertices;++vertex){
			iterateNode(vertex, K, v, p, neighbors, numNeighbors, numVertices, lambda, mu, N, D);
		}
		updateMeanVectors(model_list, data_list, p, numVertices, dim, K);
	}
	delete[] v_data;
	delete[] v;
	delete[] p;
	delete[] data_list;
	delete[] model_list;
	if(N!=NULL){
		delete[] N;
	}
	if(D!=NULL){
		delete[] D;
	}
}

void updateTensors(MultiTensorField &M, SparseMatrix &S, int *spatialIndex, int maxTensorsPerVoxel, int *labels, double lambda){
	int numVertices=S.n;
	MultiTensor *voxList=M.getVoxels();
	double ATA[9];
	for(int vertex=0;vertex<numVertices;++vertex){
		int voxelIdx=spatialIndex[vertex]/maxTensorsPerVoxel;
		int tensorIdx=spatialIndex[vertex]%maxTensorsPerVoxel;
		int numNeighbors=S.degree[vertex];
		if(numNeighbors==0){
			continue;
		}
		double *A=new double[(numNeighbors+1)*3];
		voxList[voxelIdx].getPDD(tensorIdx, A);
		int numNeighborsSameLabel=1;
		for(int i=0;i<numNeighbors;++i){
			int vertex_neigh=S.edges[vertex][i].destination;
			if(labels[vertex_neigh]!=labels[vertex]){
				continue;
			}
			int voxelIdx_neigh=spatialIndex[vertex_neigh]/maxTensorsPerVoxel;
			int tensorIdx_neigh=spatialIndex[vertex_neigh]%maxTensorsPerVoxel;
			voxList[voxelIdx_neigh].getPDD(tensorIdx_neigh, &A[3*numNeighborsSameLabel]);
			for(int j=0;j<3;++j){
				A[3*numNeighborsSameLabel+j]*=lambda;
			}
			++numNeighborsSameLabel;
		}
		if(numNeighborsSameLabel==0){
			delete[] A;
			continue;
		}

		for(int i=0;i<3;++i){
			for(int j=0;j<3;++j){
				double sum=0;
				for(int k=0;k<numNeighborsSameLabel;++k){
					sum+=A[3*k+i]*A[3*k+j];
				}
				ATA[i*3+j]=sum;
			}
		}
		double eval[3];
		double u[3];
		symmetricEigenDecomposition(ATA,eval, 3);
		memcpy(u, &ATA[(3-1)*3], 3*sizeof(double));
		normalize<double>(u,3);
		voxList[voxelIdx].setRotationMatrixFromPDD(tensorIdx,u);
		delete[] A;
	}
}

struct WeightedVertex {
	double weight;
	int vertex;
	WeightedVertex(){
	    weight=0;
	    vertex=-1;
	}
	WeightedVertex(double w, int v){
	    weight=w;
	    vertex=v;
	}
};

bool operator<(const WeightedVertex &A, const WeightedVertex &B){
  if(A.weight==B.weight){
    return A.vertex>B.vertex;
  }
  return A.weight>B.weight;
}

void dijkstra(SparseMatrix &S, int source, double *dist){
	int n=S.n;
	bool *known=new bool[n];
	memset(known,0, sizeof(bool)*n);
	for(int i=0;i<n;++i){
		dist[i]=1e100;
	}
	priority_queue<WeightedVertex> Q;
	Q.push(WeightedVertex(0.0,source));
	dist[source]=0.0;
	while(!Q.empty()){
	    WeightedVertex current;
		do{
			current=Q.top();
			Q.pop();
		}while((!Q.empty()) && known[current.vertex]);
		if(known[current.vertex]){
			break;
		}
		known[current.vertex]=true;
		for(int i=0;i<S.degree[current.vertex];++i){
			int neigh=S.edges[current.vertex][i].destination;//the i-th neighbor of current
			double opcDist=dist[current.vertex]+S.edges[current.vertex][i].w;
			if((!known[neigh]) && (opcDist < dist[neigh])){
				Q.push(WeightedVertex(opcDist, neigh));
				dist[neigh]=opcDist;
			}
		}
	}
	for(int i=0;i<n;++i)if(dist[i]>=1e100){
		dist[i]=-1;
	}
	delete[] known;
}

int checkVertex(int v, int label, SparseMatrix &S, int *labels){
	if(labels[v]!=-1){
		return 0;
	}
	int total=1;
	labels[v]=label;
	for(int i=0;i<S.degree[v];++i){
		int neigh=S.edges[v][i].destination;
		total+=checkVertex(neigh, label, S, labels);
	}
	return total;
}

int computeLargerConnectedComponents(SparseMatrix &S, int K, int *labels){
	int n=S.n;
	vector<pair<int, int> > sizes;
	memset(labels,-1, sizeof(int)*n);

	int components=0;
	for(int v=0;v<n;++v)if(labels[v]==-1){
		int checked=checkVertex(v, components+1, S, labels);
		sizes.push_back(make_pair(checked, components+1));
		++components;
	}
	if(components<K){
		return components;
	}
	sort(sizes.rbegin(), sizes.rend());
	int *newLabels=new int[n];
	memset(newLabels, 0, sizeof(int)*n);
	for(int c=0;c<K-1;++c){
		for(int v=0;v<n;++v){
			if(labels[v]==sizes[c].second){
				newLabels[v]=c+1;
			}
		}
	}

	memcpy(labels, newLabels, sizeof(int)*n);
	delete[] newLabels;
	return K;
}


void initializeKernelKMmeansPP(SparseMatrix &S, int K, double *probs){
	srand(98765431);
	int n=S.n;
	//---initialize---
	int *labels=new int[n];
	double **dists=new double*[K];
	double *D=new double[n+1];
	for(int i=0;i<K;++i){
		dists[i]=new double[n];
	}
	memset(probs,0,sizeof(double)*n*K);
	int componentsFound=computeLargerConnectedComponents(S, K, labels);
	/*for(int v=0;v<n;++v){
		probs[v*K+labels[v]]=1.0;
	}
	return;*/
	//----distribute K or K-1 scentroids along the maximal connected component--
	int centroidsToAllocate=K;
	if(componentsFound>1){
		centroidsToAllocate--;
	}
	for(int c=0;c<centroidsToAllocate;++c){
		int sel;
		if(c==0){//find the first centroid
			sel=0;
			while((sel<n) && (labels[sel]!=1)){
				++sel;
			}
		}else{//select next representant with probability proportional to the distance to its smallest centroid
			D[0]=0;
			for(int i=0;i<n;++i){
				double minDist=-1;
				for(int j=0;j<c;++j)if(dists[j][i]>0){
					double d=dists[j][i];
					if((minDist<0) || (d<minDist)){
						minDist=d;
					}
				}
				if(minDist<0){
					minDist=0;
				}
				D[i+1]=D[i]+minDist;//compute the cummilative distribution
			}
			sel=selectFromDistribution(D, n);
		}
		dijkstra(S, sel, dists[c]);
		for(int v=0;v<n;++v)if(dists[c][v]<0){
			dists[c][v]=0;
		}
	}
	//-----------------------------------------------------------------------
	//initialize probability field
	int *mlSegSize=new int[K];
	memset(mlSegSize, 0, sizeof(int)*K);
	for(int v=0;v<n;++v){
		int sel=-1;
		for(int c=0;c<centroidsToAllocate;++c)if(dists[c][v]>0){
			if((sel<0) || (dists[c][v]<dists[sel][v])){
				sel=c;
			}
		}
		if(centroidsToAllocate<K){
			++sel;
		}
		mlSegSize[sel]++;
		for(int i=0;i<K;++i){
			probs[v*K+i]=0.25/(K-1);
		}
		probs[v*K+sel]=0.75;
	}
	for(int i=0;i<K;++i){
		cerr<<i<<": "<<mlSegSize[i]<<endl;
		delete[] dists[i];
	}
	delete[] mlSegSize;
	delete[] dists;
	delete[] D;
	delete[] labels;
}


vector<string> split(const string &s_in, const string &sep){
	string s=s_in;
	vector<string> items;
	while(!s.empty()){
		size_t pos=s.find_first_of(sep);
		if(pos==string::npos){
			items.push_back(s);
			return items;
		}
		items.push_back(s.substr(0,pos));
		s=s.substr(pos+1,s.size()-(pos+1));
	}
	return items;
}

void populateIntegerSet(const string &s, set<int> &S){
	S.clear();
	vector<string> intervals=split(s, ",");
	for(int i=0;i<intervals.size();++i){
		replace<string::iterator, char>(intervals[i].begin(), intervals[i].end(), '-', ' ');
		istringstream is(intervals[i]);
		int inf, sup;
		is>>inf;
		if(!(is>>sup)){
			sup=inf;
		}
		if(sup<inf){
			int tmp=sup;
			sup=inf;
			inf=tmp;
		}
		for(int j=inf;j<=sup;++j){
			S.insert(j);
		}
	}
}

void MultiTensorFieldExplorer::runMultiDimensionalScaling(void){
	int dim=2;
	int K=3;
	double *S=NULL;
	int maxTensorsPerVoxel=-1;
	int *sequentialIndex=NULL;
	int *spatialIndex=NULL;
	double d0=ui.d0_edit->text().toDouble();
	double theta0=ui.theta0_edit->text().toDouble();
	double dTheta0=ui.deltaTheta0_edit->text().toDouble();


	string selectedSlicesStr=ui.showSlicesGT_edit->text().toStdString();
	set<int> selectedSlices;
	populateIntegerSet(selectedSlicesStr, selectedSlices);
	MultiTensorField subTensorField;
	subTensorField.copyFrom(ui.glViewerGT->tensorField,selectedSlices,0);
	ui.glViewerGT->tensorField.copyFrom(subTensorField);


	int numTensors=ui.glViewerGT->tensorField.buildFullSimilarityMatrix(d0, theta0, dTheta0, S, maxTensorsPerVoxel, sequentialIndex, spatialIndex);
	cerr<<"Num. Tensors="<<numTensors<<endl;
	double *x=NULL;
	
	FILE *F=fopen("mdsData.txt", "w");
	for(int i=0;i<numTensors;++i){
		for(int j=0;j<numTensors;++j){
			fprintf(F, "%lf\t", S[i*numTensors+j]);
		}
		fprintf(F, "\n");
	}
	fclose(F);
	multidimensionalScaling(S, numTensors, dim, x);
	

	double *means=NULL;
	double *probs=NULL;
	int maxIter=100;
	fuzzyKMeans(x, numTensors, dim, K, 1, 50, means, probs, true);
	int *labels=new int[numTensors];
	getLabelsFromProbs(probs,numTensors, K, labels);
	delete[] means;
	delete[] probs;
	

	cv::Mat img;

	showClassifiedPointCloud(x, numTensors, dim, img, 256,256, labels);
	cv::imshow("Embedded", img);
	for(int i=0;i<numTensors;++i){
		labels[i]*=2;
	}
	showClassifiedPointCloud(x, numTensors, dim, img, 256,256, NULL);
	cv::imshow("Embedded", img);

	MultiTensor *voxList=ui.glViewerGT->tensorField.getVoxels();
	for(int i=0;i<numTensors;++i){
		int voxelIdx=spatialIndex[i]/maxTensorsPerVoxel;
		int tensorIdx=spatialIndex[i]%maxTensorsPerVoxel;
		int nc=voxList[voxelIdx].getNumCompartments();
		voxList[voxelIdx].setCompartmentSegmentation(tensorIdx,labels[i]);
	}

	delete[] S;
	delete[] sequentialIndex;
	delete[] spatialIndex;
	delete[] x;
	delete[] labels;
}

void MultiTensorFieldExplorer::runSpectralClustering(void){
	int dim=2;
	int K=3;
	/*double angleWeight=ui.alpha0_edit->text().toDouble();
	double maxNeighAngle=ui.alpha1_edit->text().toDouble();*/
	SparseMatrix S;
	int maxTensorsPerVoxel=-1;
	int *sequentialIndex=NULL;
	int *spatialIndex=NULL;
	//double sigma=1;

	double d0=ui.d0_edit->text().toDouble();
	double theta0=ui.theta0_edit->text().toDouble();
	double dTheta0=ui.deltaTheta0_edit->text().toDouble();
	double offset=0;
	//ui.glViewerGT->tensorField.buildNCutSparseMatrix(d0, theta0, dTheta0, offset, S, maxTensorsPerVoxel, sequentialIndex, spatialIndex);
	ui.glViewerGT->tensorField.buildNCutSparseMatrixBestAssignment(d0, theta0, dTheta0, offset, S, maxTensorsPerVoxel, sequentialIndex, spatialIndex);
	//---draw---
	cv::Mat SM;
	SM.create(256, 256, CV_8UC1);
	S.draw((unsigned char *)SM.data, 256, 256);
	cv::imshow("S",SM);
	//----square matrix----
	/*SparseMatrix SS;
	SS.create(S.n, 10*S.k);

	int nrows=ui.glViewerGT->tensorField.getNumRows();
	int ncols=ui.glViewerGT->tensorField.getNumCols();
	int sliceSize=nrows*ncols;

	for(int i=0;i<S.n;++i){
		int si=spatialIndex[i];
		si/=maxTensorsPerVoxel;
		int slicePos=si/sliceSize;
		si=si%sliceSize;
		int rowPos=si/ncols;
		int colPos=si%ncols;
		
		//---compute diagonal element SS[i, i]---
			double sum=S.diagonal[i]*S.diagonal[i];
			for(int k=0;k<S.k;++k){
				int kk=S.edges[i][k].destination;
				if(kk<0){
					continue;
				}
				double w=S.edges[i][k].w;
				double ww=S.retrieve(kk, i);
				sum+=w*ww;
			}
			SS.diagonal[i]=sum;
		//---compute off-diagonal elements---
		//for(int j=0;j<S.k;++j){
		for(int j=0;j<S.n;++j){
			int sj=spatialIndex[j];
			sj/=maxTensorsPerVoxel;
			int slicePos_neigh=sj/sliceSize;
			sj=sj%sliceSize;
			int rowPos_neigh=sj/ncols;
			int colPos_neigh=sj%ncols;
			int manhattanDist=ABS(rowPos-rowPos_neigh)+ABS(colPos-colPos_neigh)+ABS(slicePos-slicePos_neigh);
			if((manhattanDist>3) || (manhattanDist==0)){
				continue;
			}
			//int jj=S.edges[i][j].destination;
			int jj=j;

			//---compute element SS[i, jj]---
			double sum=S.diagonal[i]*S.retrieve(i, jj);
			for(int k=0;k<S.k;++k){
				int kk=S.edges[i][k].destination;
				if(kk<0){
					continue;
				}
				double w=S.edges[i][k].w;
				double ww=S.retrieve(kk, jj);
				sum+=w*ww;
			}
			//---assign---
			SS.addEdge(i,jj,sum);
		}
	}
	S.dellocate();
	S.create(SS.n, SS.k);
	S.copyFrom(SS);

	cv::Mat SSM;
	SSM.create(256, 256, CV_8UC1);
	SS.draw((unsigned char *)SSM.data, 256, 256);
	cv::imshow("SS",SSM);
*/
	//----normalize----
	/*double *sumCol=new double[S.n];
	double *sumRow=new double[S.n];
	memset(sumCol, 0, sizeof(double)*S.n);
	memset(sumRow, 0, sizeof(double)*S.n);
	for(int i=0;i<S.n;++i){
		for(int j=0;j<S.k;++j){
			if(S.edges[i][j].destination>=0){
				int jj=S.edges[i][j].destination;
				double w=S.edges[i][j].w;
				sumCol[i]+=w*w;
				sumRow[jj]+=w*w;
			}
		}
	}

	for(int i=0;i<S.n;++i){
		for(int j=0;j<S.k;++j){
			if(S.edges[i][j].destination>=0){
				int jj=S.edges[i][j].destination;
				double w=S.edges[i][j].w;
				S.edges[i][j].w=0.5*(w*w/sumRow[i] + w*w/sumCol[jj]);
			}
		}
	}
	delete[] sumCol;
	delete[] sumRow;*/
	//-----------------
	double assym=S.computeAsymetry();
	cerr<<"Asymetry:"<<assym<<endl;
	int numTensors=S.n;
	
	
	double *X=new double[numTensors*dim];
	if(ui.loadEmbedded_check->isChecked()){
		FILE *F=fopen("embedded.txt","r");
		for(int i=0;i<numTensors;++i){
			for(int j=0;j<dim;++j){
				fscanf(F,"%lf \n", &X[i*dim+j]);
			}
		}
		fclose(F);
	}else{
		double *eval=new double[dim+1];
		double *evec=new double[numTensors*(dim+1)];	
		arpack_symetric_evd(S, numTensors, dim+1, evec, eval);
		for(int i=0;i<(dim+1);++i){
			S.testEigenPair(&evec[i*numTensors],eval[i]);
		}
		for(int i=0;i<numTensors;++i){
			for(int j=0;j<dim;++j){
				X[i*dim+j]=evec[(j)*numTensors+i];
			}
		}
		delete[] eval;
		delete[] evec;
	}

	if(ui.saveEmbedded_check->isChecked() && !ui.loadEmbedded_check->isChecked()){
		FILE *F=fopen("embedded.txt","w");
		for(int i=0;i<numTensors;++i){
			for(int j=0;j<dim;++j){
				fprintf(F,"%E\t", X[i*dim+j]);
			}
			fprintf(F,"\n");
		}
		fclose(F);
	}

	int *labels=new int[numTensors];
	if(ui.loadClustering_check->isChecked()){
		FILE *F=fopen("labels.txt","r");
		for(int i=0;i<numTensors;++i){
			fscanf(F, "%d", &labels[i]);
		}
		fclose(F);
	}else{
		double *means=NULL;
		double *probs=NULL;
		//---normalize---
		for(int i=0;i<numTensors;++i){
			double nrm=0;
			for(int j=0;j<dim;++j){
				nrm+=SQR(X[i*dim+j]);
			}
			nrm=sqrt(nrm);
			if(nrm>1e-3){
				for(int j=0;j<dim;++j){
					//X[i*dim+j]/=nrm;
				}
			}else{
				/*for(int j=0;j<dim;++j){
					X[i*dim+j]=0;
				}*/
			}
		}
		//---------------
		double lambda=ui.alpha0_edit->text().toDouble();
		double mu=ui.alpha1_edit->text().toDouble();
		int maxIter=50;
		fuzzyKMeans(X, numTensors, dim, K, 1, 50, means, probs, true);
		//---test EC-KQMMF---
		/*probs=new double[numTensors*K];
		initializeKernelKMmeansPP(S, K, probs);
		runECKQMMF(S, K, maxIter, lambda, mu, probs);*/
		//-------------------
		//spectralSegmentation(S, X, numTensors, dim, K, 50, lambda, mu, means, probs);
		/*for(int ii=0;ii<10;++ii){
			updateTensors(ui.glViewerGT->tensorField,S,spatialIndex,maxTensorsPerVoxel,labels,0.5);
		}*/
		getLabelsFromProbs(probs,numTensors, K, labels);
		delete[] means;
		delete[] probs;

	}
	
	cv::Mat img;
	for(int i=0;i<numTensors;++i){
		labels[i]*=2;
	}
	showClassifiedPointCloud(X, numTensors, dim, img, 256,256, labels);
	cv::imshow("Embedded", img);

	MultiTensor *voxList=ui.glViewerGT->tensorField.getVoxels();
	for(int i=0;i<numTensors;++i){
		int voxelIdx=spatialIndex[i]/maxTensorsPerVoxel;
		int tensorIdx=spatialIndex[i]%maxTensorsPerVoxel;
		int nc=voxList[voxelIdx].getNumCompartments();
		voxList[voxelIdx].setCompartmentSegmentation(tensorIdx,labels[i]);
	}
	delete[] sequentialIndex;
	delete[] spatialIndex;
	delete[] labels;
}



void MultiTensorFieldExplorer::testMask(void){
	time_t seed=1331241348;
	srand(seed);
	MultiTensorFieldExperimentParameters params;

	//----Alonso----
	strcpy(params.rawDataBaseDir, "DWCerebro\\alonso_aff\\");
	strcpy(params.dwListFname,"names.txt");
	strcpy(params.gradientDirsFname, "gradients.txt");
	params.inputType=MTFIT_DWMRI_FILES;
	params.b=1000*1e6;
	params.groundTruthType=MTFGTT_None;
	//----Concha, Philips----
	/*params.rawDataBaseDir=(string("DWCerebro")+PATH_SEPARATOR)+(string("concha_philips_01")+PATH_SEPARATOR);
	params.dwFname="Philips_ec.nii";
	params.gradientDirsFname="Philips.bvec";
	params.inputType=MTFIT_NIFTI;
	params.b=2000;
	params.groundTruthType=MTFGTT_None;*/
	//-----------------------
	getUIParameters(params);
	params.loadData();

	int nrows=params.reconstructed->getNumRows();
	int ncols=params.reconstructed->getNumCols();
	int nslices=params.reconstructed->getNumSlices();
	int nvoxels=nrows*ncols*nslices;
	int maxSparsity=0;
	//---------Generalized DTI-----
	GDTI H(2, params.b, params.gradients, params.numGradients);
	double longDiffusion;
	double transDiffusion;
	double *fittedTensors=new double[nvoxels*H.getNumCoefficients()];
	double *GDTI_eVal=new double[3*nvoxels];
	double *GDTI_pdd=new double[3*nvoxels];
	H.setUseNeighbors(false);
	unsigned char *mask_WM=new unsigned char[nslices*nrows*ncols];
	double *fa=new double[nslices*nrows*ncols];
	double *md=new double[nslices*nrows*ncols];

	
	H.createMask(params.s0Volume, params.dwVolume,nslices, nrows, ncols, fittedTensors,GDTI_eVal, GDTI_pdd,mask_WM,0,1, -1);
	double avProfile[3];
	H.computeAverageProfile(GDTI_eVal, nvoxels, mask_WM, 3, 1, avProfile);
	longDiffusion=avProfile[2];
	transDiffusion=0.5*(avProfile[0]+avProfile[1]);



	//---compute MD and FA---
	double thrFA=0.2;
	double thrS0=300;
	double thrMD=30;
	int pos=0;
	double minMD=1e10;
	double maxMD=-1e10;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				if(params.s0Volume[pos]>1){
					double *lambda=&GDTI_eVal[3*pos];
					md[pos]=(lambda[0]+lambda[1]+lambda[2])/3.0;
					fa[pos]=computeFractionalAnisotropy(lambda, md[pos]);
					minMD=MIN(minMD, md[pos]);
					maxMD=MAX(maxMD, md[pos]);
				}
				
			}
		}
	}
	//----------------


	int key=0;
	double *currentSlice=new double[nrows*ncols];
	double *currentCoronalSlice=new double[nslices*ncols];
	//double *currentS0Slice=NULL;
	cv::Mat slice_img;
	cv::Mat slice_img_coronal;
	slice_img.create(nrows, ncols, CV_8UC1);
	slice_img_coronal.create(nrows, ncols, CV_8UC1);
	unsigned char *slice_img_data=(unsigned char *)slice_img.data;
	unsigned char *slice_img_data_coronal=(unsigned char *)slice_img_coronal.data;
	int slice=nslices/2;
	int slice_coronal=nrows/2;
	int slice_type=0;

	//currentS0Slice=&params.s0Volume[slice*nrows*ncols];

	bool recompute=true;
	bool changeSlice=true;
	int nzero_vox=0;
	while(key!=13){
		key=cv::waitKey(1000);
		switch(key){
			case 'c':case 'C':
				nzero_vox=0;
				for(int v=0;v<nslices*nrows*ncols;++v){
					if((fa[v]>thrFA) && (params.s0Volume[v]>thrS0)){
						++nzero_vox;
					}
				}
				cerr<<"Non-zero voxels: "<<nzero_vox<<endl;
				cerr<<"Current axial slice: "<<slice<<endl;
				cerr<<"Current coronal slice: "<<slice_coronal<<endl;
			break;
			case 't':case 'T':
				slice_type=1-slice_type;
			break;
			case 'M':
				recompute=true;
				thrMD+=1;
			break;
			case 'm':
				recompute=true;
				thrMD-=1;
			break;
			case 'F':
				recompute=true;
				thrFA+=0.01;
			break;
			case 'f':
				recompute=true;
				thrFA-=0.01;
			break;
			case 'S':
				recompute=true;
				thrS0+=10;
			break;
			case 's':
				recompute=true;
				thrS0-=10;
			break;
			case '+':
				if(slice_type==0){
					++slice;
					if(slice>=nslices){
						slice=nslices-1;
					}
				}else{
					++slice_coronal;
					if(slice_coronal>=nrows){
						slice_coronal=nrows-1;
					}
				}
				
				changeSlice=true;
			break;
			case '-':
				if(slice_type==0){
					--slice;
					if(slice<0){
						slice=0;
					}
				}else{
					--slice_coronal;
					if(slice_coronal<0){
						slice_coronal=0;
					}
				}
				changeSlice=true;
			break;
		}
		if(recompute || changeSlice){
			//---update axial slice---
			//currentS0Slice=&params.s0Volume[slice*nrows*ncols];
			//currentS0Slice=&fa[slice*nrows*ncols];
			int pp=0;
			memset(currentSlice, 0, sizeof(double)*nrows*ncols);
			for(int i=0;i<nrows;++i){
				for(int j=0;j<ncols;++j, ++pp)if(params.s0Volume[slice*nrows*ncols+pp]>thrS0){
					double refMD=100*(md[slice*nrows*ncols+pp]-minMD)/(maxMD-minMD);

					if((fa[slice*nrows*ncols+pp]>thrFA) && (refMD<thrMD)){
						currentSlice[pp]=fa[slice*nrows*ncols+pp];
						//currentSlice[pp]=1;
					}
				}
			}
			showImg(currentSlice, nrows, ncols, slice_img_data);
			cv::imshow("current",slice_img);
			//-------------------------
			memset(currentCoronalSlice, 0, sizeof(double)*nslices*ncols);
			pp=0;
			for(int s=0;s<nslices;++s){
				for(int j=0;j<ncols;++j,++pp)if(params.s0Volume[s*nrows*ncols+slice_coronal*ncols+j]>thrS0){
					double refMD=100*(md[s*nrows*ncols+slice_coronal*ncols+j]-minMD)/(maxMD-minMD);

					if((fa[s*nrows*ncols+slice_coronal*ncols+j]>thrFA) && (refMD<thrMD)){
						currentCoronalSlice[pp]=fa[s*nrows*ncols+slice_coronal*ncols+j];
						//currentSlice[pp]=1;
					}
				}
			}
			showImg(currentCoronalSlice, nslices, ncols, slice_img_data_coronal);
			cv::imshow("current_coronal",slice_img_coronal);
			//-------------------------
			recompute=false;
			changeSlice=false;
			cerr<<"thrFA="<<thrFA<<", thrS0="<<thrS0<<", thrMD="<<thrMD<<endl;
		}
	}
	//----save mask---
	string fname="D:\\cimat\\experiments\\PGS_evaluation\\DWCerebro\\maskFA.raw";
	FILE *F=fopen(fname.c_str(), "wb");
	for(int s=0;s<nslices;++s){
		int pp=0;
		for(int i=0;i<nrows;++i){
			for(int j=0;j<ncols;++j, ++pp)if(params.s0Volume[s*nrows*ncols+pp]>thrS0){
				double refMD=100*(md[s*nrows*ncols+pp]-minMD)/(maxMD-minMD);
				unsigned char val;
				if((fa[s*nrows*ncols+pp]>thrFA) && (refMD<thrMD)){
					val=1;
				}else{
					val=0;
				}
				fwrite(&val, sizeof(unsigned char),1, F);
			}else{
				unsigned char val=0;
				fwrite(&val, sizeof(unsigned char),1, F);
			}
		}
	}
	fclose(F);
	delete[] currentSlice;
	delete[] currentCoronalSlice;
	delete[] mask_WM;
	delete[] fittedTensors;
	delete[] GDTI_eVal;
	delete[] GDTI_pdd;
	delete[] fa;
	delete[] md;
}

void MultiTensorFieldExplorer::LCIToMask(void){
	string LCIFname=chooseFileToRead("Select LCI file...");
	if(LCIFname.empty()){
		return;
	}

	//---load LCI image---
	double *img=NULL;
	int ndim=0;
	int *dims=NULL;
	int *vox=NULL;
	int *layout=NULL;
	int *layout_orientations=NULL;
	double *transform=NULL;
	int retVal=read_mrtrix_image(LCIFname.c_str(), img, ndim, dims, vox, layout, layout_orientations, transform);
	int nvox=dims[0]*dims[1]*dims[2];

	double *mask=new double[nvox];
	for(int i=0;i<nvox;++i){
		if(!isNumber(img[i])){
			mask[i]=0;//background
		}else if(img[i]<25){
			mask[i]=1;//white matter
		}else{
			mask[i]=2;//tumor
		}
	}
	//------------------
	string mask_nii=changeExtension(LCIFname, "_mask.nii");
	string mask_mif=changeExtension(LCIFname, "_mask.mif");
	write_mrtrix_image(mask_mif.c_str(), mask,ndim, dims, vox, layout, layout_orientations, transform);
	save3DNifti(mask_nii.c_str(), mask, dims[2], dims[1], dims[0]);

	delete[] dims;
	delete[] vox;
	delete[] layout;
	delete[] layout_orientations;
	delete[] transform;
	delete[] img;
	delete[] mask;
}

void MultiTensorFieldExplorer::mtfToLCI(void){
	string MTFFileName=chooseFileToRead("Select Multi-Tensor Field");
	if(MTFFileName.empty()){
		return;
	}
	string LCIFileName=chooseFileToWrite("Save LCI as [prefix only]...");
	if(LCIFileName.empty()){
		return;
	}
	MultiTensorField mtf;
	mtf.loadFromTxt(MTFFileName);
	double voidCost=180;
	int nvox=mtf.getNumSlices()*mtf.getNumRows()*mtf.getNumCols();
	double *LCI=new double[nvox];
	mtf.computeLocalCoherenceIndex(voidCost, LCI);

	write_mrtrix_image((LCIFileName+".mif").c_str(), LCI, mtf.getNumSlices(), mtf.getNumRows(), mtf.getNumCols());
	save3DNifti((LCIFileName+".nii").c_str(), LCI, mtf.getNumSlices(), mtf.getNumRows(), mtf.getNumCols()); 

	delete[] LCI;
}

void MultiTensorFieldExplorer::gtFromPaths(void){
	//---test---
	/*double **tracks=NULL;
	int *trackSizes=NULL;
	int numTracks=0;
	map<string, string> properties;
	int retVal=read_mrtrix_tracks("whole_challenge.tck", tracks, trackSizes, numTracks, properties);
	for(int i=0;i<numTracks;++i){
		delete[] tracks[i];
	}
	delete[] tracks;
	delete[] trackSizes;
	return;*/
	//----------
	const int nslices=50;//FIXME
	const int nrows=50;
	const int ncols=50;
	const int maxCompartments=5;
	string namesFname=chooseFileToRead();
	if(namesFname.empty()){
		return;
	}
	ifstream F(namesFname.c_str());
	if(!F.good()){
		return;
	}
	string nextLine;
	bool first=true;
	double *radii=NULL;
	double **paths=NULL;
	int *pathLengths=NULL;
	int nfibers=0;
	int cnt=0;
	do{
		getline(F, nextLine);
		if(!FileExists(nextLine)){
			cerr<<"Error: file not found '"<<nextLine<<"'"<<endl;
			return;
		}
		if(first){//radii file
			loadArrayFromFile<double>(nextLine.c_str(), radii, nfibers);
			first=false;
			paths=new double*[nfibers];
			pathLengths=new int[nfibers];
			memset(paths, 0, sizeof(double*)*nfibers);
			memset(pathLengths, 0, sizeof(int)*nfibers);
		}else{
			loadArrayFromFile(nextLine.c_str(), paths[cnt], pathLengths[cnt]);
			if((pathLengths[cnt]%3)!=0){
				cerr<<"Warning: probably wrong path file '"<<nextLine<<"'. Expected multiple of 3 values [file truncated]."<<endl;
			}
			pathLengths[cnt]/=3;
			++cnt;
		}
	}while((!F.eof()) && (cnt<nfibers));
	F.close();
	nfibers=cnt;
	if(cnt!=nfibers){
		cerr<<"Error: unexpected number of fibers: "<<cnt<<" given, "<<nfibers<<" expected."<<endl;
	}else{
		DWMRISimulator simulator(nslices, nrows, ncols, maxCompartments);
		for(int i=0;i<nfibers;++i){
			cerr<<"Adding fiber #"<<i+1<<"...";
			simulator.addPath(paths[i], pathLengths[i], radii[i]);
			cerr<<"done."<<endl;
		}
		MultiTensorField mtf;
		simulator.buildMultiTensorField(mtf);
		string mtfFname=chooseFileToWrite();
		if(!(mtfFname.empty())){
			mtf.saveToTxt(mtfFname);
		}
	}
	for(int i=0;i<nfibers;++i){
		delete[] paths[i];
	}
	delete[] paths;
	delete[] pathLengths;
	delete[] radii;
}


void MultiTensorFieldExplorer::createRandomTensor(void){
	double bestPdd[3];
	double bestAngle=1e10;
	double target=ui.target_edit->text().toDouble();
	//----test----
	int worstDir=-1;
	double worstAngle=0;
	for(int i=0;i<numDBFDirections;++i){
		double minAngle=1e10;
		for(int j=0;j<numDBFDirections;++j)if(i!=j){
			double opc=getAbsAngleDegrees(&DBFDirections[3*i], &DBFDirections[3*j], 3);
			minAngle=MIN(minAngle, opc);
		}
		cerr<<i<<":\t"<<minAngle<<endl;
		if(worstAngle<minAngle){
			worstAngle=minAngle;
			worstDir=i;
		}
	}
	cerr<<"Worst angle:\t"<<worstAngle<<endl;
	//------------
	for(int i=0;i<1000;++i){
		double pdd[3];
		double nrm=0;
		for(int j=0;j<3;++j){
			pdd[j]=uniform(0,1);
			nrm+=SQR(pdd[j]);
		}
		nrm=sqrt(nrm);
		for(int j=0;j<3;++j){
			pdd[j]/=nrm;
		}
		double minAngle=1e10;
		for(int j=0;j<numDBFDirections;++j){
			double opc=getAbsAngleDegrees(pdd, &DBFDirections[3*j], 3);
			minAngle=MIN(minAngle, opc);
		}
		
		if(fabs(target-minAngle)<bestAngle){
			bestAngle=fabs(target-minAngle);
			memcpy(bestPdd, pdd, sizeof(pdd));
		}
	}
	//double opc=getAbsAngleDegrees(bestPdd, &DBFDirections[3*129], 3);
	cerr<<"Best angle:"<<bestAngle<<"\t["<<bestPdd[0]<<", "<<bestPdd[1]<<", "<<bestPdd[2]<<endl;
	double longDiffusion=1.0*0.0013519389594264955;
	double transDiffusion=1.0*0.00025065326974918722;
	ui.glViewerGT->tensorField.allocate(1,1,1);
	MultiTensor *voxel=ui.glViewerGT->tensorField.getVoxels();
	voxel->allocate(1);
	voxel->setVolumeFraction(0,1);
	voxel->setDiffusivities(0,transDiffusion, transDiffusion, longDiffusion);
	voxel->setRotationMatrixFromPDD(0,bestPdd);
	double *alpha=new double[numDBFDirections+1];
	int *groups=new int[numDBFDirections+1];
	double *extendedDirs=new double[3*(numDBFDirections+1)];
	memcpy(extendedDirs, DBFDirections, sizeof(double)*3*numDBFDirections);
	memcpy(&extendedDirs[3*numDBFDirections], bestPdd, sizeof(double)*3);
	for(int i=0;i<numDBFDirections;++i){
		alpha[i]=1;
		double ang=getAbsAngleDegrees(&DBFDirections[3*i], bestPdd, 3);
		if(ang<15){
			groups[i]=1;
		}else{
			groups[i]=7;
		}
		
	}
	alpha[worstDir]=1;
	alpha[numDBFDirections]=1;
	groups[numDBFDirections]=4;
	voxel->setAlpha(alpha, numDBFDirections+1);
	voxel->setGroups(groups, numDBFDirections+1);
	voxel->setDirections(extendedDirs,numDBFDirections+1);
	delete[] alpha;
	delete[] extendedDirs;
	delete[] groups;
}

void MultiTensorFieldExplorer::alphaBehaviorStudy(void){
	double longDiffusion=1.0*0.0013519389594264955;
	double transDiffusion=1.0*0.00025065326974918722;
	int nrows=30;
	int ncols=30;
	ui.glViewerGT->tensorField.allocate(1,nrows,ncols);
	MultiTensor *voxels=ui.glViewerGT->tensorField.getVoxels();
	int v=0;
	int nc=1;
	for(int i=0;i<nrows;++i){
		for(int j=0;j<ncols;++j,++v){
			MultiTensor *voxel=&voxels[v];
			voxel->allocate(nc);
			for(int c=0;c<nc;++c){
				voxel->setVolumeFraction(c,1.0/nc);
				voxel->setDiffusivities(c,transDiffusion, transDiffusion, longDiffusion);
				double pdd[3];
				double nrm=0;
				for(int j=0;j<3;++j){
					pdd[j]=uniform(0,1);
					nrm+=SQR(pdd[j]);
				}
				nrm=sqrt(nrm);
				for(int j=0;j<3;++j){
					pdd[j]/=nrm;
				}
				voxel->setRotationMatrixFromPDD(c,pdd);
			}
		}
	}
	ui.glViewerGT->tensorField.saveToTxt("synthetic_monotensor.txt");
}