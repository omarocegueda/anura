#ifdef USE_QT
#include <QtGui>
#include <QGLWidget>
#include <GL/glu.h>
#endif
#include "MultiTensorField.h"
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "macros.h"
#include "geometryutils.h"
#include "hungarian.h"
#include "statisticsutils.h"
#include "dtisynthetic.h"
#include "utilities.h"
#include "linearalgebra.h"
#include <vector>
#include "hungarian.h"
#include "dtiutils.h"
#include "bits.h"
using namespace std;
#define EPS_MTF 1e-6
#define ADD_DUPLICATES
MultiTensorField::MultiTensorField(){
	nrows=0;
	ncols=0;
	nslices=0;
	voxels=NULL;
	error=NULL;
	dualLatticeRow=NULL;
	dualLatticeCol=NULL;
	dualLatticeSlice=NULL;
	DBFDirections=NULL;
	showGroupColors=false;
	samplingPoints=NULL;
	numSamplingPoints=0;
	
}

MultiTensorField::MultiTensorField(int ns, int nr, int nc){
	allocate(ns, nr, nc);
	DBFDirections=NULL;
}

MultiTensorField::~MultiTensorField(){
	dellocate();
}

void MultiTensorField::dellocate(void){
	DELETE_ARRAY(voxels);
	DELETE_ARRAY(error);
	DELETE_ARRAY(dualLatticeCol);
	DELETE_ARRAY(dualLatticeRow);
	DELETE_ARRAY(dualLatticeSlice);
	DELETE_ARRAY(samplingPoints);
}

void MultiTensorField::allocate(int ns, int nr, int nc){
	nslices=ns;
	nrows=nr;
	ncols=nc;
	voxels=new MultiTensor[ns*nr*nc];
	error=new double[ns*nr*nc];
	dualLatticeCol=new double[nslices*nrows*ncols];
	dualLatticeRow=new double[nslices*nrows*ncols];
	dualLatticeSlice=new double[nslices*nrows*ncols];
	memset(dualLatticeCol, 0, sizeof(double)*nslices*nrows*ncols);
	memset(dualLatticeRow, 0, sizeof(double)*nslices*nrows*ncols);
	memset(dualLatticeSlice, 0, sizeof(double)*nslices*nrows*ncols);
	memset(error, 0, sizeof(double)*ns*nr*nc);
	samplingPoints=NULL;
	numSamplingPoints=0;
}

void MultiTensorField::loadFromTxt(const string &fname){
	FILE *F=fopen(fname.c_str(), "r");
	int dim;
	fscanf(F, "%d", &dim);
	if((dim<=1) || (3<dim)){
		cerr<<"MultiTensorField not supported for dim="<<dim<<"."<<endl;
		fclose(F);
		return;
	}else if(dim==2){
		nslices=1;
		fscanf(F, "%d%d", &nrows, &ncols);
	}else{
		fscanf(F, "%d%d%d", &nslices, &nrows, &ncols);
	}
	allocate(nslices,nrows,ncols);
	int nzero=0;
	for(int s=0;s<nslices;++s){
		//for(int r=0;r<nrows;++r){
		for(int r=nrows-1;r>=0;--r){
			for(int c=0;c<ncols;++c){
				int pos=s*(nrows*ncols)+r*ncols+c;
				voxels[pos].loadFromTxt(F);
				if(voxels[pos].getNumCompartments()>0){
					++nzero;
				}
			}
		}
	}
	fclose(F);
}

void MultiTensorField::saveToTxt(const string &fname){
	FILE *F=fopen(fname.c_str(), "w");
	int dim=3;
	fprintf(F, "%d\n", dim);
	if((dim<=1) || (3<dim)){
		cerr<<"MultiTensorField not supported for dim="<<dim<<"."<<endl;
		fclose(F);
		return;
	}else if(dim==2){
		nslices=1;
		fprintf(F, "%d\t%d\n", nrows, ncols);
	}else{
		fprintf(F, "%d\t%d\t%d\n", nslices, nrows, ncols);
	}
	int nzero=0;
	for(int s=0;s<nslices;++s){
		//for(int r=0;r<nrows;++r){
		for(int r=nrows-1;r>=0;--r){
			for(int c=0;c<ncols;++c){
				int pos=s*(nrows*ncols)+r*ncols+c;
				if(pos==16374){
					pos=pos;
				}
				voxels[pos].saveToTxt(F);
				if(voxels[pos].getNumCompartments()>0){
					++nzero;
				}
			}
		}
	}
	fclose(F);
}

void MultiTensorField::loadFromBinary(const std::string &fname){
	FILE *F=fopen(fname.c_str(), "rb");
	int dim;
	fread(&dim, sizeof(int), 1, F);
	if((dim<=1) || (3<dim)){
		cerr<<"MultiTensorField not supported for dim="<<dim<<"."<<endl;
		fclose(F);
		return;
	}else if(dim==2){
		nslices=1;
		fread(&nrows, sizeof(int),1, F);
		fread(&ncols, sizeof(int),1, F);
	}else{
		fread(&nslices, sizeof(int),1, F);
		fread(&nrows, sizeof(int),1, F);
		fread(&ncols, sizeof(int),1, F);
	}
	allocate(nslices,nrows,ncols);
	for(int s=0;s<nslices;++s){
		//for(int r=0;r<nrows;++r){
		for(int r=nrows-1;r>=0;--r){
			for(int c=0;c<ncols;++c){
				int pos=s*(nrows*ncols)+r*ncols+c;
				voxels[pos].loadFromBinary(F);
			}
		}
	}
	fclose(F);
}

void MultiTensorField::saveToBinary(const std::string &fname){
	FILE *F=fopen(fname.c_str(), "wb");
	int dim=3;
	fwrite(&dim, sizeof(int),1, F);
	if((dim<=1) || (3<dim)){
		cerr<<"MultiTensorField not supported for dim="<<dim<<"."<<endl;
		fclose(F);
		return;
	}else if(dim==2){
		nslices=1;
		fwrite(&nrows, sizeof(int),1, F);
		fwrite(&ncols, sizeof(int),1, F);
	}else{
		fwrite(&nslices, sizeof(int),1, F);
		fwrite(&nrows, sizeof(int),1, F);
		fwrite(&ncols, sizeof(int),1, F);
	}
	for(int s=0;s<nslices;++s){
		//for(int r=0;r<nrows;++r){
		for(int r=nrows-1;r>=0;--r){
			for(int c=0;c<ncols;++c){
				int pos=s*(nrows*ncols)+r*ncols+c;
				voxels[pos].saveToBinary(F);
			}
		}
	}
	fclose(F);
}


void MultiTensorField::saveSliceToTxt(int slice, const string &fname){
	FILE *F=fopen(fname.c_str(), "w");
	int dim=3;
	fprintf(F, "%d\n", dim);
	if((dim<=1) || (3<dim)){
		cerr<<"MultiTensorField not supported for dim="<<dim<<"."<<endl;
		fclose(F);
		return;
	}else if(dim==2){
		nslices=1;
		fprintf(F, "%d\t%d\n", nrows, ncols);
	}else{
		fprintf(F, "%d\t%d\t%d\n", nslices, nrows, ncols);
	}
	if((0<=slice) && (slice<nslices)){
		for(int r=nrows-1;r>=0;--r){
			for(int c=0;c<ncols;++c){
				int pos=slice*(nrows*ncols)+r*ncols+c;
				voxels[pos].saveToTxt(F);
			}
		}
	}
		
	fclose(F);
}

void MultiTensorField::buildFromCompartments(int nslices, int nrows, int ncols, int maxCompartments, int *numCompartments, double *compartmentSizes, double *pddField){
	int nvox=nslices*nrows*ncols;
	allocate(nslices, nrows, ncols);
	MultiTensor *voxE=voxels;
	double *sc=new double[maxCompartments];
	double lambda[3]={0.000393979, 0.000393979,0.001574300 };
	int pos=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos, ++voxE){
				int nc=numCompartments[pos];
				voxE->allocate(nc);
				memcpy(sc, &compartmentSizes[pos*maxCompartments], sizeof(double)*nc);
				double *pdd=&pddField[pos*maxCompartments*3];
				double sum=0;
				for(int k=0;k<nc;++k, pdd+=3){
					sum+=sc[k];
					voxE->setRotationMatrixFromPDD(k,pdd);
				}
				for(int k=0;k<nc;++k){
					sc[k]/=sum;
				}
				voxE->setVolumeFractions(sc);
				voxE->setDiffusivities(lambda);
			}
		}
	}
	delete[] sc;
}

void MultiTensorField::loadFromNifti(const std::string &nTensFname, const std::string &sizeCompartmentFname, const std::string &fPDDFname){
	nifti_image *nTens=nifti_image_read(nTensFname.c_str(), 1);
	nifti_image *sizeCompartment=nifti_image_read(sizeCompartmentFname.c_str(), 1);
	nifti_image *fPDD=nifti_image_read(fPDDFname.c_str(), 1);
	int p=0;
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j,++p){
			mat[p]=fPDD->sto_xyz.m[i][j];
		}
	}
	
	nslices=fPDD->nz;
	nrows=fPDD->ny;
	ncols=fPDD->nx;
	int maxTensorsPerVoxel=fPDD->nt;
	allocate(nslices, nrows, ncols);

	float *nTensData=(float *)nTens->data;
	float *sizeCompartmentData=(float *)sizeCompartment->data;
	float *fPDDData=(float *)fPDD->data;
	int nVoxels=nrows*ncols*nslices;

	MultiTensor *voxE=voxels;
	int nc;
	double *sc=new double[maxTensorsPerVoxel];
	double *pdd=new double[3];
	//double lambda[3]={0.000000001393979, 0.000000001393979,0.000000001974300 };
	double lambda[3]={0.001393979, 0.001393979,0.001974300 };
	int nonzero=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++voxE){

				int pos=s*(nrows*ncols)+(r)*ncols+c;
				if(pos==16374){
					pos=pos;
				}
				nc=nTensData[c + ncols*((nrows-1-r) + nrows*s)];
				if(nc>0){
					++nonzero;
				}

				voxE->allocate(nc);
				for(int k=0;k<nc;++k){
					sc[k]=sizeCompartmentData[c + ncols*((nrows-1-r) + nrows*(s + nslices*k))];
					pdd[0]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxTensorsPerVoxel*0)))];
					pdd[1]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxTensorsPerVoxel*1)))];
					pdd[2]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxTensorsPerVoxel*2)))];
					voxE->setRotationMatrixFromPDD(k,pdd);
				}
				voxE->setVolumeFractions(sc);
				voxE->setDiffusivities(lambda);
			}
		}
	}
	cerr<<"Non zero voxels: "<<nonzero<<endl;
	delete[] pdd;
	delete[] sc;
	nifti_image_free(nTens);
	nifti_image_free(sizeCompartment);
	nifti_image_free(fPDD);
}


void MultiTensorField::loadFromNifti(const std::string &fPDDFname){
	nifti_image *fPDD=nifti_image_read(fPDDFname.c_str(), 1);
	int p=0;
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j,++p){
			mat[p]=fPDD->sto_xyz.m[i][j];
		}
	}
	
	nslices=fPDD->nz;
	nrows=fPDD->ny;
	ncols=fPDD->nx;
	int maxTensorsPerVoxel=fPDD->nt/3;
	allocate(nslices, nrows, ncols);

	float *fPDDData=(float *)fPDD->data;
	int nVoxels=nrows*ncols*nslices;

	MultiTensor *voxE=voxels;
	int nc;
	double *sc=new double[maxTensorsPerVoxel];
	double *pdd=new double[3];
	//double lambda[3]={0.000000001393979, 0.000000001393979,0.000000001974300 };
	double lambda[3]={0.001393979, 0.001393979,0.001974300 };
	int nonzero=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++voxE){
				int pos=s*(nrows*ncols)+(r)*ncols+c;
				nc=0;
				for(int k=0;k<maxTensorsPerVoxel;++k){
					pdd[0]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+0 )))];
					if(!isNumber(pdd[0])){
						continue;
					}
					pdd[1]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+1 )))];
					if(!isNumber(pdd[1])){
						continue;
					}
					pdd[2]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+2 )))];
					if(!isNumber(pdd[2])){
						continue;
					}
					double nrm=sqrt(SQR(pdd[0])+SQR(pdd[1])+SQR(pdd[1]));
					if(nrm<1e-9){
						continue;
					}
					sc[nc]=nrm;
					++nc;
				}
				voxE->allocate(nc);
				nc=0;
				for(int k=0;k<maxTensorsPerVoxel;++k){
					pdd[0]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k)))];
					if(!isNumber(pdd[0])){
						continue;
					}
					pdd[1]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+1)))];
					if(!isNumber(pdd[1])){
						continue;
					}
					pdd[2]=fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+2)))];
					if(!isNumber(pdd[2])){
						continue;
					}
					double nrm=sqrt(SQR(pdd[0])+SQR(pdd[1])+SQR(pdd[1]));
					if(nrm<1e-9){
						continue;
					}
					voxE->setRotationMatrixFromPDD(nc,pdd);
					++nc;
				}
				voxE->setVolumeFractions(sc);
				voxE->setDiffusivities(lambda);
			}
		}
	}
	cerr<<"Non zero voxels: "<<nonzero<<endl;
	delete[] pdd;
	delete[] sc;
	nifti_image_free(fPDD);
}

void MultiTensorField::saveSinglePeaksFile(const std::string &peaksFileName, int maxCompartments){
	memset(mat, 0, sizeof(mat));
	for(int i=0;i<4;++i){
		mat[4*i+i]=1;
	}
	mat[3]=ncols*0.5;
	mat[7]=nrows*0.5;
	mat[11]=nslices*0.5;
	int nvoxels=ncols*nrows*nslices;
	
	int dims[5]={4, ncols, nrows, nslices,maxCompartments*3};
	
	//create nifti images
	nifti_image *fPDD=nifti_make_new_nim(dims,DT_FLOAT32,true);	
	fPDD->sform_code=1;
	//set file names
	nifti_set_filenames(fPDD,peaksFileName.c_str(),0,1);
	//set transformation to image coordinates
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			fPDD->sto_ijk.m[i][j]=this->mat[i*4+j];
			fPDD->qto_ijk.m[i][j]=this->mat[i*4+j];
		}
	}
	//--
	mat[3]=-ncols*0.5;
	mat[7]=-nrows*0.5;
	mat[11]=-nslices*0.5;
	//set transformation to world coordinates
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			fPDD->sto_xyz.m[i][j]=this->mat[i*4+j];
			fPDD->qto_xyz.m[i][j]=this->mat[i*4+j];
		}
	}
	//fill data
	float *fPDDData=(float *)fPDD->data;
	int nVoxels=nrows*ncols*nslices;
	memset(fPDDData, 0, sizeof(float)*nvoxels*maxCompartments*3);

	MultiTensor *voxE=voxels;
	double pdd[3];
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++voxE){
				int nc=voxE->getNumCompartments();
				double *sc=voxE->getVolumeFractions();
				for(int k=0;k<nc;++k){
					voxE->getPDD(k,pdd);
					double nrm=sqrt(SQR(pdd[0])+SQR(pdd[1])+SQR(pdd[2]));
					pdd[0]*=sc[k]/nrm;
					pdd[1]*=sc[k]/nrm;
					pdd[2]*=sc[k]/nrm;
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k)))]=pdd[0];
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+1)))]=pdd[1];
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(3*k+2)))]=pdd[2];
				}
			}
		}
	}
	//write image
	nifti_image_write(fPDD);
	//free image
	nifti_image_free(fPDD);
}

void MultiTensorField::saveAsNifti(const std::string &nTensFname, const std::string &sizeCompartmentFname, const std::string &fPDDFname){
	memset(mat, 0, sizeof(mat));
	for(int i=0;i<4;++i){
		mat[4*i+i]=1;
	}
	mat[3]=ncols*0.5;
	mat[7]=nrows*0.5;
	mat[11]=nslices*0.5;
	int nvoxels=ncols*nrows*nslices;
	int maxCompartments=3;
	for(int i=0;i<nvoxels;++i){
		int opc=voxels[i].getNumCompartments();
		maxCompartments=MAX(maxCompartments, opc);
	}
	int dims3[4]={3, ncols, nrows, nslices};
	int dims4[5]={4, ncols, nrows, nslices,maxCompartments};
	int dims5[6]={5, ncols, nrows, nslices, maxCompartments,3};
	
	//create nifti images
	nifti_image *nTens=nifti_make_new_nim(dims3,DT_FLOAT32,true);
	nifti_image *sizeCompartment=nifti_make_new_nim(dims4,DT_FLOAT32,true);
	nifti_image *fPDD=nifti_make_new_nim(dims5,DT_FLOAT32,true);
	nTens->sform_code=1;
	sizeCompartment->sform_code=1;
	fPDD->sform_code=1;
	
	//set file names
	nifti_set_filenames(nTens,nTensFname.c_str(),0,1);
	nifti_set_filenames(sizeCompartment,sizeCompartmentFname.c_str(),0,1);
	nifti_set_filenames(fPDD,fPDDFname.c_str(),0,1);
	//set transformation to image coordinates
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			nTens->sto_ijk.m[i][j]=this->mat[i*4+j];
			nTens->qto_ijk.m[i][j]=this->mat[i*4+j];
			sizeCompartment->sto_ijk.m[i][j]=this->mat[i*4+j];
			sizeCompartment->qto_ijk.m[i][j]=this->mat[i*4+j];
			fPDD->sto_ijk.m[i][j]=this->mat[i*4+j];
			fPDD->qto_ijk.m[i][j]=this->mat[i*4+j];
		}
	}
	//--
	mat[3]=-ncols*0.5;
	mat[7]=-nrows*0.5;
	mat[11]=-nslices*0.5;
	//set transformation to world coordinates
	
	for(int i=0;i<4;++i){
		for(int j=0;j<4;++j){
			nTens->sto_xyz.m[i][j]=this->mat[i*4+j];
			nTens->qto_xyz.m[i][j]=this->mat[i*4+j];
			sizeCompartment->sto_xyz.m[i][j]=this->mat[i*4+j];
			sizeCompartment->qto_xyz.m[i][j]=this->mat[i*4+j];
			fPDD->sto_xyz.m[i][j]=this->mat[i*4+j];
			fPDD->qto_xyz.m[i][j]=this->mat[i*4+j];
		}
	}
	//fill data
	float *nTensData=(float *)nTens->data;
	float *sizeCompartmentData=(float *)sizeCompartment->data;
	float *fPDDData=(float *)fPDD->data;
	int nVoxels=nrows*ncols*nslices;

	MultiTensor *voxE=voxels;
	int nc;
	double pdd[3];
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++voxE){
				nc=voxE->getNumCompartments();
				nTensData[c + ncols*((nrows-1-r) + nrows*s)]=nc;
				double *sc=voxE->getVolumeFractions();
				for(int k=0;k<nc;++k){
					voxE->getPDD(k,pdd);
					sizeCompartmentData[c + ncols*((nrows-1-r) + nrows*(s + nslices*k))]=sc[k];
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxCompartments*0)))]=pdd[0];
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxCompartments*1)))]=pdd[1];
					fPDDData[c + ncols*((nrows-1-r) + nrows*(s + nslices*(k + maxCompartments*2)))]=pdd[2];
				}
				voxE->setVolumeFractions(sc);
			}
		}
	}
	//write images
	nifti_image_write(nTens);
	nifti_image_write(sizeCompartment);
	nifti_image_write(fPDD);
	//free images
	nifti_image_free(nTens);
	nifti_image_free(sizeCompartment);
	nifti_image_free(fPDD);
}

void MultiTensorField::copyFrom(MultiTensorField &M){
	allocate(M.getNumSlices(), M.getNumRows(), M.getNumCols());
	int nVoxels=nrows*ncols*nslices;
	memcpy(error, M.getError(), sizeof(double)*nVoxels);
	MultiTensor *mvox=M.getVoxels();
	for(int i=0;i<nVoxels;++i){
		voxels[i].copyFrom(mvox[i]);
	}
	memcpy(dualLatticeCol,M.dualLatticeCol, sizeof(double)*nVoxels);
	memcpy(dualLatticeRow,M.dualLatticeRow, sizeof(double)*nVoxels);
	memcpy(dualLatticeSlice,M.dualLatticeSlice, sizeof(double)*nVoxels);
}

void MultiTensorField::copyFrom(MultiTensorField &M, set<int> &slices, int sliceType){
	int numSelectedSlices=slices.size();
	if(sliceType==0){//axial
		allocate(numSelectedSlices, M.getNumRows(), M.getNumCols());
		int nVoxels=nrows*ncols*numSelectedSlices;

		MultiTensor *mvox=M.getVoxels();
		int newSliceIndex=0;
		for(set<int>::iterator it=slices.begin(); it!=slices.end(); ++it, ++newSliceIndex){
			int s=*it;
			int pos=s*(nrows*ncols);
			int newPos=newSliceIndex*(nrows*ncols);
			for(int r=0;r<nrows;++r){
				for(int c=0;c<ncols;++c, ++pos, ++newPos){
					voxels[newPos].copyFrom(mvox[pos]);
					error[newPos]=M.error[pos];
					dualLatticeCol[newPos]=M.dualLatticeCol[pos];
					dualLatticeRow[newPos]=M.dualLatticeRow[pos];
					dualLatticeSlice[newPos]=M.dualLatticeSlice[pos];
				}
			}
		}
	}else if(sliceType==1){//coronal
		allocate(M.getNumSlices(), numSelectedSlices, M.getNumCols());
		int nVoxels=nslices*numSelectedSlices*ncols;

		MultiTensor *mvox=M.getVoxels();
		for(int s=0;s<nslices;++s){
			int newPos=s*(numSelectedSlices*ncols);
			for(set<int>::iterator it=slices.begin(); it!=slices.end(); ++it){
				int r=*it;
				int pos=s*(M.getNumRows()*M.getNumCols())+r*M.getNumCols();
				for(int c=0;c<ncols;++c, ++pos, ++newPos){
					voxels[newPos].copyFrom(mvox[pos]);
					error[newPos]=M.error[pos];
					dualLatticeCol[newPos]=M.dualLatticeCol[pos];
					dualLatticeRow[newPos]=M.dualLatticeRow[pos];
					dualLatticeSlice[newPos]=M.dualLatticeSlice[pos];
				}
			}
		}
	}else{//sagital
		allocate(M.getNumSlices(), M.getNumRows(), numSelectedSlices);
		int nVoxels=nslices*nrows*numSelectedSlices;
		MultiTensor *mvox=M.getVoxels();
		for(int s=0;s<nslices;++s){
			int newPos=s*(nrows*numSelectedSlices);
			for(int r=0;r<nrows;++r){
				for(set<int>::iterator it=slices.begin(); it!=slices.end(); ++it, ++newPos){
					int c=*it;
					int pos=s*(M.getNumRows()*M.getNumCols())+r*M.getNumCols()+c;
					voxels[newPos].copyFrom(mvox[pos]);
					error[newPos]=M.error[pos];
					dualLatticeCol[newPos]=M.dualLatticeCol[pos];
					dualLatticeRow[newPos]=M.dualLatticeRow[pos];
					dualLatticeSlice[newPos]=M.dualLatticeSlice[pos];
				}
			}
		}
	}
	
}


int MultiTensorField::getNumRows(void)const{
	return nrows;
}

int MultiTensorField::getNumCols(void)const{
	return ncols;
}

int MultiTensorField::getNumSlices(void)const{
	return nslices;
}
void MultiTensorField::setShowGroupColors(bool b){
	showGroupColors=b;
}
		
bool MultiTensorField::getShowGroupColors(void){
	return showGroupColors;
}

double *MultiTensorField::getDualLatticeRow(void){
	return dualLatticeRow;
}
		
double *MultiTensorField::getDualLatticeCol(void){
	return dualLatticeCol;
}

double *MultiTensorField::getDualLatticeSlice(void){
	return dualLatticeSlice;
}

MultiTensor *MultiTensorField::getVoxels(void){
	return voxels;
}

double *MultiTensorField::getError(void){
	return error;
}

const MultiTensor *MultiTensorField::getVoxels(void)const{
	return voxels;
}

MultiTensor *MultiTensorField::getVoxelAt(int slice, int row, int col){
	if((slice<0) || (slice>=nslices)){
		return NULL;
	}
	if((row<0) || (row>=nrows)){
		return NULL;
	}
	if((col<0) || (col>=ncols)){
		return NULL;
	}
	return &(voxels[slice*nrows*ncols+row*ncols+col]);
}

void MultiTensorField::setVisualizationType(MultiTensorFieldVisualizationType _visType){
	visType=_visType;
}

void evaluatePositiveCompartmentCount(const MultiTensorField &GT, const MultiTensorField &E, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nVoxels=nrows*ncols*nslices;
	const MultiTensor *vGT=GT.getVoxels();
	const MultiTensor *vE=E.getVoxels();
	for(int i=0;i<nVoxels;++i){
		double a=vGT[i].getNumCompartments();
		if(a==0){
			continue;
		}
		double b=vE[i].getNumCompartments();
		/*if(a<b){
			double d=fabs(a-b)/a;
			vec.push_back(d);
		}else{
			vec.push_back(0);
		}*/
		if(a<b){
			vec.push_back(b-a);
		}else{
			vec.push_back(0);
		}
	}
}
void evaluateNegativeCompartmentCount(const MultiTensorField &GT, const MultiTensorField &E, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nVoxels=nrows*ncols*nslices;
	const MultiTensor *vGT=GT.getVoxels();
	const MultiTensor *vE=E.getVoxels();
	for(int i=0;i<nVoxels;++i){
		double a=vGT[i].getNumCompartments();
		if(a==0){
			continue;
		}
		double b=vE[i].getNumCompartments();
		/*if(b<a){
			double d=fabs(a-b)/a;
			vec.push_back(d);
		}else{
			vec.push_back(0);
		}*/
		if(b<a){
			vec.push_back(a-b);
		}else{
			vec.push_back(0);
		}
	}
}

void evaluateMissingWMVoxels(const MultiTensorField &GT, const MultiTensorField &E, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nVoxels=nrows*ncols*nslices;
	const MultiTensor *vGT=GT.getVoxels();
	const MultiTensor *vE=E.getVoxels();
	for(int i=0;i<nVoxels;++i){
		if((vGT[i].getNumCompartments()>0) && (vE[i].getNumCompartments()==0)){
			vec.push_back(1);
		}else{
			vec.push_back(0);
		}
	}
}

void evaluateExtraWMVoxels(const MultiTensorField &GT, const MultiTensorField &E, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nVoxels=nrows*ncols*nslices;
	const MultiTensor *vGT=GT.getVoxels();
	const MultiTensor *vE=E.getVoxels();
	for(int i=0;i<nVoxels;++i){
		if((vGT[i].getNumCompartments()==0) && (vE[i].getNumCompartments()!=0)){
			vec.push_back(1);
		}else{
			vec.push_back(0);
		}
	}
}


void evaluateAngularPrecision(const MultiTensorField &GT, const MultiTensorField &E, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nvox=nrows*ncols*nslices;
	const MultiTensor *voxGT=GT.getVoxels();
	const MultiTensor *voxE=E.getVoxels();

	int maxCompartmentsGT=0;
	for(int i=0;i<nvox;++i){
		maxCompartmentsGT=MAX(maxCompartmentsGT, voxGT[i].getNumCompartments());
	}
	int maxCompartmentsE=0;
	for(int i=0;i<nvox;++i){
		maxCompartmentsE=MAX(maxCompartmentsE, voxE[i].getNumCompartments());
	}
	int maxCompartments=MAX(maxCompartmentsGT, maxCompartmentsE);
	Hungarian hungarianSolver(maxCompartments);
	//-----
	double *pddGT=new double[maxCompartmentsGT*3];
	double *pddE=new double[maxCompartmentsE*3];
	
	int costLen=maxCompartments*maxCompartments;
	for(int v=0;v<nvox;++v)if(voxE[v].getNumCompartments()>0){
		//fill cost matrix
		int n=voxGT[v].getNumCompartments();
		int m=voxE[v].getNumCompartments();
		if(n*m==0){
			continue;
		}
		//get pdds
		voxGT[v].getPDDs(pddGT);
		voxE[v].getPDDs(pddE);
		hungarianSolver.setSize(MAX(n,m));
		hungarianSolver.setAllCosts(1e5);
		double *p=pddGT;
		for(int i=0;i<n;++i, p+=3){
			double *q=pddE;
			for(int j=0;j<m;++j, q+=3){
				double angle=getAbsAngle(p,q,3);
				hungarianSolver.setCost(i,j,angle);
			}
		}
		double cost=hungarianSolver.solve();
		cost-=1e5*abs(n-m);
		//cost+=abs(n-m)*M_PI_2;
		cost/=MIN(n,m);
		vec.push_back(cost*180/M_PI);
	}else if(voxGT[v].getNumCompartments()>0){
		vec.push_back(90);
	}
	delete[] pddE;
	delete[] pddGT;
}

void evaluateODFAccuracy(const MultiTensorField &GT, const MultiTensorField &E, double *directions, int numDirections, vector<double> &vec){
	vec.clear();
	int nrows=GT.getNumRows();
	int ncols=GT.getNumCols();
	int nslices=GT.getNumSlices();
	if(nrows!=E.getNumRows()){
		return;
	}
	if(ncols!=E.getNumCols()){
		return;
	}
	if(nslices!=E.getNumSlices()){
		return;
	}
	int nVoxels=nrows*ncols*nslices;
	const MultiTensor *vGT=GT.getVoxels();
	const MultiTensor *vE=E.getVoxels();

	double *odfGT=new double[numDirections];
	double *odfE=new double[numDirections];
	for(int i=0;i<nVoxels;++i){
		int n=vGT[i].getNumCompartments();
		if(n==0){
			continue;
		}
		int m=vE[i].getNumCompartments();
		if(n*m==0){
			continue;
		}
		vGT[i].computeODF(directions, numDirections,odfGT);
		vE[i].computeODF(directions, numDirections,odfE);
		double num=0;
		double den=0;
		for(int j=0;j<numDirections;++j){
			num+=SQR(odfGT[j]-odfE[j]);
			den+=SQR(odfGT[j]);
		}
		double d=num/den;
		vec.push_back(d);
	}
	delete[] odfGT;
	delete[] odfE;
}
MultiTensorField *createRealisticSyntheticField(int nrows, int ncols, int nslices, double minAngle, double *diffusivities,double *randomPDDs, int nRandom){
	const int maxPdds=2;
	MultiTensorField *field=new MultiTensorField(nslices, nrows, ncols);
	MultiTensor *voxels=field->getVoxels();
	
	double *currentPdds=new double[3*maxPdds];
	double currentAmounts[maxPdds];
	int pStart=0;
	int pos=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				voxels[pos].allocate(maxPdds);
				memset(currentPdds,0,sizeof(double)*maxPdds*3);
				memset(currentAmounts,0,sizeof(double)*maxPdds);
				selectPDDs(randomPDDs, nRandom, 2, minAngle, currentPdds, pStart);
				currentAmounts[0]=uniform(0.4,0.6);
				currentAmounts[1]=1.0-currentAmounts[0];

				voxels[pos].setDiffusivities(diffusivities);
				for(int k=0;k<maxPdds;++k){
					voxels[pos].setRotationMatrixFromPDD(k, &currentPdds[3*k]);
				}
				voxels[pos].setVolumeFractions(currentAmounts);
			}
		}
	}
	delete[] currentPdds;
	return field;
}

void MultiTensorField::setDualLattice(int voxIndexA, int voxIndexB, double val){
	if(voxIndexB<voxIndexA){
		int tmp=voxIndexA;
		voxIndexA=voxIndexB;
		voxIndexB=tmp;
	}
	int pSlice=voxIndexA/(nrows*ncols);
	int pRow=(voxIndexA%(nrows*ncols))/ncols;
	int pCol=voxIndexA%ncols;
	if((voxIndexA+1)==voxIndexB){
		dualLatticeCol[voxIndexA]=val;
	}else if((voxIndexA+ncols)==voxIndexB){
		dualLatticeRow[voxIndexA]=val;
	}else if((voxIndexA+ncols*nrows)==voxIndexB){//
		dualLatticeSlice[voxIndexA]=val;
	}
}

void MultiTensorField::normalizeLatticeSeparations(void){
	if(dualLatticeRow==NULL){
		return;
	}
	if(dualLatticeCol==NULL){
		return;
	}
	if(dualLatticeSlice==NULL){
		return;
	}
	int nvox=nrows*ncols*nslices;
	double minVal=getMinVal(dualLatticeRow,nvox);
	minVal=MIN(minVal, getMinVal(dualLatticeCol,nvox));
	minVal=MIN(minVal, getMinVal(dualLatticeSlice,nvox));

	double maxVal=getMaxVal(dualLatticeRow,nvox);
	maxVal=MAX(maxVal, getMaxVal(dualLatticeCol,nvox));
	maxVal=MAX(maxVal, getMaxVal(dualLatticeSlice,nvox));
	double diff=maxVal-minVal;
	if(diff<EPS_MTF){
		return;
	}
	for(int i=0;i<nvox;++i){
		dualLatticeRow[i]=(dualLatticeRow[i]-minVal)/diff;
		dualLatticeCol[i]=(dualLatticeCol[i]-minVal)/diff;
		dualLatticeSlice[i]=(dualLatticeSlice[i]-minVal)/diff;
	}
}

void MultiTensorField::generateDWMRIVolume(double *gradients, int numGradients, double b, int SNR, double *&s0Volume, double *&dwVolume){
	int nvoxels=nslices*nrows*ncols;
	s0Volume=new double[nvoxels];
	dwVolume=new double[numGradients*nvoxels];
	for(int v=0;v<nvoxels;++v){
		s0Volume[v]=1;
		double *S=&dwVolume[v*numGradients];
		memset(S, 0, sizeof(double)*numGradients);
		double sigma=0;
		if(SNR>0){
			sigma=1.0/SNR;
		}
		voxels[v].acquireWithScheme(b, gradients, numGradients, sigma,S);
		double refErr=0;
	}

}

void MultiTensorField::saveODFFieldFromAlpha(double *ODFDirections, int nODFDirections, double *DBFDirections, int nDBFDirections, double *lambda, FILE *F){
	double e[3]={0,0,1};
	double *invTensors=new double[9*nDBFDirections];
	for(int i=0;i<nDBFDirections;++i){
		double R[9];
		fromToRotation(e, &DBFDirections[3*i], R);
		double Di[9]={//Di=D^{-1}*R'
			R[0]/lambda[0], R[3]/lambda[0], R[6]/lambda[0],
			R[1]/lambda[1], R[4]/lambda[1], R[7]/lambda[1],
			R[2]/lambda[2], R[5]/lambda[2], R[8]/lambda[2]
		};
		multMatrixMatrix<double>(R,Di,3,&invTensors[9*i]);
	}
	double *ODF=new double[nODFDirections];
	double prodDiffusivities=lambda[0]*lambda[1]*lambda[2];
	int pos=0;
	int voxelsProcessed=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				REPORT_PROGRESS_100(pos, (nslices*nrows*ncols), cerr);
				double *volumeFractions=voxels[pos].getAlpha();
				if(volumeFractions==NULL){
					memset(ODF, 0, sizeof(double)*nODFDirections);
				}else{
					++voxelsProcessed;
					double thr=getMaxVal(volumeFractions, nDBFDirections);
					thr*=0.2;
					
					for(int idx=0;idx<nODFDirections;++idx){
						double *r=&ODFDirections[3*idx];//evaluate ODF along r
						ODF[idx]=0;
						for(int i=0;i<nDBFDirections;++i){
							if(volumeFractions[i]<=thr){
								continue;
							}
							double eval=evaluateQuadraticForm(&invTensors[9*i], r, 3);
							eval=sqrt(eval);
							eval=eval*eval*eval;
							ODF[idx]+=volumeFractions[i]/(4*M_PI*eval*sqrt(prodDiffusivities));
						}
					}
					double sum=0;
					for(int i=0;i<nODFDirections;++i){
						sum+=ODF[i];
					}
					for(int i=0;i<nODFDirections;++i){
						ODF[i]=ODF[i]/sum;
					}
				}
				fwrite(ODF, sizeof(double), nODFDirections, F);
			}
		}
	}
	cerr<<voxelsProcessed<<" voxels processed"<<endl;
	delete[] ODF;
	delete[] invTensors;
}

#ifdef USE_QT




void MultiTensorField::setDBFDirections(double **_DBFDirections){
	DBFDirections=_DBFDirections;
}



void MultiTensorField::drawMultiTensor(MultiTensor &mt, int component, double px, double py, double pz, double intensity){
	if(visType==MTFVT_Arrows){
		mt.drawArrows(px,py,pz,showGroupColors);
	}else if((visType==MTFVT_ApparentDiffusion) || (visType==MTFVT_OrientationDiffusion)){
		mt.drawDiffusionFunction(px,py,pz,visType==MTFVT_OrientationDiffusion);
	}else if(visType==MTFVT_ClusterColors){
		mt.drawEllipsoid(component,px,py,pz,intensity, true);
	}else if(visType==MTFVT_SampledFunction){
		mt.drawDiffusionFunction(px,py,pz,0);
		mt.drawSampledFunction(px,py,pz,samplingPoints,numSamplingPoints);
	}else{
		mt.drawEllipsoid(component,px,py,pz,intensity, false);
	}
}


void MultiTensorField::drawPDDSlice(int slice, int k_pdd){
	if(slice>=nslices){
		return;
	}
	double cx=ncols/2;
	double cy=nrows/2;
	double cz=nslices/2;
	int nVoxels=nrows*ncols*nslices;
	
	int minTensors=1e9;
	int maxTensors=-1e9;
	double maxError=-1e10;
	double minError=1e10;
	for(int i=0;i<nVoxels;++i){
		int opc=voxels[i].getNumCompartments();
		if(opc<=0){
			continue;
		}
		minTensors=MIN(minTensors, opc);
		maxTensors=MAX(maxTensors, opc);

		//if(fabs(error[i])>0){
			minError=MIN(minError, error[i]);
			maxError=MAX(maxError, error[i]);
		//}
		
	}

	//cerr<<"Slice "<<slice<<". MinDTCount="<<minTensors<<". MaxDTCount="<<maxTensors<<endl;
	//cerr<<"Slice "<<slice<<". MinError="<<minError<<". MaxError="<<maxError<<endl;
	double diffNZCount=maxTensors-minTensors;
	double diffError=maxError-minError;
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			//double *pdd=&pdds[3*maxTensorsPerVoxel*(slice*(nrows*ncols) + r*ncols + c) + 3*k_pdd];
			//MultiTensor &currentVoxel=voxels[slice*(nrows*ncols)+(nrows-1-r)*ncols+c];
			MultiTensor &currentVoxel=voxels[slice*(nrows*ncols)+r*ncols+c];
			double kth_pdd[3];
			currentVoxel.getPDD(k_pdd,kth_pdd);
			if(currentVoxel.getNumCompartments()){
				double intensity=-1;
				switch(visType){
							case MTFVT_Orientation:
								//nothing
							break;
							case MTFVT_Sparcity:
								intensity=(currentVoxel.getNumCompartments()-minTensors)/diffNZCount;
							break;
							case MTFVT_Error:
								intensity=(error[slice*(nrows*ncols)+(nrows-1-r)*ncols+c]-minError)/diffError;
							break;
							case MTFVT_Arrows:
								//nothing
							break;
							case MTFVT_ApparentDiffusion:
								//nothing
							break;
							case MTFVT_OrientationDiffusion:
								//nothing
							break;
				}
				drawMultiTensor(currentVoxel, k_pdd, c-cx, (nrows-1-r)-cy, slice-cz, intensity);
			}
		}
	}
}

void MultiTensorField::setDualLatticeCol(int s, int r, int c, double val){
	if(c>=(ncols-1)){
		return;
	}
	int pos=s*nrows*ncols+r*ncols+c;
	dualLatticeCol[pos]=val;
}

void MultiTensorField::setDualLatticeRow(int s, int r, int c, double val){
	if(r>=(nrows-1)){
		return;
	}
	int pos=s*nrows*ncols+r*ncols+c;
	dualLatticeRow[pos]=val;
}

void MultiTensorField::setDualLatticeSlice(int s, int r, int c, double val){
	if(s>=(nslices-1)){
		return;
	}
	int pos=s*nrows*ncols+r*ncols+c;
	dualLatticeSlice[pos]=val;
}



void MultiTensorField::drawLatticeSeparation(int voxIndexA, int voxIndexB, double val){
	if(val<EPS_MTF){
		return;
	}
	if(voxIndexB<voxIndexA){
		int tmp=voxIndexA;
		voxIndexA=voxIndexB;
		voxIndexB=tmp;
	}
	int pSlice=voxIndexA/(nrows*ncols);
	int pRow=(voxIndexA%(nrows*ncols))/ncols;
	int pCol=voxIndexA%ncols;
	float pddColor[4];
	double r,g,b;
	getIntensityColor<double>(255*val, r, g, b);
	pddColor[0]=r/255.0;
	pddColor[1]=g/255.0;
	pddColor[2]=b/255.0;
	pddColor[3]=val;
		
	double cx=ncols/2;
	double cy=nrows/2;
	double cz=nslices/2;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, pddColor);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glTranslated(pCol-cx,(nrows-1-pRow)-cy,pSlice-cz);	
	if((voxIndexA+1)==voxIndexB){
		glBegin( GL_QUADS );
			glVertex3d( 0.5, -0.5, -0.5);
			glVertex3d( 0.5, -0.5,  0.5);
			glVertex3d( 0.5,  0.5,  0.5);
			glVertex3d( 0.5,  0.5, -0.5);
		glEnd();
		glBegin( GL_QUADS );
			glVertex3d( 0.5, -0.5,  0.5);
			glVertex3d( 0.5, -0.5, -0.5);
			glVertex3d( 0.5,  0.5, -0.5);
			glVertex3d( 0.5,  0.5,  0.5);
		glEnd();
	}else if((voxIndexA+ncols)==voxIndexB){
		glBegin( GL_QUADS );
			glVertex3d( -0.5,  -0.5, -0.5);
			glVertex3d(  0.5,  -0.5, -0.5);
			glVertex3d(  0.5,  -0.5,  0.5);
			glVertex3d( -0.5,  -0.5,  0.5);
		glEnd();
		glBegin( GL_QUADS );
			glVertex3d( -0.5,  -0.5,  0.5);
			glVertex3d(  0.5,  -0.5,  0.5);
			glVertex3d(  0.5,  -0.5, -0.5);
			glVertex3d( -0.5,  -0.5, -0.5);
		glEnd();
	}else if((voxIndexA+ncols*nrows)==voxIndexB){//
		glBegin( GL_QUADS );
			glVertex3d(  0.5,  0.5,  0.5);
			glVertex3d( -0.5,  0.5,  0.5);
			glVertex3d( -0.5, -0.5,  0.5);
			glVertex3d(  0.5, -0.5,  0.5);
		glEnd();
		glBegin( GL_QUADS );
			glVertex3d(  0.5, -0.5,  0.5);
			glVertex3d( -0.5, -0.5,  0.5);
			glVertex3d( -0.5,  0.5,  0.5);
			glVertex3d(  0.5,  0.5,  0.5);
		glEnd();
	}
	
	glPopMatrix();
}

void MultiTensorField::drawLatticeSlice(int slice){
	if((dualLatticeCol==NULL) || (dualLatticeRow==NULL) || (dualLatticeSlice==NULL)){
		return;
	}
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			int pos=slice*nrows*ncols+r*ncols+c;
			if(c<(ncols-1)){
				drawLatticeSeparation(pos, pos+1, dualLatticeCol[pos]);
			}
			if(r<(nrows-1)){
				drawLatticeSeparation(pos, pos+ncols, dualLatticeRow[pos]);
			}
			if(slice<(nslices-1)){
				drawLatticeSeparation(pos, pos+ncols*nrows, dualLatticeSlice[pos]);
			}
		}
	}
}

void findClosest(int k, double *dir, double *base, int baseSize, vector<int> &ind){
	if(baseSize<k){
		k=baseSize;
	}
	vector<pair<double, int> >v;
	for(int i=0;i<baseSize;++i){
		double angle=getAbsAngleDegrees(dir, &base[3*i], 3);
		v.push_back(make_pair(angle,i));
	}
	sort(v.begin(), v.end());
	ind.clear();
	for(int i=0;i<k;++i){
		ind.push_back(v[i].second);
	}
}


void codifyPDD(double *pdd, int k, double *base, int baseSize, double *alpha){
	if(baseSize<k){
		k=baseSize;
	}
	vector<int> closest;
	findClosest(k, pdd, base, baseSize, closest);
	memset(alpha, 0, sizeof(double)*baseSize);
	double sumAlpha=0;
	for(int i=0;i<k;++i){
		double prod=dotProduct(pdd, &base[3*closest[i]], 3);
		alpha[closest[i]]=prod;
		sumAlpha+=prod;
	}
	for(int i=0;i<k;++i){
		alpha[closest[i]]/=sumAlpha;
	}
}

int MultiTensorField::extractCodifiedTensorList(int neighSize, double *dir, int nDir, double *&positions, double *&orientations, double *&alpha, int &numTensors, int *&tensorIndex, int &maxTensorsPerVoxel, int *&angleIndex, int slice){
	numTensors=0;
	set<double> A;
	maxTensorsPerVoxel=-1;
	for(int s=0;s<nslices;++s)if((slice==-1)||(s==slice)){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int p=s*nrows*ncols+r*ncols+c;
				int nc=voxels[p].getNumCompartments();
				if(maxTensorsPerVoxel<nc){
					maxTensorsPerVoxel=nc;
				}
				numTensors+=nc;
				double pdd[3];
				double ref[3]={1.0, 0.0, 0.0};
				for(int k=0;k<nc;++k){
					voxels[p].getPDD(k,pdd);
					double angle=getAbsAngleDegrees(pdd,ref,3);
					bool newAngle=true;
					for(set<double>::iterator it=A.begin();it!=A.end();++it){
						if(fabs(*it - angle)<0.1){
							newAngle=false;
							break;
						}
					}
					if(newAngle){
						A.insert(angle);
					}
				}
			}
		}
	}
	tensorIndex=new int[numTensors];
	angleIndex=new int[numTensors];
	positions=new double[3*numTensors];
	orientations=new double[3*numTensors];
	alpha=new double[nDir*numTensors];

	numTensors=0;
	for(int s=0;s<nslices;++s)if((slice==-1)||(s==slice)){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				//--split multitensor at [s,r,c] and codify--
				int p=s*nrows*ncols+r*ncols+c;
				
				int nc=voxels[p].getNumCompartments();
				double pdd[3];
				for(int k=0;k<nc;++k){
					int tp=maxTensorsPerVoxel*p+k;
					double *currentPos=&positions[3*numTensors];
					double *currentOrientation=&orientations[3*numTensors];
					double *currentAlpha=&alpha[nDir*numTensors];
					//--register position--
					currentPos[0]=double(c);
					currentPos[1]=double(nrows-1-r);
					currentPos[2]=double(s);
					//--extract pdd--
					voxels[p].getPDD(k,pdd);
					//--register orientations--
					currentOrientation[0]=pdd[0];
					currentOrientation[1]=pdd[1];
					currentOrientation[2]=pdd[2];
					//--decompose in terms of the base--
					//codifyPDD(pdd, neighSize, dir, nDir, currentAlpha);

					//--find the alpha-index--
					double ref[3]={1.0, 0.0, 0.0};
					double angle=getAbsAngleDegrees(pdd, ref,3);
					tensorIndex[numTensors]=tp;
					int idx=0;
					for(set<double>::iterator it=A.begin();it!=A.end();++it){
						if(fabs(*it - angle)<0.1){
							break;
						}
						++idx;
					}
					angleIndex[numTensors]=idx;
					//-----------------------
					++numTensors;
				}
			}
		}
	}
	return 0;
}

int MultiTensorField::splitSlice(int slice, MultiTensorField &mtf){
	if((slice<0) || (slice>=nslices)){
		return -1;
	}
	int maxCompartments=0;
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			MultiTensor *mt=this->getVoxelAt(slice,r,c);
			int nc=mt->getNumCompartments();
			maxCompartments=MAX(maxCompartments, nc);
		}
	}
	mtf.allocate(maxCompartments,nrows, ncols);
	
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			MultiTensor *mt=this->getVoxelAt(slice,r,c);
			int nc=mt->getNumCompartments();
			for(int k=0;k<maxCompartments;++k){
				MultiTensor *t=mtf.getVoxelAt(k,r,c);
				if(k>=nc){
					t->allocate(0);
				}else{
					t->allocate(1);
					t->copyComponentFrom(*mt,k,0);
				}
			}
		}
	}
	return 0;

}

int MultiTensorField::splitSliceToAngleVolume(int slice, MultiTensorField &mtf){
	if((slice<0) || (slice>=nslices)){
		return -1;
	}
	set<double> A;
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			MultiTensor *mt=this->getVoxelAt(slice,r,c);
			int nc=mt->getNumCompartments();
				double pdd[3];
				double ref[3]={1.0, 1.0, 0.0};
				for(int k=0;k<nc;++k){
					mt->getPDD(k,pdd);
					double angle=getAbsAngleDegrees(pdd,ref,3);
					bool newAngle=true;
					for(set<double>::iterator it=A.begin();it!=A.end();++it){
						if(fabs(*it - angle)<0.1){
							newAngle=false;
							break;
						}
					}
					if(newAngle){
						A.insert(angle);
					}
				}
		}
	}
	int numAngles=A.size();
	mtf.allocate(numAngles,nrows, ncols);
	
	for(int r=0;r<nrows;++r){
		for(int c=0;c<ncols;++c){
			MultiTensor *mt=this->getVoxelAt(slice,r,c);
			int nc=mt->getNumCompartments();
			double pdd[3];
			double ref[3]={1.0, 1.0, 0.0};
			for(int l=0;l<numAngles;++l){
				MultiTensor *t=mtf.getVoxelAt(l,r,c);
				t->allocate(0);
			}
			for(int k=0;k<nc;++k){
				mt->getPDD(k,pdd);
				double angle=getAbsAngleDegrees(pdd,ref,3);
				int idx=0;
				for(set<double>::iterator it=A.begin();it!=A.end();++it){
					if(fabs(*it - angle)<0.1){
						break;
					}
					++idx;
				}
				MultiTensor *t=mtf.getVoxelAt(idx,r,c);
				t->allocate(1);
				t->copyComponentFrom(*mt,k,0);
			}
		}
	}
	return 0;
}

int MultiTensorField::ncutDiscretization(double *evec, int k, int n, int *discrete){
	for(int j=0;j<n;++j){
		double sum=0;
		for(int i=0;i<k;++i){
			sum+=evec[i*n+j]*evec[i*n+j];
		}
		sum=sqrt(sum);
		for(int i=0;i<k;++i){
			evec[i*n+j]/=sum;
		}
	}

	double *R=new double[k*k];
	memset(R,0,sizeof(double)*k*k);
	int sel=rand()%n;
	for(int i=0;i<k;++i){
		R[i]=evec[i*n+sel];
	}
	double *c=new double[n];
	double *cc=new double[n];
	memset(c,0,sizeof(double)*n);
	for(int i=1;i<k;++i){
		multVectorMatrix<double>(&R[(i-1)*k],evec, k, n, cc);
		int minIndex=0;
		for(int j=0;j<n;++j){
			c[j]+=fabs(cc[j]);
			if(c[j]<c[minIndex]){
				minIndex=j;
			}
		}
		for(int j=0;j<k;++j){
			R[i*k+j]=evec[j*n+minIndex];
		}
	}

	const int maxIter=20;
	int numIter=0;
	double tol=1e-10;
	double error=tol+1;
	double *proj=new double[n*k];
	double *centroids=new double[k*k];
	double *u=new double[k*k];
	double *s=new double[k];
	double *vt=new double[k*k];
	double lastObjectiveValue=0;
	while((tol<error) && (numIter<maxIter)){
		multMatrixMatrix(R,evec,k,k,n,proj);
		//----assign labels----
		for(int j=0;j<n;++j){
			int sel=0;
			for(int i=0;i<k;++i){
				if(proj[sel*n+j]<proj[i*n+j]){
					sel=i;
				}
			}
			discrete[j]=sel;
		}
		//-----compute centroids-----
		memset(centroids,0,sizeof(double)*k*k);
		for(int j=0;j<n;++j){
			int sel=discrete[j];
			for(int i=0;i<k;++i){
				centroids[sel*k+i]+=evec[i*n+j];
			}
		}
		//---------------------------
		computeSVD(centroids, k, k, u, s, vt);
		double ncutValue=0;
		for(int i=0;i<k;++i){
			ncutValue+=s[i];
		}
		ncutValue=2*(n-ncutValue);
		error=fabs(lastObjectiveValue-ncutValue);
		multMatrixMatrix(u,vt,k,R);
		lastObjectiveValue=ncutValue;
		++numIter;
	}
	delete[] u;
	delete[] s;
	delete[] vt;
	delete[] c;
	delete[] cc;
	delete[] proj;
	delete[] centroids;
	return 0;
}


int MultiTensorField::buildNCutSparseMatrix(double d0, double theta0, double dTheta0, double offset, SparseMatrix &S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex){
	const int NUM_NEIGHBORS=26;
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
	//--------------------
	maxTensorsPerVoxel=getMaxCompartments();
	int numTensors=getTotalCompartments();
	S.create(numTensors, NUM_NEIGHBORS*maxTensorsPerVoxel);
	sequentialIndex=new int[nslices*nrows*ncols*maxTensorsPerVoxel];
	spatialIndex=new int[numTensors];
	int p=0;
	numTensors=0;
	memset(sequentialIndex, -1, sizeof(int)*nslices*nrows*ncols*maxTensorsPerVoxel);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				int nc=voxels[p].getNumCompartments();
				for(int k=0;k<nc;++k){
					spatialIndex[numTensors]=maxTensorsPerVoxel*p+k;
					sequentialIndex[maxTensorsPerVoxel*p+k]=numTensors;
					++numTensors;
				}
			}
		}
	}
	theta0*=(M_PI/180.0);
	dTheta0*=(M_PI/180.0);

	p=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				double location[3]={double(c), double(nrows-1-r), double(s)};//			<<---Location
				int nc=voxels[p].getNumCompartments();
				//---add diagonal values---
				for(int k=0;k<nc;++k){
					int tp=maxTensorsPerVoxel*p+k;
					S.addEdge(sequentialIndex[tp],sequentialIndex[tp],1);
				}
				//-------------------------
				for(int kn=0;kn<NUM_NEIGHBORS;++kn){
					int ss=s+dSlice[kn];
					int rr=r+dRow[kn];
					int cc=c+dCol[kn];
					if((0<=rr) && (rr<nrows) && (0<=cc) && (cc<ncols) && (0<=ss) && (ss<nslices)){
						int neighPos=ss*nrows*ncols+rr*ncols+cc;
						double neighLocation[3]={double(cc), double(nrows-1-rr), double(ss)};//			<<---Neighbor's Location
						double displacementDir[3]={location[0]-neighLocation[0], location[1]-neighLocation[1], location[2]-neighLocation[2]};

						double d2=SQR(location[0]-neighLocation[0])+SQR(location[1]-neighLocation[1])+SQR(location[2]-neighLocation[2]);
						for(int k=0;k<nc;++k){
							int tp=maxTensorsPerVoxel*p+k;//									<<---spatial-index
							double pdd[3];
							voxels[p].getPDD(k,pdd);//											<<---Pdd
							int neighNc=voxels[neighPos].getNumCompartments();//number of neighbor's compartments
							for(int kk=0;kk<neighNc;++kk){
								double neighPdd[3];
								voxels[neighPos].getPDD(kk,neighPdd);//										<<---Neighbor's Pdd
								int neighTp=maxTensorsPerVoxel*neighPos+kk;//					<<---neighbor's spatial-index
								double angle;

								//if(d2<2){
									angle=getAbsAngle(pdd,neighPdd,3);
								/*}else{
									angle=getAbsAngle(pdd,displacementDir,3);
									angle=MAX(angle, getAbsAngle(neighPdd,displacementDir,3));
								}*/
								//---------------Edge weight and constraints-----------
								if(angle<dTheta0){
									int currentSIndex=sequentialIndex[tp];
									int neighSIndex=sequentialIndex[neighTp];
									double edgeWeight=d2/(d0*d0)+angle/(1-cos(2*theta0));
									//double edgeWeight=angle;
									//S.addEdge(currentSIndex, neighSIndex, exp(-edgeWeight/d0));
									S.addEdge(currentSIndex, neighSIndex, exp(-edgeWeight));
								}
								//-----------------------------------------------------
								/*int currentSIndex=sequentialIndex[tp];
								int neighSIndex=sequentialIndex[neighTp];
								//double edgeWeight=colinearityMeasure(pdd, neighPdd, displacementDir, d0, theta0, dTheta0);
								double edgeWeight=parallelityMeasure(pdd, neighPdd, displacementDir, d0, theta0, dTheta0);
								S.addEdge(currentSIndex, neighSIndex, edgeWeight);*/
							}
						}
					}
				}
			}
		}
	}
	double *d=new double[numTensors];
	double *dr=new double[numTensors];
	S.sumRowAbsValues(d);
	S.sumRowValues(dr);

	for(int i=0;i<numTensors;++i){
		dr[i]=0.5*(d[i]-dr[i])+offset;
		d[i]+=2*offset;
	}
	//S.sumToDiagonal(dr);

	for(int i=0;i<numTensors;++i){
		d[i]=1.0/sqrt(d[i]);
	}
	S.multDiagLeftRight(d,d);
	delete[] d;
	delete[] dr;
	return 0;
}

int MultiTensorField::buildFullSimilarityMatrix(double d0, double theta0, double dTheta0, double *&S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex){
	maxTensorsPerVoxel=getMaxCompartments();
	int numTensors=getTotalCompartments();
	S=new double[numTensors*numTensors];
	sequentialIndex=new int[nslices*nrows*ncols*maxTensorsPerVoxel];
	spatialIndex=new int[numTensors];
	int pos=0;
	numTensors=0;
	memset(sequentialIndex, -1, sizeof(int)*nslices*nrows*ncols*maxTensorsPerVoxel);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				int nc=voxels[pos].getNumCompartments();
				for(int k=0;k<nc;++k){
					spatialIndex[numTensors]=maxTensorsPerVoxel*pos+k;
					sequentialIndex[maxTensorsPerVoxel*pos+k]=numTensors;
					++numTensors;
				}
			}
		}
	}
	for(int i=0;i<numTensors;++i){
		int k=spatialIndex[i]%maxTensorsPerVoxel;//index of the tensor inside the Multi-tensor
		int p=spatialIndex[i]/maxTensorsPerVoxel;//index of the multi-tensor (voxel index)
		int s=p/(nrows*ncols);//slice
		int r=p%(nrows*ncols);//row
		int c=r;//column
		r/=ncols;
		c%=ncols;
		double location[3]={double(c), double(nrows-1-r), double(s)};//			<<---Location
		double pddA[3];
		voxels[p].getPDD(k,pddA);
		S[i*numTensors+i]=0;
		for(int j=i+1;j<numTensors;++j){
			int kk=spatialIndex[j]%maxTensorsPerVoxel;
			int pp=spatialIndex[j]/maxTensorsPerVoxel;
			int ss=pp/(nrows*ncols);
			int rr=pp%(nrows*ncols);
			
			int cc=rr;
			rr/=ncols;
			cc%=ncols;
			double pddB[3];
			voxels[pp].getPDD(kk,pddB);
			double neighLocation[3]={double(cc), double(nrows-1-rr), double(ss)};//			<<---Neighbor's Location
			double displacementDir[3]={location[0]-neighLocation[0], location[1]-neighLocation[1], location[2]-neighLocation[2]};
			double d2=SQR(location[0]-neighLocation[0])+SQR(location[1]-neighLocation[1])+SQR(location[2]-neighLocation[2]);

			
			if(p==pp){
				double angle=getAbsAngle(pddA,pddB,3);
				//double edgeWeight=d2/(d0*d0)+angle/theta0;
				double edgeWeight=theta0*90;
				S[i*numTensors+j]=edgeWeight;
				S[j*numTensors+i]=edgeWeight;
			}else{
				double angle1=getAbsAngleDegrees(pddA,displacementDir,3);
				double angle2=getAbsAngleDegrees(pddB,displacementDir,3);
				double angle=MAX(angle1, angle2);

				//double edgeWeight=d2/(d0*d0)+angle/theta0;
				double edgeWeight=sqrt(d2)+theta0*angle;
				//edgeWeight=exp(-edgeWeight);
				S[i*numTensors+j]=edgeWeight;
				S[j*numTensors+i]=edgeWeight;
			}
			
			
		}
	}
	return numTensors;
}

int MultiTensorField::buildNCutSparseMatrixBestAssignment(double d0, double theta0, double dTheta0, double offset, SparseMatrix &S, int &maxTensorsPerVoxel, int *&sequentialIndex, int *&spatialIndex){
	const int NUM_NEIGHBORS=26;
	//const int NUM_NEIGHBORS=6;
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
	//--------------------
	
	maxTensorsPerVoxel=getMaxCompartments();

	Hungarian hungarianSolver(maxTensorsPerVoxel);
	
	int numTensors=getTotalCompartments();
	S.create(numTensors, NUM_NEIGHBORS*maxTensorsPerVoxel);
	sequentialIndex=new int[nslices*nrows*ncols*maxTensorsPerVoxel];
	spatialIndex=new int[numTensors];
	int p=0;
	numTensors=0;
	memset(sequentialIndex, -1, sizeof(int)*nslices*nrows*ncols*maxTensorsPerVoxel);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				int nc=voxels[p].getNumCompartments();
				for(int k=0;k<nc;++k){
					spatialIndex[numTensors]=maxTensorsPerVoxel*p+k;
					sequentialIndex[maxTensorsPerVoxel*p+k]=numTensors;
					++numTensors;
				}
			}
		}
	}
	//theta0*=(M_PI/180.0);
	dTheta0*=(M_PI/180.0);

	p=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				double location[3]={double(c), double(nrows-1-r), double(s)};//			<<---Location
				int nc=voxels[p].getNumCompartments();
				double *volumeFractions=voxels[p].getVolumeFractions();
				if(nc==0){
					continue;
				}
				//---add diagonal values---
				for(int k=0;k<nc;++k){
					int tp=maxTensorsPerVoxel*p+k;
					S.addEdge(sequentialIndex[tp],sequentialIndex[tp],1);
				}
				//-------------------------
				for(int kn=0;kn<NUM_NEIGHBORS;++kn){//add links to neighboring compartments
					int ss=s+dSlice[kn];
					int rr=r+dRow[kn];
					int cc=c+dCol[kn];
					if((0<=rr) && (rr<nrows) && (0<=cc) && (cc<ncols) && (0<=ss) && (ss<nslices)){
						int neighPos=ss*nrows*ncols+rr*ncols+cc;
						int neighNc=voxels[neighPos].getNumCompartments();
						if(neighNc==0){
							continue;
						}
						double *neigh_volumeFractions=voxels[neighPos].getVolumeFractions();
						double neighLocation[3]={double(cc), double(nrows-1-rr), double(ss)};//			<<---Neighbor's Location
						double displacementDir[3]={location[0]-neighLocation[0], location[1]-neighLocation[1], location[2]-neighLocation[2]};

						double d2=SQR(location[0]-neighLocation[0])+SQR(location[1]-neighLocation[1])+SQR(location[2]-neighLocation[2]);
						//---do assignment---
						double *pddsA=new double[nc*3];
						double *pddsB=new double[neighNc*3];
						voxels[p].getPDDs(pddsA);
						voxels[neighPos].getPDDs(pddsB);
						int *assignment=new int[MAX(nc, neighNc)];
						assignTensors(pddsA, nc, pddsB, neighNc, displacementDir, assignment, hungarianSolver);
						for(int k=0;k<nc;++k){
							int tp=maxTensorsPerVoxel*p+k;//								<<---spatial-index
							int kk=assignment[k];
							if((kk<0) || (kk>=neighNc)){
								continue;
							}
							int neighTp=maxTensorsPerVoxel*neighPos+kk;//					<<---neighbor's spatial-index
							double angle=getAbsAngle(&pddsA[3*k],&pddsB[3*kk],3);
							if(angle<dTheta0){
								int currentSIndex=sequentialIndex[tp];
								int neighSIndex=sequentialIndex[neighTp];
								//double edgeWeight=d2/(d0*d0)+(1-cos(angle))/(1-cos(2*theta0));
								//double edgeWeight=d2/(d0*d0)+angle/(1-cos(2*theta0));
								double edgeWeight=d2/(d0*d0)+angle/theta0;
								//double edgeWeight=angle;
								//S.addEdge(currentSIndex, neighSIndex, exp(-edgeWeight/d0));
								//edgeWeight=volumeFractions[k]*neigh_volumeFractions[kk]*exp(-edgeWeight);


								edgeWeight=exp(-edgeWeight);
								//edgeWeight=d2;



								//--- symetry---
								if(neighSIndex<currentSIndex){
									S.addEdge(currentSIndex, neighSIndex, edgeWeight);
									S.addEdge(neighSIndex, currentSIndex, edgeWeight);
									/*int prev=S.retrieve(neighSIndex, currentSIndex);
									if(fabs(edgeWeight-prev)>1e-9){
										cerr<<"Asymetry: "<<neighSIndex<<", "<<currentSIndex<<": "<<prev<<", "<<edgeWeight<<endl;
									}*/
								}
							}
						}
						delete[] pddsA;
						delete[] pddsB;
						delete[] assignment;
						//-------------------
					}
				}//neighboring voxels
				//-----add links to random voxels----
				const int nrandom=0;
				for(int k=0;k<nc;++k){
					double pddA[3];
					voxels[p].getPDD(k, pddA);
					int tp=maxTensorsPerVoxel*p+k;
					srand(tp);
					for(int kn=0;kn<nrandom;++kn){
						int sel=rand()%numTensors;
						int rnsi=spatialIndex[sel];//random neighbor spatial index
						int rnvp=rnsi/maxTensorsPerVoxel;//random neighbor voxel position
						int rnci=rnsi%maxTensorsPerVoxel;//random neighbor compartment index
						int rnnc=voxels[rnvp].getNumCompartments();//random neighbor - number of compartments
						if(rnci>=rnnc){
							--kn;
							continue;
						}
						int ss=rnvp/(nrows*ncols);
						int rr=(rnvp%(nrows*ncols))/ncols;
						int cc=rnvp%ncols;
						double neighLocation[3]={double(cc), double(nrows-1-rr), double(ss)};//			<<---Neighbor's Location
						double displacementDir[3]={location[0]-neighLocation[0], location[1]-neighLocation[1], location[2]-neighLocation[2]};
						double d2=SQR(location[0]-neighLocation[0])+SQR(location[1]-neighLocation[1])+SQR(location[2]-neighLocation[2]);
						
						double pddB[3];
						voxels[rnvp].getPDD(rnci, pddB);
						double angle=getAbsAngle(pddA,pddB,3);
						if(angle>=dTheta0){
							--kn;
							continue;
						}
						double edgeWeight=d2/(d0*d0)+angle/theta0;
						edgeWeight=exp(-edgeWeight);
						int retVal=S.addEdge(sequentialIndex[tp],sel,edgeWeight);
						if(retVal==-1){
							--kn;
							continue;
						}
						retVal=S.addEdge(sel,sequentialIndex[tp],edgeWeight);
						if(retVal==-1){
							--kn;
							continue;
						}
					}
				}
				
				
			}
		}
	}
	double *d=new double[numTensors];
	double *dr=new double[numTensors];
	S.sumRowAbsValues(d);
	S.sumRowValues(dr);

	for(int i=0;i<numTensors;++i){
		dr[i]=0.5*(d[i]-dr[i])+offset;
		d[i]+=2*offset;
	}
	//S.sumToDiagonal(dr);

	for(int i=0;i<numTensors;++i){
		d[i]=1.0/sqrt(d[i]);
	}
	S.multDiagLeftRight(d,d);
	delete[] d;
	delete[] dr;
	return 0;
}




int MultiTensorField::getMaxCompartments(void){
	int maxCompartments=-1;
	int p=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				int op=voxels[p].getNumCompartments();
				if(maxCompartments<op){
					maxCompartments=op;
				}
			}
		}
	}
	return maxCompartments;
}

int MultiTensorField::getMaxCompartments(set<int> &slices){
	int maxCompartments=-1;
	int p=0;
	for(set<int>::iterator it=slices.begin(); it!=slices.end();++it){
		int s=*it;
		if((s<0) || (s>=nslices)){
			continue;
		}
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				int op=voxels[p].getNumCompartments();
				if(maxCompartments<op){
					maxCompartments=op;
				}
			}
		}
	}
	return maxCompartments;
}

int MultiTensorField::getTotalCompartments(void){
	int totalCompartments=0;
	int p=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				totalCompartments+=voxels[p].getNumCompartments();
			}
		}
	}
	return totalCompartments;
}

int MultiTensorField::getTotalCompartments(set<int> &slices){
	int totalCompartments=0;
	int p=0;
	for(set<int>::iterator it=slices.begin(); it!=slices.end();++it){
		int s=*it;
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				totalCompartments+=voxels[p].getNumCompartments();
			}
		}
	}
	return totalCompartments;
}

void MultiTensorField::loadSamplingPoints(const char *fname){
	int *s0Indices=NULL;
	int numS0;
#ifdef ADD_DUPLICATES
	double *halfSamplings;
	loadOrientations(fname,halfSamplings, numSamplingPoints, s0Indices, numS0);
	samplingPoints=new double[2*3*numSamplingPoints];
	memcpy(samplingPoints, halfSamplings, sizeof(double)*3*numSamplingPoints);
	memcpy(&samplingPoints[3*numSamplingPoints], halfSamplings, sizeof(double)*3*numSamplingPoints);
	for(int i=3*numSamplingPoints;i<6*numSamplingPoints;++i){
		samplingPoints[i]*=-1;
	}
	numSamplingPoints*=2;
	delete[] halfSamplings;
#else
	loadOrientations(fname,samplingPoints, numSamplingPoints, s0Indices, numS0);
#endif
	delete[] s0Indices;
	
}

double *MultiTensorField::getSamplingPoints(void){
	return samplingPoints;
}

int MultiTensorField::getNumSamplingPoints(void){
	return numSamplingPoints;
}



double computeBestVectorAssignment(double A[3][3], double B[3][3], double voidCost, int *assign){
	const unsigned char assignments[6][3]={{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}};
	double C[3][3];
	for(int ii=0;ii<3;++ii){
		bool isVoid_ii=(sqrNorm(A[ii],3)<1e-9);
		for(int jj=0;jj<3;++jj){
			bool isVoid_jj=(sqrNorm(B[jj],3)<1e-9);
			if(isVoid_ii&&isVoid_jj){
				C[ii][jj]=0;
			}else if(isVoid_ii||isVoid_jj){
				C[ii][jj]=voidCost;
			}else{
				C[ii][jj]=getAbsAngleDegrees(A[ii], B[jj],3);
			}
		}
	}
	double costs[6];
	int best=0;
	double bestCost=C[0][assignments[0][0]]+C[1][assignments[0][1]]+C[2][assignments[0][2]];
	costs[0]=bestCost;
	for(int i=1;i<6;++i){
		double opc=C[0][assignments[i][0]]+C[1][assignments[i][1]]+C[2][assignments[i][2]];
		costs[i]=opc;
		if(opc<bestCost){
			best=i;
			bestCost=opc;
		}
	}
	for(int k=0;k<3;++k){
		assign[k]=assignments[best][k];
	}
	return bestCost;
}

int MultiTensorField::computeLocalCoherenceIndex(double voidCost, double *LCI){
	const int NUM_NEIGHBORS=26;
	const int dRow[]={  0, -1, 0, 1,     0, 0,    0, -1,  0,  1,  0, -1,  0,  1,    -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1};
	const int dCol[]={ -1,  0, 1, 0,     0, 0,   -1,  0,  1,  0, -1,  0,  1,  0,    -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1};
	const int dSlice[]={0,  0, 0, 0,    -1, 1,   -1, -1, -1, -1,  1,  1,  1,  1,     0,  0,  0,  0, -1, -1, -1, -1,  1,  1,  1,  1};
	int assign[3];
	int p=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++p){
				LCI[p]=0;
				if(voxels[p].getNumCompartments()<1){
					LCI[p]=QNAN64;
					continue;
				}
				double A[3][3];
				memset(A, 0, sizeof(A));
				int ncp=MIN(voxels[p].getNumCompartments(), 3);
				for(int kk=0;kk<ncp;++kk){
					voxels[p].getPDD(kk, A[kk]);
				}

				int nTerms=0;
				for(int k=0;k<NUM_NEIGHBORS;++k){
					int ss=s+dSlice[k];
					if(!IN_RANGE(ss, 0, nslices)){
						continue;
					}
					int rr=r+dRow[k];
					if(!IN_RANGE(rr, 0, nrows)){
						continue;
					}
					int cc=c+dCol[k];
					if(!IN_RANGE(cc, 0, ncols)){
						continue;
					}
					int q=ss*nrows*ncols+rr*ncols+cc;
					if(voxels[q].getNumCompartments()<1){
						continue;
					}
					double B[3][3];
					memset(B, 0, sizeof(B));
					int ncq=MIN(voxels[q].getNumCompartments(), 3);
					for(int kk=0;kk<ncq;++kk){
						voxels[q].getPDD(kk, B[kk]);
					}
					//------------------------------
					computeBestVectorAssignment(A, B, voidCost, assign);
					for(int kk=0;kk<ncp;++kk){
						if(assign[kk]>=ncq){
							continue;
						}
						double angle=getAbsAngleDegrees(A[kk], B[assign[kk]], 3);
						LCI[p]+=angle;
						++nTerms;
					}
				}
				if(nTerms>0){
					LCI[p]/=nTerms;
				}
			}
		}
	}
	return 0;
}


#endif

