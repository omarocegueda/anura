#ifdef USE_GSL
#include "CSDeconv.h"
#include "linearAlgebra.h"
#include "SphericalHarmonics.h"
#include "ls.h"
#include "GDTI.h"
#include "geometryutils.h"
#include "SphericalHarmonics.h"
#include "histogram.h"
#include "icp.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "SHEvaluator.h"
#include "bits.h"
using namespace std;
CSDeconv::CSDeconv(){
	fconv=NULL;
	rconv=NULL;
	HR_trans=NULL;
	F=NULL;
	init_F=NULL;
	S=NULL;
	S_padded=NULL;
	HR_amps=NULL;
	neg=NULL;
}

CSDeconv::~CSDeconv(){
}


void CSDeconv::dellocate(void){
	DELETE_ARRAY(fconv);
	DELETE_ARRAY(rconv);
	DELETE_ARRAY(HR_trans);
	DELETE_ARRAY(F);
	DELETE_ARRAY(F_ant);
	DELETE_ARRAY(init_F);
	DELETE_ARRAY(S);
	DELETE_ARRAY(S_padded);
}

void CSDeconv::init(const std::vector<double> &responseCoefs, const std::vector<double> &initFilter, double *directions, int _ndirs, double *constraintDirs, int _nConstraints, int lmax, double _threshold, double _lambda){
	thresholdFraction=_threshold;
	lambda=_lambda;
	nConstraints=_nConstraints;
	ngrads = _ndirs;
	lmax_data = (responseCoefs.size()-1)*2;
	int n=SphericalHarmonics::LforN (ngrads);
	if(lmax_data>n){
		lmax_data = n;
	}
	if(lmax_data>lmax){
		lmax_data = lmax;
	}
	if(initFilter.size() < (1+lmax_data/2)){
		cerr<<"Not enough initial filter coefficients supplied for lmax = "<<lmax_data<<endl;
		return;
	}
	
	std::vector<double> RH;
	SphericalHarmonics::SH2RH(responseCoefs, RH);
	nharm=SphericalHarmonics::NforL(lmax_data);
	fconv=new double[nharm*_ndirs];
	SphericalHarmonics::initTransformation(directions, _ndirs, lmax_data, fconv);
	rconv=new double[_ndirs*nharm];
	int retval=computePseudoInverseRMO(fconv, _ndirs, nharm, rconv);
	

	//---------apply filter------
	int l=0;
	int nl = 1;
	for(int row=0;row<nharm;++row){
		if(row>=nl){
			l++; 
			nl=SphericalHarmonics::NforL(2*l); 
		}
		for(int col=0;col<ngrads;++col){
            rconv[row*ngrads+col]*=initFilter[l] / RH[l];
            fconv[col*nharm+row]*=RH[l];
		}
	}
	//---------
	nHarmConstraints=SphericalHarmonics::NforL(lmax);
	HR_trans=new double[nConstraints*nHarmConstraints];
	SphericalHarmonics::initTransformation(constraintDirs, nConstraints, lmax, HR_trans);
	double factor=((double) ngrads)*responseCoefs[0]/((double) nConstraints);
	multVectorScalar(HR_trans, factor, nConstraints*nHarmConstraints, HR_trans);

	//F=new double[nHarmConstraints];
	//F_ant=new double[nHarmConstraints];
	F=new double[_ndirs+nConstraints];
	F_ant=new double[_ndirs+nConstraints];
	init_F=new double[nharm];
	//S=new double[_ndirs];
	S=new double[_ndirs+nConstraints];
	S_padded=new double[_ndirs+nConstraints];
	HR_amps=new double[nConstraints];
	neg=new int[nConstraints];

	lmax_constraints=lmax;
}

void CSDeconv::setInputSignal(double s0, double *_S, int len){
	memset(S_padded, 0, sizeof(double)*(ngrads+nConstraints));
	double *tmpMatrix=new double[ngrads*nharm];
	for(int i=0;i<len;++i){
		S[i]=S_padded[i]=_S[i]/s0;
	}
	multMatrixVector(rconv, S_padded, nharm, ngrads, init_F);
	//solveSubsetLeastSquares(fconv, ngrads, nharm, S_padded, NULL, nharm, tmpMatrix, init_F);
	memset(F, 0, sizeof(double)*(ngrads+nConstraints));
	memcpy(F, init_F, sizeof(double)*nharm);
	multMatrixVector(HR_trans, F, nConstraints, nHarmConstraints, HR_amps);
	double hramps_mean=0;
	for(int i=0;i<nConstraints;++i){
		hramps_mean+=HR_amps[i];
	}
	hramps_mean/=nConstraints;
	thresholdValue=thresholdFraction*hramps_mean;
	delete[] tmpMatrix;
}

void CSDeconv::readConstraintDirections(const char *fname, double *&dirs, int &n){
	FILE *F=fopen(fname, "r");
	fscanf(F, "%d", &n);
	dirs=new double[2*n];
	for(int i=0;i<n;++i){
		fscanf(F, "%lf%lf", &dirs[2*i], &dirs[2*i+1]);
	}
	fclose(F);
}

void CSDeconv::readGradients(const char *fname, double *&dirs, double *&bs, int &n){
	ifstream F(fname);
	string nextLine;
	getline(F, nextLine);
	vector<double> v;
	vector<double> bv;
	while(!(nextLine.empty())){
		istringstream is(nextLine);
		double x,y,z,b;
		if(is>>x>>y>>z>>b){
			if(b>0){
				double norm=sqrt(SQR(x)+SQR(y)+SQR(z));
				double azimuth=atan2(y, x);
				double elevation=acos(z/norm);
				v.push_back(azimuth);
				v.push_back(elevation);
				bv.push_back(b);
			}
		}
		getline(F, nextLine);
	}
	n=v.size()/2;
	dirs=new double[2*n];
	bs=new double[n];
	memcpy(dirs, &v[0],sizeof(double)*2*n);
	memcpy(bs, &bv[0],sizeof(double)*n);
	F.close();
}

bool CSDeconv::iterate(void){
	multMatrixVector(HR_trans, F, nConstraints, nHarmConstraints, HR_amps);
	int negCount=0;
	for(int i=0;i<nConstraints;++i){
		if(HR_amps[i]<thresholdValue){
			neg[negCount]=i;
			++negCount;
		}
	}
	if(negCount+ngrads<nHarmConstraints){
		cerr<<"not enough negative directions! failed to converge."<<endl;
		for(int i=0;i<nharm;++i){
			F[i]=NAN;
		}
		return true;
	}
	//--build design matrix--
	int nrows=ngrads+negCount;
	int ncols=nHarmConstraints;
	double *M=new double[nrows*ncols];
	memset(M,0,sizeof(double)*nrows*ncols);
	for(int i=0;i<ngrads;++i){
		for(int j=0;j<nharm;++j){
			//M[i*ncols+j]=fconv[i*nharm+j];
			M[j*nrows+i]=fconv[i*nharm+j];
		}
	}

	for(int i=0;i<negCount;++i){
		for(int j=0;j<nHarmConstraints;++j){
			//M[(ngrads+i)*ncols+j]=lambda*HR_trans[neg[i]*nharm+j];
			M[j*nrows+(ngrads+i)]=lambda*HR_trans[neg[i]*nHarmConstraints+j];
		}
	}
	memcpy(F_ant, F, sizeof(double)*nHarmConstraints);
	solveLeastSquares(M, nrows, ncols, S_padded, F);
	double error=0;
	for(int i=0;i<nHarmConstraints;++i){
		error+=SQR(F[i]-F_ant[i]);
	}
	//error=sqrt(error);
	delete[] M;
	return error<1e-10;
}

int CSDeconv::estimateResponseFunction(double *s0Volume, double *dwVolume, int nslices, int nrows, int ncols, unsigned char *mask, unsigned char filterMask, unsigned char filterVal, double *pddField, double *gradients, int numGradients, int lmax, double *coefs){
	int nvoxels=nslices*nrows*ncols;
	double *signal=new double[numGradients];
	double vertical[]={0,0,1};
	double rotation[9];
	double *rotatedGradients=new double[3*numGradients];
	double *rotatedDirections=new double[2*numGradients];
	int nr_transform=numGradients;
	int nc_transform=SphericalHarmonics::NforL(lmax);
	double *sht				=new double[nr_transform*nc_transform];
	double *sht_transpose	=new double[nr_transform*nc_transform];
	double *local_coefs=new double[MAX(nc_transform, nr_transform)];
	
	int pos=0;
	int cnt=0;
	memset(coefs, 0, sizeof(double)*nc_transform);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c, ++pos){
				if(((mask[pos])&filterMask)!=filterVal){
					continue;
				}
				
				++cnt;
				double *pdd=&pddField[3*pos];
				
				fromToRotation(pdd, vertical, rotation);

				//------
				double test[3];
				multMatrixVector<double>(rotation, pdd, 3,3, test);
				//------
				
				for(int k=0;k<numGradients;++k){
					multMatrixVector<double>(rotation, &gradients[3*k], 3,3, &rotatedGradients[3*k]);
				}
				
				SphericalHarmonics::getAzimuthElevationPairs(rotatedGradients, numGradients, rotatedDirections);
				
				SphericalHarmonics::initTransformation(rotatedDirections, numGradients, lmax, sht);
				
				memcpy(signal, &dwVolume[pos*numGradients],sizeof(double)*numGradients);
				
				for(int i=0;i<numGradients;++i){
					signal[i]/=s0Volume[pos];
				}
				
				solveSubsetLeastSquares(sht, nr_transform, nc_transform, signal, NULL, nc_transform, sht_transpose, local_coefs);
				
				for(int i=0;i<nc_transform;++i){
					coefs[i]+=local_coefs[i];
					if(!isNumber(coefs[i])){
						coefs[i]=coefs[i];
					}
				}
				
			}
		}
	}
	if(cnt>0){
		for(int i=0;i<nc_transform;++i){
			coefs[i]/=cnt;
		}
	}
	
	delete[] rotatedGradients;
	delete[] signal;
	delete[] sht;
	delete[] sht_transpose;
	delete[] local_coefs;
	return cnt;

}


double *CSDeconv::getSHCoefficients(void){
	return F;
}



void CSDeconv::denoiseDWVolume_align(double *s0Volume, double *dwVolume, int nrows, int ncols, int nslices, int numGradients, double *gradientOrientations, double *&aligningErrors, int &nSamples, int &sampleLength){
	const int lmax=8;//FIXME: pass as parameter
	//-----------
	int nvoxels=nrows*ncols*nslices;
	double *denoised=new double[nvoxels*numGradients];
	double *signalList=new double[27*3*2*numGradients];//neighSize*3D*2copies(symmetric)*directions
	vector<double *> vAE;
	double ae[27];
	memcpy(denoised, dwVolume, sizeof(double)*nvoxels*numGradients);
	int nc_transform=SphericalHarmonics::NforL(lmax);
	double *sht				=new double[numGradients*nc_transform];
	double *sht_transpose	=new double[numGradients*nc_transform];
	double *rotatedDirections=new double[2*numGradients];
	double *local_coefs=new double[MAX(nc_transform, numGradients)];
	double *averaged_coefs=new double[MAX(nc_transform, numGradients)];
	double *amplitudes=new double[numGradients];

	SHEvaluator evaluator(gradientOrientations, numGradients, lmax);

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
				//----compute threshold to select well aligned shapes
				double aecp[27];
				memcpy(aecp, ae, sizeof(double)*nSignals);
				sort(aecp, aecp+nSignals);
				double thr=aecp[nSignals/2];
				//----Fit Spherical Harmonics coefficients to well aligned shapes----
				
				int nAveraged=0;
				memset(averaged_coefs, 0, sizeof(double)*nc_transform);
				double *currentSignal=signalList;
				for(int i=0;i<nSignals;++i, currentSignal+=3*numGradients*2){
					if((i>0) && (ae[i-1]>thr)){
						continue;
					}
					SphericalHarmonics::getAmplitudeAndAzimuthElevationPairs(currentSignal, numGradients, rotatedDirections, amplitudes);
					SphericalHarmonics::initTransformation(rotatedDirections, numGradients, lmax, sht);
					solveSubsetLeastSquares(sht, numGradients, nc_transform, amplitudes, NULL, nc_transform, sht_transpose, local_coefs);
					for(int j=0;j<nc_transform;++j){
						averaged_coefs[j]+=local_coefs[j];
					}
					++nAveraged;
				}
				for(int j=0;j<nc_transform;++j){
					averaged_coefs[j]/=nAveraged;
				}
				evaluator.evaluateFunction_amplitudes(averaged_coefs, nc_transform, lmax, &denoised[v*numGradients]);
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
	delete[] rotatedDirections;
	delete[] sht;
	delete[] sht_transpose;
	delete[] amplitudes;
	delete[] local_coefs;
	delete[] averaged_coefs;
}
#endif
