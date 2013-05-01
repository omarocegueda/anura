#include "dtiutils.h"
#include "macros.h"
#include "nifti1_io.h"
#include "utilities.h"
#include <iostream>
#include <algorithm>
#include "geometryutils.h"
#include "linearalgebra.h"
#include "Tensor.h"
#include "expm.h"

using namespace std;
#define EPS_ORIENTATION_INTEGRITY 1e-5
/*#define INF_HC 1e10
void groupCoefficientsHC(GDTI &H, double *alpha, double *DBFDirections, double *Phi, int numDBFDirections, double b, double *diffusivities, vector<pair<pair<int, int>, double > > &F, MultiTensor &result){
	set<pair<int, double *> > nonZero;
	vector<int> alphaIndex;
	for(int i=0;i<numDBFDirections;++i){
		if(alpha[i]>0){
			nonZero.insert(make_pair(i, (double*)(NULL)));
			alphaIndex.push_back(i);
		}
	}
	int N=nonZero.size();
	int numGradients=H.getNumGradients();
	double *gradients=H.getGradients();
	double *S=new double[N*numGradients];
	double *coeff=new double[N];
	double *directions=new double[3*N];
	for(int i=0;i<numGradients;++i){//copy the DB Functions
		for(int j=0;j<N;++j){
			S[j*numGradients+i]=Phi[i*numDBFDirections+alphaIndex[j]];
		}
	}
	int pos=0;
	for(set<pair<int, double *> >::iterator it=nonZero.begin();it!=nonZero.end();++it, ++pos){
		coeff[pos]=alpha[it->first];
		memcpy(&directions[3*pos], &DBFDirections[3*(it->first)], sizeof(double)*3);
		it->second=&S[pos*numGradients];//this preserves set ordering
		
		it->first=pos;//this preserves set ordering
	}
	double *distance=new double[N*N];
	double *mixedSignal=new double[numGradients];
	//compute initial distance matrix
	for(int i=0;i<N;++i){
		distance[i*N+i]=0;
		for(int j=i+1;j<N;++j){
			double sumCoeff=coeff[i]+coeff[j];
			linCombVector<double>(&S[i*numGradients], coeff[i]/sumCoeff, &S[j*numGradients], coeff[j]/sumCoeff, numGradients,mixedSignal);
			Tensor T;
			T.fitFromSignal(1.0, mixedSignal, H, b);
			double newPdd[3];
			T.getPDD(newPdd);
			double dist_i=getAbsAngleDegrees(newPdd, &directions[3*i], 3);
			double dist_j=getAbsAngleDegrees(newPdd, &directions[3*j], 3);
			double dist=MIN(dist_i, dist_j);
			distance[i*N+j]=distance[j*N+i]=dist;
		}
	}
	//start HC
	int numClusters=N;
	F.clear();
	while(numClusters>2){
		double minDist=INF_HC;
		pair<pair<int, int>, double > best;
		//----select a pair of cluster for joining----
		for(int i=0;i<N;++i){
			for(int j=i+1;j<N;++j){
				if(distance[i*N+j]<minDist){
					minDist=distance[i*N+j];
					best.first.first=i;
					best.first.second=j;
					best.second=minDist;
				}
			}
		}
		//-----merge-------
		F.push_back(make_pair(make_pair(best.first.first, best.first.second), best.second));
		int ii=best.first.first;
		int jj=best.first.second;
		double sumCoeff=coeff[ii]+coeff[jj];
		linCombVector<double>(&S[ii*numGradients], coeff[ii]/sumCoeff, &S[jj*numGradients], coeff[jj]/sumCoeff, numGradients,mixedSignal);
		Tensor T;
		T.fitFromSignal(1.0, mixedSignal, H, b);
		double newPdd[3];
		T.getPDD(newPdd);
		//-----update-----
		T.acquireWithScheme(b,gradients, numGradients, 0, &S[ii*numGradients]);//signal
		coeff[ii]+=coeff[jj];
		coeff[jj]=0;
		memcpy(&directions[3*ii], newPdd, sizeof(double)*3);//pdd
		for(int i=0;i<N;++i){
			distance[jj*N+i]=INF_HC;
			distance[i*N+jj]=INF_HC;//distances to eliminated cluster
			if((i!=ii) && (coeff[i]>0)){//distance to new cluster
				double sumCoeff=coeff[i]+coeff[ii];
				linCombVector<double>(&S[i*numGradients], coeff[i]/sumCoeff, &S[ii*numGradients], coeff[ii]/sumCoeff, numGradients,mixedSignal);
				Tensor T;
				T.fitFromSignal(1.0, mixedSignal, H, b);
				double newPdd[3];
				T.getPDD(newPdd);
				double dist_i=getAbsAngleDegrees(newPdd, &directions[3*i], 3);
				double dist_ii=getAbsAngleDegrees(newPdd, &directions[3*ii], 3);
				double dist=MIN(dist_i, dist_ii);
				distance[i*N+ii]=distance[ii*N+i]=dist;
			}
		}
		--numClusters;
		//--------------------------------------------
	}

	result.allocate(numClusters);
	pos=0;
	for(int i=0;i<numClusters;++i){
		while((pos<N) && coeff[pos]==0){
			++pos;
		}
		result.setRotationMatrixFromPDD(i,&directions[3*pos]);
		result.setDiffusivities(i,diffusivities);
		result.setVolumeFraction(i,coeff[pos]);
	}
	delete[] coeff;
	delete[] mixedSignal;
	delete[] S;
}
*/
void showOrientationHistogram(double *H, int *groups, int n, double *directions, int rows, int cols, cv::Mat &M){
	double maxVal=getMaxVal(H, n);
	int w=(cols+n-1)/n;
	
	M.create(rows, n*w, CV_8UC3);
	unsigned char *M_data=(unsigned char*)M.data;
	memset(M_data, -1, sizeof(unsigned char)*3*rows*n*w);
	bool showClusters=false;
	if(groups!=NULL){
		showClusters=true;
		for(int k=0;k<n;++k)if(groups[k]<0){
			showClusters=false;
			break;
		}
	}

	
	for(int k=0;k<n;++k)if(H[k]>0){
		cv::Point frameA=cv::Point(k*w, rows-1);
		cv::Point frameB=cv::Point((k+1)*w-1, rows-1-int(rows*double(H[k])/double(maxVal)));
		cv::Point A=cv::Point(k*w+1, rows-2);
		cv::Point B=cv::Point((k+1)*w-2, rows-int(rows*double(H[k])/double(maxVal)));
		

		cv::Scalar color;
		cv::Scalar gray(128,128,128);
		if(showClusters){//first priority: groups
			unsigned char r=(groups[k]&(1<<2))?192:64;
			unsigned char g=(groups[k]&(1<<1))?192:64;
			unsigned char b=(groups[k]&(1<<0))?192:64;
			color=cv::Scalar(b,g,r);
		}else if(directions!=NULL){//next priority: orientation
			double *dir=&directions[3*k];
			color=cv::Scalar(255*fabs(dir[2]),255*fabs(dir[1]),255*fabs(dir[0]));
		}else{//no info for coloring
			color=cv::Scalar(192,0,0);
		}
		cv::rectangle(M, frameA, frameB, gray,CV_FILLED);
		cv::rectangle(M, A, B, color,CV_FILLED);
	}
}
/*void showOrientationSimilarityMatrix(double *H, int n, double *directions, GDTI &gdti, double transDiffusion, double longDiffusion, int rows, int cols, cv::Mat &M, cv::Mat &expM){
	int w=(cols+n-1)/n;
	int h=(rows+n-1)/n;
	rows=h*n;
	cols=w*n;
	M.create(rows, cols, CV_8UC3);
	expM.create(rows, cols, CV_8UC3);
	double *simmatrix=new double[n*n];
	double *exp_simmatrix=new double[n*n];
	int pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j,++pos){
			//double prod=dotProduct(&directions[3*i], &directions[3*j], 3);
			//simmatrix[pos]=fabs(prod);
			double sumCoeff=H[i]+H[j];
			MultiTensor P;
			P.allocate(2);
			P.setRotationMatrixFromPDD(0,&directions[3*i]);
			P.setDiffusivities(0,transDiffusion, transDiffusion,longDiffusion);
			P.setVolumeFraction(0,H[i]/sumCoeff);

			P.setRotationMatrixFromPDD(1,&directions[3*j]);
			P.setDiffusivities(1,transDiffusion, transDiffusion,longDiffusion);
			P.setVolumeFraction(0,H[j]/sumCoeff);
			P.recomputeBestR2Tensor(gdti);
			double newPdd[3];
			P.getPDD(0, newPdd);
			double sim_i=fabs(dotProduct(newPdd, &directions[3*i], 3));
			double sim_j=fabs(dotProduct(newPdd, &directions[3*j], 3));
			double sim=MAX(sim_i, sim_j);
			simmatrix[i*n+j]=sim;
		}
	}
	normalizeDinamicRange(simmatrix, n*n);
	for(int i=0;i<n;++i){
		simmatrix[i*(n+1)]=0;
	}
	//saveMatrix(simmatrix, n,n,"simmatrix.txt");
	expm(simmatrix, n, exp_simmatrix);
	//saveMatrix(exp_simmatrix, n,n,"exp_simmatrix.txt");
	for(int i=0;i<n;++i){
		exp_simmatrix[i*(n+1)]-=1.0;
	}
	for(int i=n*n-1;i>=0;--i){
		exp_simmatrix[i]-=simmatrix[i];
	}

	normalizeDinamicRange(exp_simmatrix, n*n);
	pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j, ++pos){
			cv::Point A=cv::Point(j*w, i*h);
			cv::Point B=cv::Point((j+1)*w-1, (i+1)*h-1);
			if(i==j){
				cv::Scalar color=cv::Scalar(255,255,255);
				cv::rectangle(M, A, B, color,CV_FILLED);
			}else{
				unsigned char r,g,b;
				getIntensityColor<unsigned char>(int(255*simmatrix[pos]), r,g,b);
				cv::Scalar color=cv::Scalar(b,g,r);
				cv::rectangle(M, A, B, color,CV_FILLED);
			}
			unsigned char r,g,b;
			getIntensityColor<unsigned char>(int(255*exp_simmatrix[pos]), r,g,b);
			cv::Scalar color=cv::Scalar(b,g,r);
			cv::rectangle(expM, A, B, color,CV_FILLED);
		}
	}
	delete[] simmatrix;
	delete[] exp_simmatrix;
}
*/

void showOrientationSimilarityMatrix(double *H, int n, double *directions, int rows, int cols, cv::Mat &M, cv::Mat &expM){
	int w=(cols+n-1)/n;
	int h=(rows+n-1)/n;
	rows=h*n;
	cols=w*n;
	M.create(rows, cols, CV_8UC3);
	expM.create(rows, cols, CV_8UC3);
	double *simmatrix=new double[n*n];
	double *exp_simmatrix=new double[n*n];
	int pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j,++pos){
			double prod=dotProduct(&directions[3*i], &directions[3*j], 3);
			simmatrix[pos]=fabs(prod);
		}
	}
	normalizeDinamicRange(simmatrix, n*n);
	for(int i=0;i<n;++i){
		simmatrix[i*(n+1)]=0;
	}
	//saveMatrix(simmatrix, n,n,"simmatrix.txt");
	expm(simmatrix, n, exp_simmatrix);
	//saveMatrix(exp_simmatrix, n,n,"exp_simmatrix.txt");
	for(int i=0;i<n;++i){
		exp_simmatrix[i*(n+1)]-=1.0;
	}
	for(int i=n*n-1;i>=0;--i){
		exp_simmatrix[i]-=simmatrix[i];
	}

	normalizeDinamicRange(exp_simmatrix, n*n);
	pos=0;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j, ++pos){
			cv::Point A=cv::Point(j*w, i*h);
			cv::Point B=cv::Point((j+1)*w-1, (i+1)*h-1);
			if(i==j){
				cv::Scalar color=cv::Scalar(255,255,255);
				cv::rectangle(M, A, B, color,CV_FILLED);
			}else{
				unsigned char r,g,b;
				getIntensityColor<unsigned char>(int(255*simmatrix[pos]), r,g,b);
				cv::Scalar color=cv::Scalar(b,g,r);
				cv::rectangle(M, A, B, color,CV_FILLED);
			}
			unsigned char r,g,b;
			getIntensityColor<unsigned char>(int(255*exp_simmatrix[pos]), r,g,b);
			cv::Scalar color=cv::Scalar(b,g,r);
			cv::rectangle(expM, A, B, color,CV_FILLED);
		}
	}
	delete[] simmatrix;
	delete[] exp_simmatrix;
}
double computeFractionalAnisotropy(double *eigenValues, double meanDiffusivity){
	double factor=sqrt(3.0)/sqrt(2.0);
	double term0=SQR(eigenValues[0]-meanDiffusivity);
	double term1=SQR(eigenValues[1]-meanDiffusivity);
	double term2=SQR(eigenValues[2]-meanDiffusivity);
	double deno=sqrt(SQR(eigenValues[0])+SQR(eigenValues[1])+SQR(eigenValues[2]));
	double fa=factor*(sqrt(term0+term1+term2)/deno);
	return fa;
}

double computeFractionalAnisotropy(double *eigenValues){
	if(eigenValues==NULL){
		return 0;
	}
	double meanDiffusivity=(eigenValues[0]+eigenValues[1]+eigenValues[2])/3.0;
	double factor=sqrt(3.0)/sqrt(2.0);
	double term0=SQR(eigenValues[0]-meanDiffusivity);
	double term1=SQR(eigenValues[1]-meanDiffusivity);
	double term2=SQR(eigenValues[2]-meanDiffusivity);
	double deno=sqrt(SQR(eigenValues[0])+SQR(eigenValues[1])+SQR(eigenValues[2]));
	double fa=factor*(sqrt(term0+term1+term2)/deno);
	return fa;
}

double getMinimum(double *data, int n){
	double best=*data;
	for(double *current=data+n-1;current!=data;--current){
		if((*current)<best){
			best=*current;
		}
	}
	return best;
}

double getMaximum(double *data, int n){
	double best=*data;
	for(double *current=data+n-1;current!=data;--current){
		if((*current)>best){
			best=*current;
		}
	}
	return best;
}

double getMean(double *data, int n){
	double sum=*data;
	for(double *current=data+n-1;current!=data;--current){
		sum+=*current;
	}
	return sum/n;
}

double getStdev(double *data, int n){
	double sum=*data;
	double sqSum=sum*sum;
	for(double *current=data+n-1;current!=data;--current){
		sum+=*current;
		sqSum+=SQR(*current);
	}
	return sqrt(sqSum/n-SQR(sum/n));
}

double getMinimum(double *data, int n, unsigned char *mask){
	if(mask==NULL){
		return getMinimum(data, n);
	}
	unsigned char *binPos=mask+n;
	double *current=data+n;
	while((mask<=binPos)&&((*binPos)==0)){
		--binPos;
		--current;
	}
	double best=0;
	if(data<=current){
		best=*current;
		while(mask<=binPos){
			if(((*binPos)!=0) && (*current<best)){
				best=*current;
			}
			--mask;
			--current;
		}
	}
	return best;
}

double getMaximum(double *data, int n, unsigned char *mask){
	if(mask==NULL){
		return getMaximum(data, n);
	}
	unsigned char *binPos=mask+n;
	double *current=data+n;
	while((mask<=binPos)&&((*binPos)==0)){
		--binPos;
		--current;
	}
	double best=0;
	if(data<=current){
		best=*current;
		while(mask<=binPos){
			if(((*binPos)!=0) && (*current<best)){
				best=*current;
			}
			--mask;
			--current;
		}
	}
	return best;
}

double getMean(double *data, int n, unsigned char *mask){
	if(mask==NULL){
		return getMean(data, n);
	}
	unsigned char *binPos=mask+n;
	double *current=data+n;
	while((mask<=binPos)&&((*binPos)==0)){
		--binPos;
		--current;
	}
	double sum=0;
	int used=1;
	if(data<=current){
		used=0;
		while(mask<=binPos){
			if((*binPos)!=0){
				sum+=*current;
				++used;
			}
			--mask;
			--current;
		}
	}
	return sum/used;
}

double getStdev(double *data, int n, unsigned char *mask){
	if(mask==NULL){
		return getMean(data, n);
	}
	unsigned char *binPos=mask+n;
	double *current=data+n;
	while((mask<=binPos)&&((*binPos)==0)){
		--binPos;
		--current;
	}
	double sum=0;
	double sqSum=0;
	int used=1;
	if(data<=current){
		used=0;
		while(mask<=binPos){
			if((*binPos)!=0){
				sum+=*current;
				sqSum+=SQR(*current);
				++used;
			}
			--mask;
			--current;
		}
	}
	return sqrt(sqSum/used-SQR(sum/used));
}


int loadOrientations(const string &fname, double *&orientations, int &numOrientations, int *&s0Indices, int &numS0){
	FILE *F=fopen(fname.c_str(), "r");
    if(F==NULL){
        cerr<<"unable to open file "<<fname<<endl;
    }
	vector<double> v;
	int cc=0;
	vector<int> discarded;
	while(!feof(F)){
		double x,y,z;
		if(fscanf(F,"%lf%lf%lf", &x, &y, &z)==3){
			double nrm=sqrt(x*x+y*y+z*z);
			if((fabs(x)>EPS_ORIENTATION_INTEGRITY) || (fabs(y)>EPS_ORIENTATION_INTEGRITY) || (fabs(z)>EPS_ORIENTATION_INTEGRITY)){
				v.push_back(x/nrm);
				v.push_back(y/nrm);
				v.push_back(z/nrm);
			}else{
				discarded.push_back(cc);
			}
			++cc;
		}
	}
	numS0=discarded.size();
	s0Indices=new int[numS0];
	for(int i=0;i<numS0;++i){
		s0Indices[i]=discarded[i];
	}
	numOrientations=v.size()/3;
	//---verify integrity of orientations---
	int nRep=0;
	for(int i=0;i<numOrientations;++i){
		double sum=EPS_ORIENTATION_INTEGRITY+v[0]*v[3*i]+v[1]*v[3*i+1]+v[2]*v[3*i+2];
		if(sum>1.0){
			nRep++;
		}
	}
	if((numOrientations%nRep)!=0){
		cerr<<"Error: unable to group vectors: each orientation must have the same number of repetitions."<<endl;
		return -1;
	}
	int nVec=numOrientations/nRep;
	
	for(int i=nVec;i<numOrientations;++i){
		double sum=EPS_ORIENTATION_INTEGRITY+v[3*(i-nVec)]*v[3*i] + v[3*(i-nVec)+1]*v[3*i+1] + v[3*(i-nVec)+2]*v[3*i+2];
		if(sum<1.0){
			cerr<<"Error: there are repetitions of DW orientations but the entries are not the same or are not sorted."<<endl;
			return -1;
		}
	}
	//--------------------------------------
	orientations=new double[3*nVec];
	for(int i=0;i<3*nVec;++i){
		orientations[i]=v[i];
	}
	fclose(F);
	
	numOrientations=nVec;
	return 0;
}

void loadRandomPDDs(const string &fname, double *&pdds, int &n){
	ifstream F(fname.c_str());
	double x,y,z;
	vector<double> v;
	while(!(F.eof())){
		if(F>>x>>y>>z){
			v.push_back(x);
			v.push_back(y);
			v.push_back(z);
		}
	}
	n=v.size()/3;
	pdds=new double[v.size()];
	for(unsigned i=0;i<v.size();++i){
		pdds[i]=v[i];
	}
	F.close();
}

void loadDWMRIFiles(const vector<string> &names, int *s0Indices, int numS0, double *&S0Volume, double *&dwVolume, int &nr, int &nc, int &ns){
	int n=names.size();
	int currentS0=0;
	int currentOrientation=0;
	int numOrientations=(n-numS0)/numS0;
	int len=0;
	for(int i=0;i<n;++i){
		cerr<<"Loading file: "<<names[i]<<"...";
		nifti_image *nii=nifti_image_read(names[i].c_str(), 1);
		if(i==0){
			nc=nii->nx;
			nr=nii->ny;
			ns=nii->nz;
			len=nc*nr*ns;
			if(S0Volume==NULL){
				S0Volume=new double[nc*nr*ns];
			}
			memset(S0Volume, 0, sizeof(double)*nc*nr*ns);
			if(dwVolume==NULL){
				dwVolume=new double[numOrientations*nc*nr*ns];
			}
			memset(dwVolume,0, sizeof(double)*numOrientations*nc*nr*ns);
		}else{
			if((nc!=nii->nx) || (nr!=nii->ny) || (ns!=nii->nz)){
				cerr<<"Error: not all the .nii files have the same size."<<endl;
				return;
			}
		}
		float *nii_data=(float *)nii->data;
		
		if((currentS0<numS0) && (i==s0Indices[currentS0])){//current nii is a S0 image
			int pos=0;
			for(int s=0;s<ns;++s){
				for(int r=0;r<nr;++r){
					for(int c=0;c<nc;++c, ++pos){
						int posNifti=c+nc*(nr-1-r)+s*nr*nc;
						S0Volume[pos]+=nii_data[posNifti];
					}
				}
			}
			++currentS0;
		}else{
			int pos=0;
			for(int s=0;s<ns;++s){
				for(int r=0;r<nr;++r){
					for(int c=0;c<nc;++c, ++pos){
						int posNifti=c+nc*(nr-1-r)+s*nr*nc;
						dwVolume[pos*numOrientations+currentOrientation]+=nii_data[posNifti];
					}
				}
			}
			currentOrientation=(currentOrientation+1)%numOrientations;
		}
		nifti_image_free(nii);
		cerr<<"done."<<endl;
	}
	for(int i=0;i<len;++i){
		S0Volume[i]/=numS0;
	}
	for(int i=len*numOrientations-1;i>=0;--i){
		dwVolume[i]/=numS0;
	}
}

void loadVolumeFromNifti(std::string fname, double *&dwVolume, int &nrows, int &ncols, int &nslices){
	nifti_image *nii=nifti_image_read(fname.c_str(), 1);
	float *niiData=(float *)nii->data;
	nslices=nii->nz;
	nrows=nii->ny;
	ncols=nii->nx;
	int nvoxels=ncols*nrows*nslices;
	dwVolume=new double[nvoxels];
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				int posNifti=s*nrows*ncols+(nrows-1-r)*ncols+c;
				dwVolume[pos]=niiData[posNifti];
			}
		}
	}
	nifti_image_free(nii);
}

void load4DNifti(const string &fname, double *&data, int &nslices, int &nrows, int &ncols, int &len){
	nifti_image *nii=nifti_image_read(fname.c_str(), 1);
	float *niiData=(float *)nii->data;
	nslices=nii->nz;
	nrows=nii->ny;
	ncols=nii->nx;
	len=nii->nt;
	int nvoxels=ncols*nrows*nslices;
	data=new double[nvoxels*len];
	int pos=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				for(int t=0;t<len;++t,++pos){
					int posNifti=t*nslices*nrows*ncols + s*nrows*ncols + (nrows-1-r)*ncols + c;
					data[pos]=niiData[posNifti];
				}
			}
		}
	}
	nifti_image_free(nii);
}

void load4DNifti(const string &fname, float *&data, int &nslices, int &nrows, int &ncols, int &len){
	nifti_image *nii=nifti_image_read(fname.c_str(), 1);
	float *niiData=(float *)nii->data;
	nslices=nii->nz;
	nrows=nii->ny;
	ncols=nii->nx;
	len=nii->nt;
	int nvoxels=ncols*nrows*nslices;
	data=new float[nvoxels*len];
	int pos=0;
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				for(int t=0;t<len;++t,++pos){
					int posNifti=t*nslices*nrows*ncols + s*nrows*ncols + (nrows-1-r)*ncols + c;
					data[pos]=niiData[posNifti];
				}
			}
		}
	}
	nifti_image_free(nii);
}


void niiToPlain4D(const string &ifname, const string &ofname){
	nifti_image *nii=nifti_image_read(ifname.c_str(), 1);
	float *niiData=(float *)nii->data;
	int nslices=nii->nz;
	int nrows=nii->ny;
	int ncols=nii->nx;
	int len=nii->nt;
	int nvoxels=ncols*nrows*nslices;
	FILE *F=fopen(ofname.c_str(),"wb");
	fwrite(&nslices, sizeof(int),1,F);
	fwrite(&nrows, sizeof(int),1,F);
	fwrite(&ncols, sizeof(int),1,F);
	fwrite(&len, sizeof(int),1,F);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				for(int t=0;t<len;++t){
					int posNifti=t*nslices*nrows*ncols + s*nrows*ncols + (nrows-1-r)*ncols + c;
					double val=niiData[posNifti];
					fwrite(&val, sizeof(double),1,F);
				}
			}
		}
	}
	fclose(F);
	nifti_image_free(nii);
}


void loadDWMRIFromNifti(std::string fname, int *s0Indices, int numS0, double *&s0, double *&dwVolume, int &nrows, int &ncols, int &nslices, int &signalLength){
	nifti_image *nii=nifti_image_read(fname.c_str(), 1);
	nslices=nii->nz;
	nrows=nii->ny;
	ncols=nii->nx;
	signalLength=nii->nt-numS0;
	void *niiData=nii->data;
	if(signalLength<=0){
		cerr<<"Warning: no diffusion signals at loadDWMRIFromNifti."<<endl;
		nifti_image_free(nii);
		return;
	}
	int nvoxels=ncols*nrows*nslices;
	if(signalLength>0){
		dwVolume=new double[nvoxels*signalLength];
	}else{
		dwVolume=NULL;
	}
	if(numS0>0){
		s0=new double[nvoxels];
		memset(s0, 0, sizeof(double)*nvoxels);
	}else{
		s0=NULL;
	}
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				double *currentSignal=&dwVolume[signalLength*pos];
				int dwPosition=0;
				int s0Position=0;
				for(int k=0;k<nii->nt;++k){
					int posNifti=k*nslices*nrows*ncols+s*nrows*ncols+(nrows-1-r)*ncols+c;
					if((s0Position<numS0) && (k==s0Indices[s0Position])){
						if(nii->datatype==4){//NIFTI_TYPE_INT16
							s0[pos]+=((short*)niiData)[posNifti];
						}else{
							s0[pos]+=((float*)niiData)[posNifti];
						}

						++s0Position;
					}else{
						if(nii->datatype==4){//NIFTI_TYPE_INT16
							currentSignal[dwPosition]=((short*)niiData)[posNifti];
						}else{
							currentSignal[dwPosition]=((float*)niiData)[posNifti];
						}
						
						++dwPosition;
					}
					
				}
				if(numS0>0){
					s0[pos]/=numS0;
				}
			}
		}
	}
	nifti_image_free(nii);
}

void getDWSignalAtVoxel(double *S0, double *dwVolume, int nr, int nc, int ns, int numOrientations, int pr, int pc, int ps, double &s0, double *dwSignal){
	pr=nr-1-pr;
	s0=S0[ps*(nc*nr)+pr*(nc)+pc];
	int index=ps*(nc*nr)+pr*(nc)+pc;
	int nVoxels=nc*nr*ns;
	for(int i=0;i<numOrientations;++i){
		dwSignal[i]=dwVolume[index];
		index+=nVoxels;
	}
}

void getMaximumConnectedComponentMask(double *data, int nr, int nc, int ns, unsigned char *mask){
	int len=nr*nc*ns;
	unsigned char *binaryMask=new unsigned char[len];
	applyThreshold(data, len, 100, binaryMask);
	memset(mask, 0, sizeof(unsigned char)*len);
	unsigned current=1;
	int pos=0;
	int sel=-1;
	int maxSize=-1;
	for(int k=0;k<ns;++k){
		for(int i=nr-1;i>=0;--i){
			for(int j=0;j<nc;++j,++pos)if((binaryMask[pos]==1) && (mask[pos]==0)){
				int regionSize=floodFill<unsigned char>(binaryMask, nr, nc, ns, i, j, k, current, mask);
				if(maxSize<regionSize){
					maxSize=regionSize;
					sel=current;
				}
				++current;
			}
		}
	}
	for(int i=0;i<len;++i){
		if(mask[i]!=sel){
			mask[i]=0;
		}else{
			mask[i]=1;
		}
	}
	delete[] binaryMask;
}

void getMaximumConnectedComponent(unsigned char *binaryMask, int nr, int nc, int ns, unsigned char *mask){
	int len=nr*nc*ns;
	int *labels=new int[len];
	memset(labels, 0, sizeof(int)*len);
	int current=1;
	int pos=0;
	int sel=-1;
	int maxSize=-1;
	for(int k=0;k<ns;++k){
		for(int i=nr-1;i>=0;--i){
			for(int j=0;j<nc;++j,++pos)if((binaryMask[pos]!=0) && (labels[pos]==0)){
				int regionSize=floodFill<int>(binaryMask, nr, nc, ns, i, j, k, current, labels);
				if(maxSize<regionSize){
					maxSize=regionSize;
					sel=current;
				}
				++current;
			}
		}
	}
	for(int i=0;i<len;++i){
		if(labels[i]!=sel){
			mask[i]=0;
		}else{
			mask[i]=1;
		}
	}
	delete[] labels;
}

void buildNeighborhood(double *pdds, int npdds, int neighSize, vector<set<int> > &neighborhoods){
	pair<double, int> *v=new pair<double, int>[npdds];
	double *p=pdds;
	neighborhoods.resize(npdds);
	for(int i=0;i<npdds;++i, p+=3){
		double *q=pdds;
		for(int j=0;j<npdds;++j, q+=3)if(i!=j){
			double prod=fabs(dotProduct(p,q,3));
			v[j]=make_pair(prod, j);
		}
		sort(v,v+npdds);
		neighborhoods[i].clear();
		for(int j=0;neighborhoods[i].size()<neighSize;++j)if(v[npdds-1-j].second!=i){
			neighborhoods[i].insert(v[npdds-1-j].second);
		}
	}
	delete[] v;
}

void buildNeighborhood(double *pdds, int npdds, double maxAngle, std::vector<std::set<int> > &neighborhoods){
	double *p=pdds;
	neighborhoods.resize(npdds);
	for(int i=0;i<npdds;++i, p+=3){
		neighborhoods[i].clear();
		double *q=pdds;
		for(int j=0;j<npdds;++j, q+=3)if(i!=j){
			double angle=getAbsAngleDegrees(p,q,3);
			if(angle<=maxAngle){
				neighborhoods[i].insert(j);
			}
		}
	}
}

void groupCoefficients(double *alpha, double *diffusionDirections, int numDirections, vector<set<int> > &neighborhoods, double *RES_pdds, double *RES_amount, int &RES_count){
	vector<set<int> > groups;
	groupCoefficients(alpha, numDirections, 1e-2, neighborhoods, groups);
	RES_count=MIN(groups.size(), 3);
	double sumAmount=0;
	for(int i=0;i<RES_count;++i){
		double *centroid=&RES_pdds[3*i];
		double &amountDiff=RES_amount[i];
		computeCentroid(diffusionDirections, numDirections, alpha, groups[i], centroid, amountDiff);
		sumAmount+=amountDiff;
	}
	//memset(alpha, 0, sizeof(double)*numDirections);
	for(int i=0;i<RES_count;++i){
		RES_amount[i]/=sumAmount;
		//alpha[i]=RES_amount[i];
	}
}

void groupCoefficients(double *alpha, double *diffusionDirections, int numDirections, std::vector<std::set<int> > &neighborhoods, double bigPeaksThreshold, double transDiffusion, double longDiffusion, MultiTensor &result){
	int RES_count;
	//------group coefficients
	vector<set<int> > groups;
	groupCoefficients(alpha, numDirections, 1e-2, neighborhoods, groups);
	RES_count=MIN(groups.size(), 3);
	if(RES_count<=0){
		return;
	}
	double *RES_pdds=new double[3*RES_count];
	double *RES_amount=new double[RES_count];
	double sumAmount=0;
	for(int i=0;i<RES_count;++i){
		double *centroid=&RES_pdds[3*i];
		double &amountDiff=RES_amount[i];
		computeCentroid(diffusionDirections, numDirections, alpha, groups[i], centroid, amountDiff);
		sumAmount+=amountDiff;
	}
	for(int i=0;i<RES_count;++i){
		RES_amount[i]/=sumAmount;
	}
	//------
	getBigPeaks(bigPeaksThreshold, RES_pdds, RES_amount, RES_count);
	result.dellocate();
	result.allocate(RES_count);
	//---sort and assign coefficients and directions---
	result.setSortedCoefficientsAndDirections(alpha, diffusionDirections, numDirections);
	//result.setAlpha(alpha, numDirections);
	//-------------------------------------------------
	//result.setGroups(groups);
	for(int i=0;i<RES_count;++i){
		result.setDiffusivities(i, transDiffusion, transDiffusion, longDiffusion);
	}
	result.setVolumeFractions(RES_amount);
	for(int k=0;k<RES_count;++k){
		result.setRotationMatrixFromPDD(k,&RES_pdds[3*k]);
	}

	delete[] RES_pdds;
	delete[] RES_amount;
}

void createMultiTensorSingleComponent(double *S, GDTI &H, double *diffusionDirections, set<pair<double, int> > &group, MultiTensor &result){
	result.allocate(1);
	double tensor[6];
	double eVec[9];
	double eVal[3];
	H.solve(1,S, tensor);
	//----now force non-negative and get profile and pdd
	forceNonnegativeTensor(tensor,eVec, eVal);
	int maxIndex=getMaxIndex(eVal,3);
	result.setRotationMatrixFromPDD(0, &eVec[3*maxIndex]);
	sort(eVal, eVal+3);
	result.setDiffusivities(0,eVal);
	//---finaly compute the amount. We can simply sum the alpha-coefficients or do a weighted sum with respect to the dot product of each direction with the PDD
	double amountDiff=0;
	for(set<pair<double, int> >::iterator it=group.begin();it!=group.end();++it){
		//double absProd=fabs(dotProduct(&eVec[3*maxIndex], &diffusionDirections[3*(*it)],3));
		//amountDiff+=absProd*alpha[*it];
		amountDiff+=it->first;
	}
	result.setVolumeFraction(0,amountDiff);
}

void groupCoefficientsGDTI(GDTI &H, double longDiffusion, double transDiffusion, double *alpha, double *diffusionDirections, int numDirections, vector<set<int> > &neighborhoods, MultiTensor &result, vector<set<int> > &groups){
	int numGradients=H.getNumGradients();
	double b=H.get_b();
	double *gradients=H.getGradients();
	double tensor[6];
	double eVal[3];
	double eVec[9];
	//----quick test----
	/*double fixedDiffusivityProfile[3]={0.00022368980632455027, 0.00022368980632455027, 0.0012668930701427031};
	double fixedPDD0[3]={1, 0, 0};
	double fixedPDD1[3]={0, 1, 0};
	result.allocate(1);
	result.setDiffusivities(0,fixedDiffusivityProfile);
	result.setVolumeFraction(0, 0.5);
	result.setRotationMatrixFromPDD(0,fixedPDD0);
	double *S=new double[numGradients];
	result.acquireWithScheme(b,gradients,numGradients,0,S);
	H.solve(1,S,tensor);
	forceNonnegativeTensor(tensor,eVec, eVal);
	int maxIndex=getMaxIndex(eVal,3);
	sort(eVal, eVal+3);
	delete[] S;*/
	//------------------
	groupCoefficients(alpha, numDirections, 1e-2, neighborhoods, groups);
	
	while(groups.size()>3){
		groups.pop_back();
	}
	int numGroups=groups.size();
	if(numGroups==0){
		result.dellocate();
		return;
	}
	double sumAmount=0;
	
	double *synthetic=new double[numGradients];
	
	
	
	double *alpha_backup=result.getAlpha();
	int nAlpha_backup=result.getNumAlpha();
	result.setAlpha(NULL, 0);
	result.allocate(numGroups);
	result.setAlpha(alpha_backup, nAlpha_backup);
	result.setGroups(groups);

	for(int i=0;i<numGroups;++i){
		int groupNC=groups[i].size();
		MultiTensor current;
		current.allocate(groupNC);
		int k=0;
		double sumFractions=0;
		for(set<int>::iterator it=groups[i].begin();it!=groups[i].end();++it, ++k){
			current.setDiffusivities(k, transDiffusion, transDiffusion, longDiffusion);
			current.setRotationMatrixFromPDD(k, &diffusionDirections[3*(*it)]);
			sumFractions+=alpha[*it];
		}
		k=0;
		for(set<int>::iterator it=groups[i].begin();it!=groups[i].end();++it, ++k){
			current.setVolumeFraction(k,alpha[*it]/sumFractions);
		}
		
		current.acquireWithScheme(b, gradients, numGradients,0,synthetic);
		H.solve(1,synthetic, tensor);
		//----now force non-negative and get profile and pdd
		forceNonnegativeTensor(tensor,eVec, eVal);
		int maxIndex=getMaxIndex(eVal,3);
		result.setRotationMatrixFromPDD(i, &eVec[3*maxIndex]);
		sort(eVal, eVal+3);
		result.setDiffusivities(i,eVal);
		//---finaly compute the amount. We can simply sum the alpha-coefficients or do a weighted sum with respect to the dot product of each direction with the PDD
		double amountDiff=0;
		double sumProds=0;
		for(set<int>::iterator it=groups[i].begin();it!=groups[i].end();++it){
			double absProd=fabs(dotProduct(&eVec[3*maxIndex], &diffusionDirections[3*(*it)],3));
			//amountDiff+=absProd*alpha[*it];
			amountDiff+=alpha[*it];
			sumProds+=absProd;
		}
		//amountDiff/=sumProds;
		result.setVolumeFraction(i,amountDiff);
	}
	delete[] synthetic;
}


void computeDiffusionFunction(double *dir, double _lambdaMin, double _lambdaMiddle, double _lambdaLong, double b, double *gradients, int numGradients, double *phi, int idxInc){
	double e0[3]={1,0,0};
	double D[9]={
		_lambdaLong, 0, 0,
		0, _lambdaMiddle, 0,
		0, 0, _lambdaMin
	};
	double T[9];
	fromToRotation(e0, dir, T);
	double	Ti[9]={//Ti=D*T'
				T[0]*_lambdaLong, T[3]*_lambdaLong, T[6]*_lambdaLong,
				T[1]*_lambdaMiddle, T[4]*_lambdaMiddle, T[7]*_lambdaMiddle,
				T[2]*_lambdaMin, T[5]*_lambdaMin, T[8]*_lambdaMin
			};
	multMatrixMatrix<double>(T,Ti,3,Ti);
	for(int j=0;j<numGradients;++j, phi+=idxInc){
		double *g=&gradients[3*j];
		double eval=evaluateQuadraticForm(Ti, g, 3);
		*phi=exp(-b*eval);
	}
}

void computeDiffusionFunction(double *dir, double _lambdaScale, double _lambdaDifference, double b, double *gradients, int numGradients, double *phi, int idxInc){
	for(int j=0;j<numGradients;++j, phi+=idxInc){
		double *g=&gradients[3*j];
		double prod=dotProduct(g,dir,3);
		*phi=exp(-b*(_lambdaDifference*prod*prod+_lambdaScale));
	}
}


void groupCoefficients(double *_alpha, int nalphas, double threshold, vector<set<int> > &neighborhoods, vector<set<int> > &groups){
	pair<double, int> *alpha=new pair<double, int>[nalphas];
	bool *used=new bool[nalphas];
	memset(used, 0, sizeof(bool)*nalphas);
	for(int i=0;i<nalphas;++i){
		alpha[i]=make_pair(_alpha[i], i);
	}
	sort(alpha, alpha+nalphas);
	bool groupAdded=true;
	groups.clear();
	while(groupAdded){
		groupAdded=false;
		set<int> cluster;
		bool groupChanged=true;
		while(groupChanged && (cluster.size()<5)){
			groupChanged=false;
			for(int i=nalphas-1;i>=0;--i)if((alpha[i].first>threshold) && !(used[alpha[i].second])){
				if(cluster.empty()){
					cluster.insert(alpha[i].second);
					used[alpha[i].second]=true;
					groupAdded=true;
				}else{//check if alpha[i].second is in the neighborhood of any of the cluster members
					int candidate=alpha[i].second;
					for(set<int>::iterator it=cluster.begin();it!=cluster.end();++it){// for each cluster member *it
						if(neighborhoods[*it].find(candidate)!=neighborhoods[*it].end()){
							cluster.insert(candidate);
							used[candidate]=true;
							groupChanged=true;
							break;
						}
					}
				}
				if(cluster.size()==5){
					break;
				}
			}
		}
		if(!(cluster.empty())){
			groups.push_back(cluster);
			groupAdded=true;
		}
	}
	delete[] used;
	delete[] alpha;
}

void getBigPeaks(double prop, double *RES_pdds, double *RES_amount, int &RES_count){
	double thr=getMaxVal(RES_amount, RES_count);
	thr*=prop;
	int selected=0;
	for(int i=0;i<RES_count;++i)if(thr<RES_amount[i]){
		if(selected<i){
			memcpy(&RES_pdds[3*selected], &RES_pdds[3*i], sizeof(double)*3);
			RES_amount[selected]=RES_amount[i];
		}
		++selected;
	}
	RES_count=selected;
}


void computeCentroid(double *pdds, int npdds, double *alphas, const set<int> &cluster, double *centroid, double &amountDiff){
	memset(centroid, 0, sizeof(double)*3);
	if(cluster.empty()){
		return;
	}
	double maxAlpha=-1;
	double *reference=NULL;
	for(set<int>::const_iterator it=cluster.begin();it!=cluster.end();++it){
		if(alphas[*it]>maxAlpha){
			maxAlpha=alphas[*it];
			reference=&pdds[3*(*it)];
		}
	}
		
	for(set<int>::const_iterator it=cluster.begin();it!=cluster.end();++it){
		double *p=&pdds[3*(*it)];
		double prod=dotProduct(p,reference,3);
		if(prod<0){
			for(int i=0;i<3;++i){
				centroid[i]-=alphas[*it]*p[i];
			}
		}else{
			for(int i=0;i<3;++i){
				centroid[i]+=alphas[*it]*p[i];
			}
		}
	}
	double centroidNorm=sqrt(dotProduct(centroid, centroid, 3));
	for(int i=0;i<3;++i){
		centroid[i]/=centroidNorm;
	}
	amountDiff=0;
	for(set<int>::const_iterator it=cluster.begin();it!=cluster.end();++it){
		double *p=&pdds[3*(*it)];
		double prod=dotProduct(p,centroid,3);
		amountDiff+=fabs(prod)*alphas[*it];
	}
}


void save4DNifti(const string &fname, double *data, int nslices, int nrows, int ncols, int len){
	int dims[5]={4, ncols, nrows, nslices, len};
	//create nifti images
	nifti_image *nii=nifti_make_new_nim(dims,DT_FLOAT32,true);
	float *nii_data=(float *)nii->data;
	//set file names
	nifti_set_filenames(nii,fname.c_str(),0,1);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				double *currentVector=&data[pos*len];
				for(int k=0;k<len;++k){
					nii_data[c + ncols*((nrows-1-r) + nrows*(s + nslices*k))]=currentVector[k];
				}
			}
		}
	}
	nifti_image_write(nii);
	//free images
	nifti_image_free(nii);
}





void saveDWINifti(const string &fname, double *s0, double *dw, int nslices, int nrows, int ncols, int len){
	int dims[5]={4, ncols, nrows, nslices, len+1};
	//create nifti images
	nifti_image *nii=nifti_make_new_nim(dims,DT_FLOAT32,true);
	float *nii_data=(float *)nii->data;
	//set file names
	nifti_set_filenames(nii,fname.c_str(),0,1);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				double *currentVector=&dw[pos*len];

				nii_data[c + ncols*((nrows-1-r) + nrows*(s + nslices*0))]=s0[pos];
				for(int k=1;k<=len;++k){
					nii_data[c + ncols*((nrows-1-r) + nrows*(s + nslices*k))]=currentVector[k-1];
				}
			}
		}
	}
	nifti_image_write(nii);
	//free images
	nifti_image_free(nii);
}

void save3DNifti(const string &fname, unsigned char *vol, int nslices, int nrows, int ncols){
	int dims[4]={3, ncols, nrows, nslices};
	//create nifti images
	nifti_image *nii=nifti_make_new_nim(dims,DT_UINT8,true);
	unsigned char *nii_data=(unsigned char *)nii->data;
	//set file names
	nifti_set_filenames(nii,fname.c_str(),0,1);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				nii_data[c + ncols*((nrows-1-r) + nrows*s)]=vol[pos];
			}
		}
	}
	nifti_image_write(nii);
	//free images
	nifti_image_free(nii);
}

void save3DNifti(const string &fname, double *vol, int nslices, int nrows, int ncols){
	int dims[4]={3, ncols, nrows, nslices};
	//create nifti images
	nifti_image *nii=nifti_make_new_nim(dims,DT_FLOAT32,true);
	float *nii_data=(float *)nii->data;
	//set file names
	nifti_set_filenames(nii,fname.c_str(),0,1);
	for(int s=0;s<nslices;++s){
		for(int r=0;r<nrows;++r){
			for(int c=0;c<ncols;++c){
				int pos=s*nrows*ncols+r*ncols+c;
				nii_data[c + ncols*((nrows-1-r) + nrows*s)]=float(vol[pos]);
			}
		}
	}
	nifti_image_write(nii);
	//free images
	nifti_image_free(nii);
}