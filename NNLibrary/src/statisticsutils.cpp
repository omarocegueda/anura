#include "statisticsutils.h"
#include <time.h>
#include <set>
#include <iostream>
#include <cstdlib>
#include "macros.h"
#include "linearalgebra.h"
#include <fstream>
#include <algorithm>
using namespace std;

int computeDistance(double **data, int n, int m, double *D){
	for(int i=0;i<n;++i){
		D[i*(n+1)]=0;
		for(int j=i+1;j<n;++j){
			double &sum=D[i*n+j];
			sum=0;
			for(int k=0;k<m;++k){
				sum+=SQR(data[i][k]-data[j][k]);
			}
			D[j*n+i]=sum;
		}
	}
	return n*(n-1)/2;
}

void debugMatrix(double *M, int n, int m){
	FILE *F=fopen("debugMatrix.txt", "w");
	for(int i=0;i<n;++i){
		for(int j=0;j<m;++j){
			fprintf(F,"%0.8E\t",M[i*m+j]);
		}
		fprintf(F,"\n");
	}
	fclose(F);
}


int numCategories(int *v, int n){
	set<int> S;
	for(int i=0;i<n;++i){
		S.insert(v[i]);
	}
	return S.size();
}

int unique(int *v, int n, int *&_unique){
	set<int> S;
	for(int i=0;i<n;++i){
		S.insert(v[i]);
	}
	int c=S.size();
	_unique=new int[c];
	for(set<int>::iterator it=S.begin(); it!=S.end();++it){
		*_unique=*it;
		++_unique;
	}
	_unique-=c;
	return c;
}

int select(int nth, set<int>&forbidden){
	if(forbidden.empty()){
		return nth;
	}
	set<int>::iterator it=forbidden.begin();
	if(nth<(*it)){
		return nth;
	}
	int current=*it;
	nth-=current;
	++it;
	while(it!=forbidden.end()){
		if(current+1+nth<(*it)){
			return current+1+nth;
		}
		nth-=(*it-current-1);
		current=*it;
		++it;
	}
	return current+1+nth;
}

//returns 'samples' random numbers in [0, n-1] without replacement
void randomSampleReplacement(int samples, int n, int *selected){
	for(int i=0;i<samples;++i){
		selected[i]=rand()%n;
	}
}

//returns 'samples' random numbers in [0, n-1] without replacement. It assumes samples <= n
//the time complexity is O(M*log(M)) and the memory complexity is Th(M), where M='samples'
//a simple modification of the algorithm can reduce the time complexity to min{O(M*log(M)), O((n-M)*log((n-M)))}
//but for now the user must call the routine efficiently (i.e. if 2*M>n, then it is better to choose n-M elements
//and take the complement)
void randomSampleNoReplacement(int samples, int n, int *selected){
	set<int> S;
	for(int i=0;i<samples;++i){
		int r=rand()%(n-i);
		int sel=select(r, S);
		S.insert(sel);
	}
	samples=0;
	for(set<int>::iterator it=S.begin();it!=S.end();++it){
		selected[samples]=*it;
		++samples;
	}
}


double evaluateBiasedQuadraticForm(double *Q, double *v, double *bias, int n){
	double *tmp=new double[n];
	for(int i=0;i<n;++i){
		tmp[i]=0;
		for(int j=0;j<n;++j){
			tmp[i]+=Q[i*n+j]*(v[j]-bias[j]);
		}
	}
	double sum=0;
	for(int i=0;i<n;++i){
		sum+=(v[i]-bias[i])*tmp[i];
	}
	delete[] tmp;
	return 0.5*sum;
}





void computeCovariance(double **x, int n, int m, double *s){
	for(int i=0;i<m;++i){
		cout<<"Computing covariance matrix:"<<i+1<<"/"<<m<<endl;
		for(int j=i;j<m;++j){
			double sum=0;
			for(int k=0;k<n;++k){
				sum+=x[k][i]*x[k][j];
			}
			sum/=(n-1);
			s[i*m+j]=s[j*m+i]=sum;
		}
	}
}

void computeMean(double **data, int n, int m, double *mean){
	memset(mean, 0, sizeof(double)*m);
	for(int i=0;i<n;++i){
		
		for(int j=0;j<m;++j){
			mean[j]+=data[i][j];
		}
	}
	for(int j=0;j<m;++j){
		mean[j]/=n;
	}
}

void computeCentroids(double **data, int n, int m, int *labels, int numClasses, double **means, int *groupSize){
	memset(groupSize, 0, sizeof(int)*numClasses);
	for(int i=0;i<numClasses;++i){
		memset(means[i], 0, sizeof(double)*m);
	}
	
	for(int i=0;i<n;++i){
		int sel=labels[i];
		groupSize[sel]++;
		for(int j=0;j<m;++j){
			means[sel][j]+=data[i][j];
		}
	}
	for(int i=0;i<numClasses;++i)if(groupSize[i]>0){
		double size=groupSize[i];
		for(int j=0;j<m;++j){
			means[i][j]/=size;
		}
	}
	
	/*for(int i=0;i<n;++i){
		int sel=labels[i];
		for(int j=0;j<m;++j){
			data[i][j]-=means[sel][j];
		}
	}*/
}

void buildCenteringMatrix(double *J, int n){
	double diagTerm=1-1.0/n;
	double rest=-1.0/n;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			if(i==j){
				J[i*n+j]=diagTerm;
			}else{
				J[i*n+j]=rest;
			}
		}
	}
}

void computeROCData(vector<pair<double, int> > &scores, int n, int numSubjects, FILE *F){
	sort(scores.begin(), scores.end());
	int totalProbe=n-numSubjects;
	int falseAcceptanceCount=0;
	int verificationCount=0;
	double verificationAt001FAR=1;
	bool checked=false;
	for(unsigned i=0;i<scores.size();++i){
		if(scores[i].second==1){
			++verificationCount;	
			double falseAcceptanceRate=double(falseAcceptanceCount)/double(scores.size());
			double verificationRate=double(verificationCount)/double(totalProbe);
			if((1<=1000*falseAcceptanceRate) && (!checked)){
				verificationAt001FAR=verificationRate;
				checked=true;
			}
			if(F==NULL){
				cout<<falseAcceptanceRate<<"\t"<<verificationRate<<endl;
			}else{
				fprintf(F, "%lf\t%lf\n", falseAcceptanceRate, verificationRate);
			}
		}else{
			++falseAcceptanceCount;
		}
	}
	cout<<verificationAt001FAR<<endl;
}

void computeROCData(double *D, int *labels, int n, FILE *F){
	set<int> subjects;
	for(int i=0;i<n;++i){
		subjects.insert(labels[i]);
	}
	int numSubjects=subjects.size();
	cerr<<endl<<"Num subjects:"<<numSubjects<<endl;
	//mark gallery
	int *seen=new int[numSubjects];
	int *galleryIndex=new int[numSubjects];
	memset(galleryIndex, -1, sizeof(int)*numSubjects);
	memset(seen, 0, sizeof(int)*numSubjects);
	for(int i=0;i<n;++i){
		if(seen[labels[i]]==0){
			galleryIndex[labels[i]]=i;
			seen[labels[i]]=1;
		}
	}
	//============================================
	vector<pair<double, int> > scores;
	int index=0;
	for(int i=0;i<n;++i){
		if(galleryIndex[labels[i]]!=i){//it means "i is probe"
			for(int claim=0;claim<numSubjects;++claim){
				int pairPosition=i*n+galleryIndex[claim];
				scores.push_back(make_pair(D[pairPosition], labels[i]==claim));
			}
		}
	}
	computeROCData(scores, n, numSubjects, F);
	
}

void computeROCThreshold(vector<pair<double, int> > &scores, int n, int numSubjects, double maxFAR, double &threshold, double &verificationRate){
	sort(scores.begin(), scores.end());
	int totalProbe=n-numSubjects;
	int falseAcceptanceCount=0;
	int verificationCount=0;
	verificationRate=1;
	bool checked=false;
	for(unsigned i=0;i<scores.size();++i){
		if(scores[i].second==1){
			++verificationCount;	
			double falseAcceptanceRate=double(falseAcceptanceCount)/double(scores.size());
			double v=double(verificationCount)/double(totalProbe);
			if((maxFAR<=falseAcceptanceRate) && (!checked)){
				verificationRate=v;
				threshold=scores[i].first;
				checked=true;
			}
		}else{
			++falseAcceptanceCount;
		}
	}
}


void randomPermutation(int n, int *perm){
	for(int i=0;i<n;++i){
		perm[i]=i;
	}
	for(int i=0;i<n-1;++i){
		int p=i+rand()%(n-i);
		if(p!=i){
			int temp=perm[i];
			perm[i]=perm[p];
			perm[p]=temp;
		}
	}
}

void computeVerificationScores(double *D, std::vector<int> &ids, std::vector<std::pair<double, int> > &scores, int &numSubjects){
	if(D==NULL){
		return;
	}
	int n=ids.size();
	set<int> subjects;
	for(int i=0;i<n;++i){
		subjects.insert(ids[i]);
	}
	numSubjects=subjects.size();
	//mark gallery
	int *seen=new int[numSubjects];
	int *galleryIndex=new int[numSubjects];
	memset(galleryIndex, -1, sizeof(int)*numSubjects);
	memset(seen, 0, sizeof(int)*numSubjects);
	for(int i=0;i<n;++i){
		if(seen[ids[i]]==0){
			galleryIndex[ids[i]]=i;
			seen[ids[i]]=1;
		}
	}
	//============================================
	scores.clear();
	int index=0;
	for(int i=0;i<n;++i){
		if(galleryIndex[ids[i]]!=i){//it means "i is probe"
			for(int claim=0;claim<numSubjects;++claim){
				int pairPosition=i*n+galleryIndex[claim];
				scores.push_back(make_pair(D[pairPosition], ids[i]==claim));
			}
		}
	}
}


void printROCData(vector<double > &matchScores, vector<double> &nonMatchScores, FILE *file_handle){
	unsigned p=0;
	for(unsigned i=0;i<matchScores.size();++i){
		while((p<nonMatchScores.size()) && nonMatchScores[p]<matchScores[i]){
			++p;
		}
		double far1 = double(p)/double(nonMatchScores.size());
		double ver1 = double(i)/double(matchScores.size());
		if(file_handle!=NULL){
			fprintf(file_handle,"%0.8E\t%0.8E\n", far1, ver1);
		}else{
			cout << far1 << "\t" << ver1 << endl;
		}
	}
}

void printROCData(vector<double > *matchScores, vector<double > *nonMatchScores, int numCurves, FILE *F){
	unsigned *p=new unsigned[numCurves];
	memset(p,0,sizeof(unsigned)*numCurves);
	unsigned i=0;
	bool finished=false;
	while(!finished){
		finished=true;
		for(int c=0;c<numCurves;++c)if(i<matchScores[c].size()){
			finished=false;
			while((p[c]<nonMatchScores[c].size()) && nonMatchScores[c][p[c]]<matchScores[c][i]){
				p[c]++;
			}
			double far1 = double(p[c])/double(nonMatchScores[c].size());
			double ver1 = double(i)/double(matchScores[c].size());
			if(F!=NULL){
				fprintf(F,"%0.8E\t%0.8E\t", far1, ver1);
			}else{
				cout << far1 << "\t" << ver1 << endl;
			}
		}else{
			if(F!=NULL){
				fprintf(F,"1\t1\t");
			}else{
				cout<<"1\t1\t"<<endl;
			}
		}
		if(F!=NULL){
				fprintf(F,"\n");
			}else{
				cout<<endl;
			}
		++i;
	}
	delete[] p;
}


bool ReadMask(string filename, vector<vector<unsigned char> > &mask) {
	ifstream infile(filename.c_str(), ios::binary | ios::in);
	if (infile.fail()) return false;
		string junk;
		int ijunk;
		unsigned int rows, cols;
		unsigned char ujunk;
		unsigned char tjunk;
		// gallery path
		getline(infile, junk);
		//cout << junk << endl;
		// probe path
		getline(infile, junk);
		//cout << junk << endl;
		
		// get rid of 0x00! 
		infile.read((char*) &ujunk, sizeof(unsigned char));
		
		// id
		infile.read((char *) &ijunk, sizeof(int));
		//cout << ijunk << endl;
		// ver
		infile.read((char *) &ijunk, sizeof(int));
		//cout << ijunk << endl;
		// M
		infile.read((char *) &ujunk, sizeof(unsigned char));
		//cout << ujunk << endl;
		// F
		infile.read((char *) &ujunk, sizeof(unsigned char));
		//cout << ujunk << endl;
		// id
		infile >> rows;
		//cout << rows << endl;
		infile >> cols;
		//cout << cols << endl;

		// get rid of new line! 
		infile.read((char*) &ujunk, sizeof(unsigned char));
		vector < unsigned char > temp;
		
		for (unsigned int i=0;i<rows;i++) {
			temp.clear();	
			for (unsigned int j=0;j<cols;j++) {
				infile.read((char *)&tjunk, sizeof(unsigned char));
				temp.push_back(tjunk);
			}
			mask.push_back(temp);
		}
		
		infile.close();
		int channels=1;
		FILE *F=fopen((filename+".fwv").c_str(),"wb");
		fwrite(&rows, sizeof(int), 1, F);
		fwrite(&cols, sizeof(int), 1, F);
		fwrite(&channels, sizeof(int), 1, F);
		int numZeroes=0;
		int numPositives=0;
		int numNegatives=0;
		int numPairs=0;
		for(unsigned i=0;i<rows;++i)
			for(unsigned j=0;j<cols;++j){
				double val=mask[i][j];
				if(mask[i][j]==0){
					numZeroes++;
				}else{
					numPairs++;
				}
				if(mask[i][j]==0x7f){
					numNegatives++;
				}
				if(mask[i][j]==0xff){
					numPositives++;
				}
				fwrite(&val, sizeof(double), 1, F);
			}
		fclose(F);
		cerr<<"Total pairs:"<<rows*cols<<endl;
		cerr<<"Zeroes:"<<numZeroes<<endl;
		cerr<<"Positives:"<<numPositives<<endl;
		cerr<<"Negatives:"<<numNegatives<<endl;
		cerr<<"Verification pairs:"<<numPairs<<endl;
		cerr<<"----------------"<<endl;
	
	return true;
}

double computeVerificationRate(double *D, int n, std::vector<std::vector<unsigned char> > &mask, double falseAcceptanceRate){
	vector<double> matchScores;
	vector<double> nonMatchScores;
	for(int i=0;i<n;++i){
		for(int j=0;j<n;++j){
			if(mask[i][j]==0x7f){//nonmatch
				nonMatchScores.push_back(D[i*n+j]);
			}else if(mask[i][j]==0xff){//match
				matchScores.push_back(D[i*n+j]);
			}
		}
	}
	sort(matchScores.begin(), matchScores.end());
	sort(nonMatchScores.begin(), nonMatchScores.end());
	int nm=int(falseAcceptanceRate*nonMatchScores.size());
	for(int i=-20;i<=20;++i){
		printf("%s%0.8lf\n",i==0?"*":"", nonMatchScores[nm+i]);
	}
	double thr=nonMatchScores[nm];
	double verif=0;
	for(unsigned i=0;i<matchScores.size();++i){
		if(matchScores[i]<thr){
			verif+=1;
		}
	}
	verif/=double(matchScores.size());
	return verif;
}


double uniform(double a, double b){
	return a+(b-a)*(double(rand())/double(RAND_MAX));
}

double gaussianSample(double mean, double sigma){//replace this with the matrix_math.cpp implementation for delivery
	double x1, x2, w, y1, y2;
	do {
		x1 = 2.0 * uniform() - 1.0;
		x2 = 2.0 * uniform() - 1.0;
		w = x1 * x1 + x2 * x2;
	} while ( w >= 1.0 );
	w = sqrt( (-2.0 * log( w ) ) / w );
	y1 = x1 * w;
	y2 = x2 * w;
	return mean+y1*sigma;
}

//returns the largest index i such that v[i]<=thr. Assumes v is sorted in non-decreasing order
int binary_search_verifRate(vector<double> &v, double thr){
	if(v.empty()){
		return -1;
	}
	int a=0;
	int b=v.size()-1;
	while(a<b){
		int mid=(a+b+1)/2;
		if(thr<v[mid]){
			b=mid-1;
		}else{
			a=mid;
			
		}
	}
	if(thr<v[b]){
		return -1;
	}
	return b;
}

void permuteRelativePosition(int *labels, int n, int *labPerm){
	convertToConsecutiveIds(labels, n);
	int numClasses=numCategories(labels,n);
	int *perm=new int[numClasses];
	randomPermutation(numClasses, perm);
	int p=0;
	int current=0;
	while(p<n){
		for(int i=0;i<n;++i){
			if(labels[i]==perm[current]){
				labPerm[p]=i;
				++p;
			}
		}
		++current;
	}
	delete[] perm;
}

int convertToConsecutiveIds(int *ids, int n){
	map<int, int> M;
	for(int i=0;i<n;++i){
		if(M.find(ids[i])==M.end()){
			int currentSize=M.size();
			M[ids[i]]=currentSize;
		}
	}
	for(int i=0;i<n;++i){
		ids[i]=M[ids[i]];
	}
	return M.size();
}
int convertToConsecutiveIds(vector<int> &ids){
	map<int, int> M;
	int n=ids.size();
	for(int i=0;i<n;++i){
		if(M.find(ids[i])==M.end()){
			int currentSize=M.size();
			M[ids[i]]=currentSize;
		}
	}
	for(int i=0;i<n;++i){
		ids[i]=M[ids[i]];
	}
	return M.size();
}


double generateWeibull(double uniform_sample/*uniform [0,1]sample*/, double alpha/*centering parameter*/, double beta/*scale parameter*/, double gamma/*shape parameter*/){
	double l=-log(1.0-uniform_sample);
	double p=pow(l, 1.0/gamma);
	return alpha+beta*p;
}

double generateRician(double nu, double sigma){
	double X=gaussianSample(nu,sigma);
	double Y=gaussianSample(0,sigma);
	double retVal=sqrt(SQR(X)+SQR(Y));
	return retVal;
}