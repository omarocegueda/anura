#ifndef UTILITIES_H
#define UTILITIES_H
#include <sys/stat.h>
#include "nifti1_io.h"
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include "macros.h"
#define std_voxel_index(i,j,k) ((k)*(rows*cols)+(rows-1-(i))*cols+(j))
#define std_voxel_slice(x)	((x)/(rows*cols))
#define std_voxel_row(x)	(rows-1-(((x)%(rows*cols))/cols))
#define std_voxel_col(x)	(((x)%(rows*cols))%cols)

extern const int NUM_NEIGHBORS;
extern int dRow[];
extern int dCol[];
extern int dSlice[];



bool fileExists(const std::string& file_name);
//void floodFill(unsigned char *data, int rows, int cols, int slices, int i0, int j0, int k0, unsigned char r, unsigned char g, unsigned char b);
//void markStates(unsigned char *data, int rows, int cols, int slices);

/*
	asumimos val>0
*/
template<class T> int floodFill(unsigned char *data, int rows, int cols, int slices, int i0, int j0, int k0, T val, T *marked){
	unsigned char target=data[std_voxel_index(i0,j0,k0)];
	std::queue<int> Q;
	Q.push(std_voxel_index(i0,j0,k0));
	marked[std_voxel_index(i0,j0,k0)]=val;
	int total=1;
	while(!Q.empty()){
		int current=Q.front();
		Q.pop();
		i0=std_voxel_row(current);
		j0=std_voxel_col(current);
		k0=std_voxel_slice(current);
		for(int k=0;k<NUM_NEIGHBORS;++k){
			int ii=i0+dRow[k];
			int jj=j0+dCol[k];
			int kk=k0+dSlice[k];
			
			if(IN_RANGE(ii, 0, rows) && IN_RANGE(jj, 0, cols) && IN_RANGE(kk, 0, slices)){
				int next=std_voxel_index(ii,jj,kk);
				if((data[next]==target) && (marked[next]==0)){
					marked[next]=val;
					Q.push(next);
					++total;
				}
			}
		}
	}
	return total;
}

//void getTensors(nifti_image * nii, int k, double *tensors);
//void getMask(nifti_image * nii, int k, unsigned char *output);

//int drawIntensityMap(cv::Mat &M, int rows, int cols, double *x, int n, int *indexToPosition, int slice, bool falseColor);

double getMinVal(double *v, int n);
double getMaxVal(double *v, int n);

//considers values of the form v[k*offset+idx]
double getMaxVal_offset(double *v, int n, int offset, int idx);

//considers values of the form v[k*offset+idx]
double getMinVal_offset(double *v, int n, int offset, int idx);

int getMinIndex(double *v, int n);
int getMaxIndex(double *v, int n);

void normalizeToRange(double *v, int n, double a, double b);
unsigned char getRed_circular(unsigned char x);
unsigned char getGreen_circular(unsigned char x);
unsigned char getBlue_circular(unsigned char x);
void getFalseColor_circular(unsigned char x, unsigned char &r, unsigned char &g, unsigned char &b);
void getFalseColor_circular(double y, double x, unsigned char &r, unsigned char &g, unsigned char &b);
void readFileNamesFromFile(const std::string &fname, std::vector<std::string> &names);
std::string changeExtension(const std::string &fname, const std::string &newExt);
std::string getExtension(const std::string &fname);

template<class T> void getIntensityColor(int val, T &r, T &g, T &b){
	if(val<64){
		r=0;
		g=(255*val)/63;
		b=255;
	}else if(val<128){
		r=0;
		g=255;
		b=255-(255*(val-64))/63;
	}else if(val<192){
		r=255*(val-128)/63;
		g=255;
		b=0;
	}else{
		r=255;
		g=255-(255*(val-192))/63;
		b=0;
	}
}

template<class T> void freeMatrix(T **&data, int rows, int cols){
	for(int i=0;i<rows;++i){
		delete[] data[i];
	}
	delete[] data;
	data=NULL;
}

template<class T> void newMatrix(T **&data, int rows, int cols){
	data=new T*[rows];
	for(int i=0;i<rows;i++){
		data[i]=new T[cols];
	}
}

template<class T> void freeArray(T*&data){
	if(data!=NULL){
		delete[] data;
		data=NULL;
	}
}

template<class T> void newArray(T*&data, int n){
	if(n<=0){
		data=NULL;
		return;
	}
	data=new T[n];
}

void applyThreshold(double *data, int n, double thr, unsigned char *output);

template<class T> void readPlainMatrixFile(const char *fileName, T **&data, int &n, int &m){
	char *buff=new char[1<<20];//1MB
	std::ifstream F(fileName);
	std::vector<std::vector<T> > table;
	bool ok=true;
	int current=0;
	do{
		F.getline(buff, (1<<20)-1);
		std::istringstream is(buff);
		T nextElement;
		std::vector<T> nextRow;
		while(is>>nextElement){
			nextRow.push_back(nextElement);
		}
		if(!nextRow.empty()){
			table.push_back(nextRow);
			++current;
		}
		if(current>1){
			if(table[current-1].size()!=table[current-2].size()){
				ok=false;
			}
		}
	}while(!F.eof());
	n=0;
	m=0;
	data=NULL;
	if(ok){
		n=table.size();
		m=table[0].size();
		data=new T*[n];
		for(int i=0;i<n;++i){
			data[i]=new T[m];
			for(int j=0;j<m;++j){
				data[i][j]=table[i][j];
			}
		}
	}
}

template<class T> void loadArrayFromFile(const char *fileName, T *&data, int &len){
	FILE *F=fopen(fileName, "r");
	std::vector<T> v;
	while(!feof(F)){
		double val;
		if(fscanf(F,"%lf", &val)==1){
			v.push_back(val);
		}
	}
	fclose(F);
	len=v.size();
	data=new T[len];
	for(int i=0;i<len;++i){
		data[i]=v[i];
	}
}

std::vector<std::string> ReadFileNamesFromDirectory(const char* pattern);


#endif
