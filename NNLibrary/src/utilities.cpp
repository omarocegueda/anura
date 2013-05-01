#include "utilities.h"
#include "macros.h"
#include <queue>
#include <fstream>
#include "DirectoryListing.h"
#include <vector>
#include <string>
#include <iostream>
using namespace std;

const int NUM_NEIGHBORS=26;
int dRow[]	={-1, 0, 1,  0,-1, 1, 1,-1, 0,  -1, 0, 1, 0,-1, 1, 1,-1,  -1, 0, 1, 0,-1, 1, 1,-1, 0};
int dCol[]	={ 0, 1, 0, -1, 1, 1,-1,-1, 0,   0, 1, 0,-1, 1, 1,-1,-1,  0, 1, 0, -1, 1, 1,-1,-1, 0};
int dSlice[]={-1,-1,-1, -1,-1,-1,-1,-1,-1,   0, 0, 0, 0, 0, 0, 0, 0,   1, 1, 1, 1, 1, 1 ,1 ,1, 1};


bool fileExists(const string& file_name){
#if defined(_WIN32) || defined(_WIN64)
	struct _stat buffer;
	if (!_stat(file_name.c_str(), &buffer)){
		return true;
	} 
#else
	struct stat buffer;
	if (!stat(file_name.c_str(), &buffer)){
		return true;
	} 
#endif
	return false;
}




double getMinVal(double *v, int n){
	double retVal=v[0];
	for(int i=0;i<n;++i){
		retVal=MIN(retVal, v[i]);
	}
	return retVal;
}
double getMaxVal(double *v, int n){
	double retVal=v[0];
	for(int i=0;i<n;++i){
		retVal=MAX(retVal, v[i]);
	}
	return retVal;
}

//considers values of the form v[k*offset+idx]
double getMinVal_offset(double *v, int n, int offset, int idx){
	double retVal=v[idx];
	for(;idx<n;idx+=offset){
		retVal=MIN(retVal, v[idx]);
	}
	return retVal;
}

//considers values of the form v[k*offset+idx]
double getMaxVal_offset(double *v, int n, int offset, int idx){
	double retVal=v[idx];
	for(;idx<n;idx+=offset){
		retVal=MAX(retVal, v[idx]);
	}
	return retVal;
}

int getMinIndex(double *v, int n){
	int retVal=0;
	for(int i=1;i<n;++i){
		if(v[i]<v[retVal]){
			retVal=i;
		}
	}
	return retVal;
}

int getMaxIndex(double *v, int n){
	int retVal=0;
	for(int i=1;i<n;++i){
		if(v[retVal]<v[i]){
			retVal=i;
		}
	}
	return retVal;
}

void normalizeToRange(double *v, int n, double a, double b){
	double minVal=getMinVal(v,n);
	double maxVal=getMaxVal(v,n);
	double diff=maxVal-minVal;
	double rangeSize=b-a;
	if(diff<EPSILON){
		for(int i=0;i<n;++i){
			v[i]=a;
		}
	}else{
		for(int i=0;i<n;++i){
			v[i]=a+rangeSize*(v[i]-minVal)/diff;
		}
	}
}

unsigned char getRed_circular(unsigned char x){
	if(x<64){
		return 255-x*255/63;
	}else if(x<128){
		return (x-63)*255/64;
	}else if(x<192){
		return 255-(x-127)*255/64;
	}
	return (x-191)*255/64;
}

unsigned char getGreen_circular(unsigned char x){
	if(x<64){
		return x*255/63;
	}else if(x<128){
		return 255;
	}else if(x<192){
		return 255-(x-127)*255/64;
	}
	return 0;
}

unsigned char getBlue_circular(unsigned char x){
	if(x<64){
		return 0;
	}else if(x<128){
		return (x-63)*255/64;
	}else if(x<192){
		return 255;
	}
	return 255-(x-191)*255/64;
}

void getFalseColor_circular(unsigned char x, unsigned char &r, unsigned char &g, unsigned char &b){
	r=getRed_circular(x);
	g=getGreen_circular(x);
	b=getBlue_circular(x);
}

void getFalseColor_circular(double y, double x, unsigned char &r, unsigned char &g, unsigned char &b){
	double a=2*atan2(y,x);
	unsigned char c=(unsigned char)(255*((a+M_PI)/(2*M_PI)));
	r=getRed_circular(c);
	g=getGreen_circular(c);
	b=getBlue_circular(c);
}

void applyThreshold(double *data, int n, double thr, unsigned char *output){
	int total=0;
	for(int i=0;i<n;++i){
		if(data[i]>thr){
			output[i]=1;
			++total;
		}else{
			output[i]=0;
		}
		
	}
}

void readFileNamesFromFile(const string &fname, vector<string> &names){
	names.clear();
	ifstream F(fname.c_str());
	if(!(F.good())){
		cerr<<"Error: could not open file '"<<fname<<"'"<<endl;
		return;
	}
	string nextString;
	while(!F.eof()){
		if(F>>nextString){
			names.push_back(nextString);
		}
	}
	F.close();
}


vector<string> ReadFileNamesFromDirectory(const char* pattern){
  size_t pos;
  string spattern=pattern;
  string path="";
  pos=spattern.find_last_of(PATH_SEPARATOR);
  if(pos!=string::npos){
    path=spattern.substr(0,pos+1);
  }
  DirectoryListing dir;
  char fileName[1<<8];
  dir.First(fileName, pattern);
  vector<string> names;
  if(strcmp(fileName,"")==0){
    cout << "No files match the pattern" << endl;
    names;
  }else{
    names.push_back(path+fileName);
  }
  while(dir.Next(fileName)){
    names.push_back(path+fileName);
  }
  return names;
}

string changeExtension(const string &fname, const string &newExt){
	size_t p=fname.find_last_of(".");
	if(p==string::npos){
		return fname+newExt;
	}
	string retVal=fname.substr(0,p)+newExt;
	return retVal;
}

string getExtension(const string &fname){
	size_t p=fname.find_last_of(".");
	if(p==std::string::npos){
		return "";
	}
	string retVal=fname.substr(p,fname.size()-p);
	return retVal;
}


