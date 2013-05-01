#include "mrtrixio.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <vector>
#include "bits.h"
using namespace std;
int read_mrtrix_tracks(const char *fname, double **&tracks, int *&trackSizes, int &numTracks, map<string, string> &properties){
	if(tracks!=NULL){
		cerr<<"Warning: tracks pointer is not null. It may cause a memory leak."<<endl; 
	}
	string nextLine;
	ifstream F(fname);
	getline(F, nextLine);
	if(nextLine.find("mrtrix tracks")==string::npos){
		cerr<<"Error: invalid tracks file."<<endl; 
		return -1;
	}
	do{
		getline(F, nextLine);
		size_t pos;
		if((pos=nextLine.find(":"))!=string::npos){
			string fieldname=nextLine.substr(0,pos);
			size_t pos2=nextLine.find_first_not_of(" \t",pos+1);
			if(pos2!=string::npos){
				properties[fieldname]=nextLine.substr(pos2);
			}else{
				properties[fieldname]="";
			}
		}
	}while((!F.eof()) && (nextLine!="END"));
	F.close();
	map<string, string>::iterator it=properties.find("file");
	if(it==properties.end()){
		cerr<<"Error: offset not specified in file '"<<fname<<"'."<<endl; 
		return -1;
	}
	size_t pos=it->second.find(".");
	if(pos==string::npos){
		cerr<<"Error: invalid tracks format file '"<<fname<<"'. '.' expected."<<endl; 
		return -1;
	}
	if(properties["datatype"]!="Float32LE"){
		cerr<<"Error: unsupported data type: '"<<properties["datatype"]<<"'."<<endl;
		return -1;
	}
	F.open(fname,ios::binary);
	F.seekg(0, ios::end);
	int fsize=F.tellg();
	
	istringstream is(it->second.substr(pos+1));
	int offset;
	is>>offset;
	F.seekg(offset, ios::beg);
	char *data=new char[fsize-offset];
	
	F.read(data, fsize-offset);
	F.close();
	float *fdata=(float*)data;
	int N=(fsize-offset)/4;
	if(((fsize-offset)%4)!=0){
		cerr<<"Warning: unexpected data length: "<<fsize-offset<<" (multiple of 4 expected)."<<endl;
	}
	istringstream is2(properties["count"]);
	is2>>numTracks;
	trackSizes=new int[numTracks];
	tracks=new double*[numTracks];
	int len=0;
	int cnt=0;
	for(int i=0;i<N;i+=3){
		if(isQNAN(fdata[i])){
			trackSizes[cnt]=len;
			tracks[cnt]=new double[3*len];
			for(int j=i-3*len,k=0;j<i;++j,++k){
				tracks[cnt][k]=fdata[j];
			}
			++cnt;
			len=0;
		}else{
			++len;
		}
	}
	delete[] data;
	return 0;
}


int write_mrtrix_image(const char *fname, double *img, int ndim, int *dims, int *voxSizes, int *layout,int *layout_orientations, double *transform){
	FILE *F=fopen(fname, "w");
	//=================print header================
	fprintf(F, "mrtrix image\ndim: %d", dims[0]);
	for(int i=1;i<ndim;++i){
		fprintf(F, ",%d", dims[i]);
	}
	//-------voxel size-----
	fprintf(F, "\nvox: ");
	if(voxSizes!=NULL){
		fprintf(F, "%d", voxSizes[0]);
		for(int i=1;i<ndim;++i){
			fprintf(F, ",%d", voxSizes[i]);
		}
	}else{
		fprintf(F, "2");
		for(int i=1;i<ndim;++i){
			fprintf(F, ",2");
		}
	}
	//-------layout------
	if((layout!=NULL)&&(layout_orientations!=NULL)){
		if(layout_orientations[0]<0){
			fprintf(F, "\nlayout: -%d", layout[0]);
		}else{
			fprintf(F, "\nlayout: +%d", layout[0]);
		}
		
		for(int i=1;i<ndim;++i){
			if(layout_orientations[i]<0){
				fprintf(F, ",-%d",layout[i]);
			}else{
				fprintf(F, ",+%d",layout[i]);
			}
		}
	}else{
		fprintf(F, "\nlayout: +0");
		for(int i=1;i<ndim;++i){
			fprintf(F, ",+%d",i);
		}
	}
	
	//-------datatype----
	fprintf(F, "\ndatatype: Float32LE");
	//-------transform---
	if(transform==NULL){
		fprintf(F, "\ntransform: 1,0,0,-%0.1lf", (dims[0])*0.5);
		fprintf(F, "\ntransform: 0,1,0,-%0.1lf", (dims[1])*0.5);
		fprintf(F, "\ntransform: 0,0,1,-%0.1lf", (dims[2])*0.5);
	}else{
		for(int i=0;i<3;++i){
			fprintf(F, "\ntransform: %lf", transform[4*i]);
			for(int j=1;j<4;++j){
				fprintf(F, ",%lf", transform[4*i+j]);
			}
		}
	}
	
	
	long offset=ftell(F)+20;
	fprintf(F, "\nfile: . %d\nEND\n", offset);
	fclose(F);
	//=============print binary data===============
	F=fopen(fname, "ab");
	fseek (F, offset, SEEK_SET);
	int nvox=1;
	for(int i=0;i<ndim;++i){
		nvox*=dims[i];
	}
	for(int i=0;i<nvox;++i){
		float val=img[i];
		img[i]=val;
		fwrite(&val, sizeof(float), 1, F);
	}
	fclose(F);
	return 0;
}

int write_mrtrix_image(const char *fname, double *img, int nslices, int nrows, int ncols){
	int ndim=3;
	int dims[]={ncols, nrows, nslices};
	int voxSizes[]={1,1,1};
	int layout[]={0,1,2};
	int layout_orientations[]={1,-1,1};
	int retVal=write_mrtrix_image(fname, img, ndim, dims, voxSizes, layout, layout_orientations);
	return retVal;
}

int write_mrtrix_image(const char *fname, double *img, int nslices, int nrows, int ncols, int signalLength){
	int ndim=4;
	int dims[]={ncols, nrows, nslices, signalLength};
	int voxSizes[]={1,1,1,1};
	int layout[]={1,2,3,0};
	int layout_orientations[]={1,-1,1,1};
	int retVal=write_mrtrix_image(fname, img, ndim, dims, voxSizes, layout, layout_orientations);
	return retVal;
}

int read_mrtrix_image(const char *fname, double *&img, int &ndims, int *&dims, int *&vox, int *&layout, int *&layout_orientations, double *&transform){
	ifstream fs(fname);
	if(!fs.good()){
		return -1;
	}
	string nextLine;
	getline(fs, nextLine);
	if(nextLine!="mrtrix image"){
		return -1;
	}
	int transformLine=0;
	string ffname;
	int offset=-1;
	ndims=0;
	transform=new double[12];
	while(nextLine!="END"){
		replace(nextLine.begin(), nextLine.end(), ',', ' ');
		istringstream is(nextLine);
		string header;
		is>>header;
		if(header=="dim:"){
			vector<double> v;
			int sz;
			while(is>>sz){
				v.push_back(sz);
			}
			ndims=v.size();
			dims=new int[ndims];
			for(int i=0;i<ndims;++i){
				dims[i]=v[i];
			}
		}else if(header=="vox:"){
			vector<int> v;
			int sz;
			while(is>>sz){
				v.push_back(sz);
			}
			vox=new int[v.size()];
			for(unsigned i=0;i<v.size();++i){
				vox[i]=v[i];
			}
		}else if(header=="layout:"){
			string order;
			vector<int> lo;
			vector<int> l;
			while(is>>order){
				if(order[0]=='+'){
					lo.push_back(1);
				}else{
					lo.push_back(-1);
				}
				istringstream is2(order.substr(1));
				int x;
				is2>>x;
				l.push_back(x);
			}
			layout=new int[l.size()];
			layout_orientations=new int[lo.size()];
			for(unsigned i=0;i<l.size();++i){
				layout[i]=l[i];
				layout_orientations[i]=lo[i];
			}
		}else if(header=="datatype:"){
			string datatype;
			is>>datatype;
			if(datatype!="Float32LE"){
				cerr<<"Data type "<<datatype<<" not suported."<<endl;
				return -1;
			}
		}else if(header=="transform:"){
			for(int i=0;i<4;++i){
				is>>transform[transformLine*4+i];
			}
			++transformLine;
		}else if(header=="file:"){
			is>>ffname>>offset;
		}
		getline(fs, nextLine);
	}
	fs.close();
	if(offset<0){
		return -1;
	}
	if(transform==NULL)
	if(transformLine!=3){
		memset(transform, 0, sizeof(double)*12);
		for(int i=0;i<3;++i){
			transform[4*i+i]=1;
		}
	}
	//-------scan binary data---------
	int nEntries=1;
	for(int i=0;i<ndims;++i){
		nEntries*=dims[i];
	}
	img=new double[nEntries];
	FILE *F=fopen(fname,"rb");
	fseek (F, offset, SEEK_SET);
	for(int i=0;i<nEntries;++i){
		float f;
		fread(&f, sizeof(float),1, F);
		img[i]=f;
	}
	//---------------------------------
	fclose(F);
	return 0;
}

