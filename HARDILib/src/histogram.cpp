//Adapted from MRTrix
#include "histogram.h"
#include <stdio.h>
#include <iostream>
#include <set>
#include "bits.h"
using namespace std;
Histogram::Histogram(double *v, int n, int nbuckets){
	if(nbuckets<10){
		cerr<<"Error initialising Histogram: number of buckets must be greater than 10"<<endl;
	}
	cerr<<"Initialising histogram with "<<nbuckets<<" buckets..."<<endl;
    list.resize(nbuckets);

    double minVal=1e300, maxVal=-1e300;
	for(int i=0;i<n;++i)if(isNumber(v[i])){
		if(v[i]<minVal){
			minVal=v[i];
		}
		if(v[i]>maxVal){
			maxVal=v[i];
		}
	}
	if((maxVal-minVal)<1e-9){
		good_flag=false;
	}else{
		good_flag=true;
		for (unsigned i=0;i<list.size();++i){
			list[i].value=minVal+((maxVal-minVal)*(i+0.5)/list.size());
		}
		for(int i=0;i<n;++i)if(isNumber(v[i])){
			double val=v[i];
			int pos=(int)(list.size()*((val-minVal)/(maxVal-minVal)));
			if(pos>=list.size()){
				pos=list.size()-1;
			}
			list[pos].frequency++;
		}
	}
	
}

Histogram::~Histogram(){
}

int Histogram::getFrequency(int index)const{ 
	return list[index].frequency;
}

double Histogram::getValue(int index)const{
	return list[index].value;
}

unsigned Histogram::size()const{
	return (list.size()); 
}

double Histogram::getFirstMin(void)const{
	int first_peak = 0;
	int first_peak_index = 0;
	int second_peak = 0;
	int second_peak_index = 0;
	int first_minimum = 0;
	int first_min_index = 0;
	int range_step = list.size()/20;
	int range = list.size()/20;
	int index;

	for(index=0; index<range;index++){
		if(list[index].frequency > first_peak){
			first_peak=list[index].frequency;
			first_minimum=first_peak;
			first_min_index=first_peak_index=index;
		}
	}
	range=first_peak_index+range_step;
	for(index=first_peak_index;index<range;index++){
		if (list[index].frequency<first_minimum){
			first_minimum=list[index].frequency;
			first_min_index=second_peak_index=index;
		}
	}
	range = first_min_index + range_step;
	for(index=first_min_index;index<range;index++){
		if (list[index].frequency > second_peak){
			second_peak = list[index].frequency;
			second_peak_index = index;
		}
	}
    return list[first_min_index].value;
}

bool Histogram::good(void){
	return good_flag;
}

void Histogram::printFrequencies(const char *fname){
	FILE *F=fopen(fname, "w");
	for(unsigned i=0;i<list.size();++i){
		fprintf(F, "%0.15lf\t%d\n", list[i].value, list[i].frequency);
	}
	fclose(F);
}


double Histogram::getLastPeak(int windowSize)const{
	set<pair<int, int> > F;
	int n=list.size();
	if(windowSize>n){
		windowSize=n;
	}
	for(int i=0;i<windowSize;++i){
		F.insert(make_pair(-list[n-1-i].frequency, n-1-i));
	}
	for(int i=n-windowSize-1;i>=0;--i){
		int currentFreqMax=-F.begin()->first;
		int currentMaxPos=F.begin()->second;
		F.erase(make_pair(-list[i+windowSize].frequency, i+windowSize));
		F.insert(make_pair(-list[i].frequency, i));
		int newFreqMax=-F.begin()->first;
		if(newFreqMax<currentFreqMax){
			return list[currentMaxPos].value;
		}
	}
	return list[F.begin()->second].value;
}

