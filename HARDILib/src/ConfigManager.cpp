#include "ConfigManager.h"
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

ConfigManager::ConfigManager(){
	vars=NULL;
	nvars=0;
}

ConfigManager::ConfigManager(ConfigField *_vars, int _nvars){
	vars=_vars;
	nvars=_nvars;
}

void ConfigManager::loadFromFile(const char *fname){
	ifstream F(fname);
	string nextLine;
	getline(F, nextLine);
	while(!(F.eof())){
		istringstream is(nextLine);
		string name;
		is>>name;
		ConfigField *field=getField(name.c_str());
		if(field==NULL){
			if((!(name.empty())) && (name[0]!='#')){
				cerr<<"ConfigManager: unknown field '"<<name<<"'"<<endl;
			}
			getline(F, nextLine);
			continue;
		}
		int intVar;
		double doubleVar;
		size_t pos;
		string sub;
		switch(field->type){
			case ConfigField::CFFT_Bool:
				if(is>>intVar){
					*((bool*)(field->data))=(intVar!=0);
				}
			break;
			case ConfigField::CFFT_Double:
				if(is>>doubleVar){
					*((double*)(field->data))=doubleVar;
				}
			break;
			case ConfigField::CFFT_Int:
				if(is>>intVar){
					*((int*)(field->data))=intVar;
				}
			break;
			case ConfigField::CFFT_String:
				pos=nextLine.find(name);
				pos=nextLine.find_first_not_of(" \t",pos+name.size());
				if(pos==std::string::npos){
					((char*)(field->data))[0]='\0';
				}else{
					sub=nextLine.substr(pos);
					strcpy((char*)(field->data), sub.c_str());
				}
				
			break;
			default:
				cerr<<"ConfigField type not supported."<<endl;
			break;
		}
		getline(F, nextLine);
	}
	F.close();
}

void ConfigManager::saveToFile(const char *fname){
}

ConfigManager::~ConfigManager(){
}

ConfigField *ConfigManager::getField(const char *name){
	for(int i=0;i<nvars;++i){
		if(strcmp(name, vars[i].name)==0){
			return &vars[i];
		}
	}
	return NULL;
}

void ConfigManager::showConfig(void){
	for(int i=0;i<nvars;++i){
		cerr<<vars[i].name<<":\t";
		switch(vars[i].type){
			case ConfigField::CFFT_Bool:
				cerr<<((*(bool *)(vars[i].data))?"true":"false")<<endl;
			break;
			case ConfigField::CFFT_Int:
				cerr<<(*(int *)(vars[i].data))<<endl;
			break;
			case ConfigField::CFFT_Double:
				cerr<<(*(double *)(vars[i].data))<<endl;
			break;
			case ConfigField::CFFT_String:
				cerr<<((char *)(vars[i].data))<<endl;
			break;
			default:
				cerr<<"?"<<endl;
			break;
		}
	}
}
