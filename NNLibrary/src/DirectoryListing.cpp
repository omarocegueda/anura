#if defined(_WIN32) || defined(_WIN64)
	#include <windows.h>
	#include <winbase.h>
	#include <stdlib.h>
	#include <direct.h>
#else
	#include <stddef.h>
	#include <stdio.h>
	#include <ctype.h>
	#include <string.h>
	#include <unistd.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <dirent.h>
#endif

#include "DirectoryListing.h"
#include <iostream>
#include <errno.h>
#include <string>
using namespace std;

bool DirUtil::CreateDir(const string& name){
#if defined(_WIN32) || defined(_WIN64)
	int ret = mkdir(name.c_str());
#else
	// RWX for owner/group, and RX for others
	int ret = mkdir(name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif
	if(ret != 0 && errno != EEXIST){
		fprintf(stderr, "ERROR: cannot create directory %s.\n", name.c_str());
		return false;
	}
	return true;
}

void DirUtil::RemoveExtension(char *out,char *str){
	strcpy(out,str);
	int i=strlen(out);
	while (out[i]!='.') i--;
	out[i]='\0';
}

#if (!defined(_WIN32)) && (!defined(_WIN64))

// see linuxquestions for this function
int patiMatch (const char* pattern, const char* string){
	switch (pattern[0]){
		case '\0':
			return !string[0];
		case '*' :
			return patiMatch(pattern+1, string) || string[0] && patiMatch(pattern, string+1);
		case '?' :
			return string[0] && patiMatch(pattern+1, string+1);
		default:
			return (toupper(pattern[0]) == toupper(string[0])) && patiMatch(pattern+1, string+1);
	}
}

bool getNextFile(DIR **dp, dirent **ep, char *pattern){
	while(*ep = readdir(*dp)){
		if(patiMatch(pattern, (*ep)->d_name)) return true;
	}
	return false;
}

#endif

void DirectoryListing::First(char *result,const char *type){
#if defined(_WIN32) || defined(_WIN64)
	WIN32_FIND_DATA FindFileData;
	h1=FindFirstFile(type,&FindFileData);
	
	if (h1!=INVALID_HANDLE_VALUE){
		strcpy(result,FindFileData.cFileName);
	}else{
		strcpy(result,"");
	}
#else
	string filePattern=type;
	string directory;
	size_t pos=filePattern.find_last_of(PATH_SEPARATOR);
	if(pos!=string::npos){
		directory=filePattern.substr(0,pos+1);
		filePattern=filePattern.substr(pos+1, filePattern.size()-(pos+1));
	}else{
		directory="./";
	}
	if(dp){
		closedir(dp);
	}
	if(pattern){
		delete[] pattern;
	}
	pattern=new char[filePattern.size()+1];
	strcpy(pattern, filePattern.c_str());
	dp=opendir(directory.c_str());
	if( dp != NULL ){
		if(!getNextFile(&dp, &ep, pattern)) {
			closedir(dp);
			dp = NULL;
			result[0] = '\0';
			return;
		}
		if(strlen(ep->d_name) > 0) {
			strcpy(result, ep->d_name);
		}else{
			result[0] = '\0';
		}
	}
#endif
}

bool DirectoryListing::Next(char *result){
#if defined(_WIN32) || defined(_WIN64)
	WIN32_FIND_DATA FindFileData;
	if (!FindNextFile(h1,&FindFileData)){
		FindClose(h1);
		h1 = NULL;
		return false;
	}
	strcpy(result,FindFileData.cFileName);
	return true;
#else
	if(dp!=NULL){
		if(!getNextFile(&dp, &ep, pattern)) {
			closedir(dp);
			dp = NULL;
			return false;
		}
		strcpy(result, ep->d_name);
		return true;
	}
	return false;
#endif
}
