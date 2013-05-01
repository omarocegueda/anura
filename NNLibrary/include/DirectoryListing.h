#ifndef DIRECTORYLISTING_H
#define DIRECTORYLISTING_H
#ifdef _WIN32
#include <windows.h>
#include <winbase.h>
#else
#include <stddef.h>
#include <stdio.h>
#include <sys/types.h>
#include <dirent.h>
#endif
#include <string>
#if defined(_WIN32) || defined(_WIN64)
  #define PATH_SEPARATOR "\\"
#else
  #define PATH_SEPARATOR "/"
#endif

class DirectoryListing{
#ifdef _WIN32
	HANDLE h1;
	public:
		DirectoryListing() { h1 = NULL; }
		~DirectoryListing() { if(h1) FindClose(h1); }
#else
	DIR *dp;
	dirent *ep;
	char *pattern;
	public:
		DirectoryListing() { dp = NULL; ep = NULL; pattern = NULL; }
		~DirectoryListing() { if(dp) closedir(dp); if(pattern) delete[] pattern; }
#endif
		void First(char *result,const char *type);
		bool Next(char *result);
};

class DirUtil{
	public:
	static bool CreateDir(const std::string& name);
	static void RemoveExtension(char *out,char *str);
};
#endif
