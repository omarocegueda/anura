#ifndef CONFIGMANAGER_H
#define CONFIGMANAGER_H

struct ConfigField{
	const char *name;
	void *data;
	enum FieldType{CFFT_String, CFFT_Int, CFFT_Bool, CFFT_Double} type;
};

class ConfigManager{
	protected:
		ConfigField *vars;
		int nvars;
	public:
		ConfigManager();
		ConfigManager(ConfigField *_vars, int _nvars);
		void loadFromFile(const char *fname);
		void saveToFile(const char *fname);
		ConfigField *getField(const char *name);
		void showConfig(void);
		~ConfigManager();
};
#endif
