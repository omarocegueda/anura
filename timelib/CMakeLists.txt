cmake_minimum_required(VERSION 2.6)
project(timelib)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")
include_directories(.)
add_library(timelib ${SRC_CPP} ${INCLUDES})
target_link_libraries(timelib )

SET(TIMELIB_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/timelib")
INSTALL(TARGETS timelib
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${TIMELIB_INSTALL_INCLUDE_DIR} COMPONENT Development)	
