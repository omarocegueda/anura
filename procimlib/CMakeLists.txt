cmake_minimum_required(VERSION 2.6)
project(procimlib)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")

include_directories(.)
include_directories(${CLAPACK_INCLUDE_DIR})
link_directories(${CLAPACK_LIBRARY_DIR})
add_library(procimlib ${SRC_CPP} ${INCLUDES})
target_link_libraries(procimlib)

SET(PROCIMLIB_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/procimlib")
INSTALL(TARGETS procimlib
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${PROCIMLIB_INSTALL_INCLUDE_DIR} COMPONENT Development)	
