cmake_minimum_required(VERSION 2.6)
project(sparselib)
file(GLOB SRC_CPP "src/*.cpp")
file(GLOB INCLUDES "include/*.h")

include_directories(.)
include_directories(${CLAPACK_INCLUDE_DIR})
link_directories( ${CLAPACK_LIBRARY_DIR})

add_library(sparselib ${SRC_CPP} ${INCLUDES})
target_link_libraries(sparselib ${ARPACK_LIBRARY} ${CLAPACK_LIBS} )

SET(SPARSELIB_INSTALL_INCLUDE_DIR "${ANURA_INSTALL_INCLUDE_DIR}/sparselib")
INSTALL(TARGETS sparselib
    RUNTIME DESTINATION ${ANURA_INSTALL_BIN_DIR} COMPONENT RuntimeLibraries
    LIBRARY DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT RuntimeLibraries
    ARCHIVE DESTINATION ${ANURA_INSTALL_LIB_DIR} COMPONENT Development)
	
INSTALL(FILES ${INCLUDES} 
	DESTINATION ${SPARSELIB_INSTALL_INCLUDE_DIR} COMPONENT Development)	
