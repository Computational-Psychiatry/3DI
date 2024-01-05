add_definitions(-DKERNEL_RADIUS=7 -DWRITE_OBJ_FILE=0 -D_IGNORE_BOUNDS=0 )


cmake_minimum_required(VERSION 3.5)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

project(3DI LANGUAGES CXX CUDA)
enable_language("CUDA")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_path(CUDA_SDK_ROOT_DIR common/inc/cutil.h)


set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)


find_package( CUDA REQUIRED )
find_package( OpenCV REQUIRED )

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-use_fast_math)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories("${CUDA_INCLUDE_DIRS}")
include_directories( "./lib" )


file( GLOB LIB_SOURCES lib/*.cu )
file( GLOB LIB_HEADERS lib/*.h )

set(VISUALIZE_3D 1)

set(LIBS "${CUDA_LIBRARIES}" "${CUDA_CUBLAS_LIBRARIES}" "${OpenCV_LIBRARIES}" "${CUDA_cusolver_LIBRARY}" "stdc++fs")

if (${VISUALIZE_3D})
    set(LIBS ${LIBS} GL GLU glut)
    add_definitions(-DVISUALIZE_3D=1)
    set(LIBS ${LIBS})
    include(GNUInstallDirs)
endif(${VISUALIZE_3D})

add_library( open3DI ${LIB_SOURCES} ${LIB_HEADERS} )

set_property(TARGET open3DI PROPERTY CXX_STANDARD 14)
set_property(TARGET open3DI PROPERTY CUDA_ARCHITECTURES 61)


file( GLOB APP_SOURCES apps/*.cu )

foreach( testsourcefile ${APP_SOURCES} )
    string( REPLACE ".cu" "" testname ${testsourcefile} )
    get_filename_component(bn ${testname} NAME)
    
    add_executable( ${bn} ${testsourcefile})
    target_link_libraries( ${bn} open3DI ${LIBS})
    set_property(TARGET ${bn} PROPERTY CXX_STANDARD 14)
    set_property(TARGET ${bn} PROPERTY CUDA_ARCHITECTURES 61)
endforeach( testsourcefile ${APP_SOURCES} )

