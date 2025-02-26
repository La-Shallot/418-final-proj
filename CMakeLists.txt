cmake_minimum_required(VERSION 2.8.12.2)
project(Parellel_OpFlow)

set(CFLAGS
    -std=c++11
    -m64
    -O3
    -fopenmp=libomp
    # -g3
)

add_compile_options(${CFLAGS})

set(OpenCV_DIR /home/la-shallot/build/)
find_package(OpenCV REQUIRED)

include_directories(
    include
	${OpenCV_INCLUDE_DIRS}
)

set(ALL_LIBS
	${OpenCV_LIBS}
)

find_package(OpenMP)
if (OPENMP_FOUND)
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
   set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(${PROJECT_NAME}_SOURCES
    src/main.cpp
    src/MorevacCorner.cpp
    src/LkFlow.cpp
    src/Harris.cpp
    src/BothLk.cpp
)

# Main
add_executable(${PROJECT_NAME}_main ${${PROJECT_NAME}_SOURCES})
target_link_libraries(${PROJECT_NAME}_main ${ALL_LIBS})
