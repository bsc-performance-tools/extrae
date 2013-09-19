include ../../Makefile.inc

ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-std=c++11 -std=gnu++11 -g -I$(OPENCL_HOME)/include -O3  -Wall
INC = -I Cpp_common
LIBS = $(OPENCL_LDFLAGS) -lOpenCL -lrt

# Check our platform and make sure we define the APPLE variable
# and set up the right compiler flags and libraries
PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	CCFLAGS += -DAPPLE
	LIBS = -framework OpenCL
endif

vadd: vadd.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -fr vadd set-0 TRACE.mpits TRACE.sym
