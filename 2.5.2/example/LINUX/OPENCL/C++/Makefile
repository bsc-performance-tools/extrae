include ../../Makefile.inc

ifndef CPPC
	CPPC=g++
endif

CCFLAGS=-std=c++11 -std=gnu++11 -g -I$(OPENCL_HOME)/include -O3  -Wall
INC = -I Cpp_common

PLATFORM = $(shell uname -s)
ifeq ($(PLATFORM), Darwin)
	LIBS = -framework OpenCL
else
	LIBS = $(OPENCL_LDFLAGS) -lOpenCL -lrt
endif

vadd: vadd.cpp
	$(CPPC) $^ $(INC) $(CCFLAGS) $(LIBS) -o $@

clean:
	rm -fr vadd set-0 TRACE.mpits TRACE.sym
