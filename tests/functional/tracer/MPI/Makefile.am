include $(top_srcdir)/PATHS

check_PROGRAMS = \
 mpi_initfini_c_nolinked \
 mpi_initfini_c_linked \
 mpi_initfini_f \
 mpi_ranksize_f


CC=$(MPICC_COMPILER)
FC=$(MPIF90)

# TESTS = \
#  opencl-check.sh

mpi_initfini_c_nolinked_SOURCES = mpi_initfini_c.c

mpi_initfini_c_linked_SOURCES = mpi_initfini_c.c
mpi_initfini_c_linked_LDADD = $(TRACER_LIB)/libmpitrace.la

mpi_initfini_f_SOURCES = mpi_initfini_f.f90
mpi_ranksize_f_SOURCES = mpi_ranksize_f.f90


