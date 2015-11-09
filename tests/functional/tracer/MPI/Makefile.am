include $(top_srcdir)/PATHS

check_PROGRAMS = \
 pass_argument_MPIRUN \
 mpi_initfini_c_nolinked \
 mpi_initfini_c_linked \
 mpi_initfini_f \
 mpi_commranksize_c \
 mpi_commranksize_f \
 mpi_barrier_c \
 mpi_bcast_c \
 mpi_reduce_c

if HAVE_MPI3
check_PROGRAMS += \
 mpi_ibarrier_c \
 mpi_ibcast_c \
 mpi_ireduce_c 
endif

CC=$(MPICC_COMPILER)
FC=$(MPIF90)

TESTS = \
  mpi_initfini_c_linked_1proc.sh \
  mpi_initfini_c_linked_2proc.sh \
  mpi_initfini_c_linked_4proc.sh \
  mpi_initfini_c_nolinked_1proc.sh \
  mpi_initfini_f_nolinked_1proc.sh \
  mpi_commranksize_c_1proc.sh \
  mpi_commranksize_f_1proc.sh \
  mpi_barrier_c_1proc.sh \
  mpi_bcast_c_1proc.sh \
  mpi_reduce_c_1proc.sh

if HAVE_MPI3
TESTS += \
  mpi_ibarrier_c_1proc.sh \
  mpi_ibcast_c_1proc.sh \
  mpi_ireduce_c_1proc.sh
endif

pass_argument_MPIRUN_SOURCES = pass_argument_MPIRUN.c
pass_argument_MPIRUN_CFLAGS = -DMPIRUN="\"$(MPIRUN)\""

mpi_initfini_c_nolinked_SOURCES = mpi_initfini_c.c

mpi_initfini_c_linked_SOURCES = mpi_initfini_c.c
mpi_initfini_c_linked_LDADD = $(TRACER_LIB)/libmpitrace.la

mpi_initfini_f_SOURCES = mpi_initfini_f.f90
mpi_commranksize_f_SOURCES = mpi_commranksize_f.f90
mpi_commranksize_c_SOURCES = mpi_commranksize_c.c

mpi_barrier_c_SOURCES = mpi_barrier_c.c
mpi_ibarrier_c_SOURCES = mpi_ibarrier_c.c

mpi_bcast_c_SOURCES = mpi_bcast_c.c
mpi_ibcast_c_SOURCES = mpi_ibcast_c.c

mpi_reduce_c_SOURCES = mpi_reduce_c.c
mpi_ireduce_c_SOURCES = mpi_ireduce_c.c
