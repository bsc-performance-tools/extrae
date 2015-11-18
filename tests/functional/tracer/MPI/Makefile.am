include $(top_srcdir)/PATHS

ALL_TESTS = \
  mpi_initfini_c_linked_1proc.sh \
  mpi_initfini_c_linked_2proc.sh \
  mpi_initfini_c_linked_4proc.sh \
  mpi_initfini_c_nolinked_1proc.sh \
  mpi_initfini_f_nolinked_1proc.sh \
  mpi_commranksize_c_1proc.sh \
  mpi_commranksize_f_1proc.sh \
  mpi_barrier_c_1proc.sh \
  mpi_bcast_c_1proc.sh \
  mpi_reduce_c_1proc.sh \
  mpi_gather_c_1proc.sh \
  mpi_scatter_c_1proc.sh \
  mpi_sendirecv_c.sh \
  mpi_isendirecv_c.sh \
  mpi_isendirecvwaitall_c.sh \
  mpi_sendirecvprobewait_c_2proc.sh \
  mpi_sendirecviprobewait_c_2proc.sh \
  mpi_sendirecvtestwait_c_2proc.sh

ALL_TESTS_MPI3 = \
  mpi_ibarrier_c_1proc.sh \
  mpi_ibcast_c_1proc.sh \
  mpi_ireduce_c_1proc.sh

EXTRA_DIST = \
 $(ALL_TESTS) \
 $(ALL_TESTS_MPI3) \
 trace-ldpreload.sh \
 trace-ldpreloadf.sh \
 trace-static.sh \
 extrae.xml \
 extrae-callers.xml

check_PROGRAMS = \
 pass_argument_MPIRUN \
 mpi_initfini_c_nolinked \
 mpi_initfini_c_linked \
 mpi_initfini_f \
 mpi_commranksize_c \
 mpi_commranksize_f \
 mpi_barrier_c \
 mpi_bcast_c \
 mpi_reduce_c \
 mpi_gather_c \
 mpi_scatter_c \
 mpi_sendirecv_c \
 mpi_isendirecv_c \
 mpi_isendirecvwaitall_c \
 mpi_sendirecvprobewait_c \
 mpi_sendirecviprobewait_c \
 mpi_sendirecvtestwait_c

if HAVE_MPI3
check_PROGRAMS += \
 mpi_ibarrier_c \
 mpi_ibcast_c \
 mpi_ireduce_c 
endif

CC=$(MPICC)
FC=$(MPIF90)

TESTS = $(ALL_TESTS)

if HAVE_MPI3
TESTS += $(ALL_TESTS_MPI3)
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
mpi_gather_c_SOURCES = mpi_gather_c.c
mpi_scatter_c_SOURCES = mpi_scatter_c.c

mpi_isendirecvwaitall_c_SOURCES = mpi_isendirecvwaitall_c.c
mpi_isendirecv_c_SOURCES = mpi_isendirecv_c.c
mpi_sendirecv_c_SOURCES = mpi_sendirecv_c.c
mpi_sendirecvprobewait_c_SOURCES = mpi_sendirecvprobewait_c.c
mpi_sendirecviprobewait_c_SOURCES = mpi_sendirecviprobewait_c.c
mpi_sendirecvtestwait_c_SOURCES = mpi_sendirecvtestwait_c.c
