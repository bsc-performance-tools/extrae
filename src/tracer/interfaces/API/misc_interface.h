/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                  MPItrace                                 *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *                                                             ___           *
 *   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
 *   |    o//o |     http:// www.bsc.es                      /  /  _____     *
 *   |   o//o  |                                            /  /  /     \    *
 *   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
 *   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
 *   +---------+     Fax:            +34-93-401 25 77        \  \__          *
 *    C E P B A                                               \___           *
 *                                                                           *
 * This software is subject to the terms of the CEPBA/BSC license agreement. *
 *      You must accept the terms of this license to use this software.      *
 *                                 ---------                                 *
 *                European Center for Parallelism of Barcelona               *
 *                      Barcelona Supercomputing Center                      *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/tracer/interfaces/API/misc_interface.h,v $
 | 
 | @last_commit: $Date: 2007/11/28 14:38:18 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MPI_INTERFACE_H_INCLUDED
#define MPI_INTERFACE_H_INCLUDED

#include <config.h>

/**** Create synonims of the very same routine using replication of code! ****/

#if !defined(IS_CELL_MACHINE)
# define EXPAND_ROUTINE_WITH_PREFIXES(x) \
   x(OMPI); \
   x(MPI); \
   x(OMP); \
   x(SEQ);
# define EXPAND_F_ROUTINE_WITH_PREFIXES(x) \
   x(ompi); \
   x(mpi); \
   x(omp); \
   x(seq);
#else
# define EXPAND_ROUTINE_WITH_PREFIXES(x) \
   x(OMPI); \
   x(MPI); \
   x(OMP); \
   x(SEQ); \
   x(PPU);
# define EXPAND_F_ROUTINE_WITH_PREFIXES(x) \
   x(ompi); \
   x(mpi); \
   x(omp); \
   x(seq); \
   x(ppu);
#endif

/**** Create synonims of the very same routine using 'alias' of the same routine (preferred) ****/

#if !defined(IS_CELL_MACHINE)
# define INTERFACE_ALIASES_C(base,orig,params) \
  void OMP##base params  __attribute__ ((alias (#orig))); \
  void OMPI##base params __attribute__ ((alias (#orig))); \
	void SEQ##base params  __attribute__ ((alias (#orig)));
#else
# define INTERFACE_ALIASES_C(base,orig,params) \
  void OMP##base params  __attribute__ ((alias (#orig))); \
  void OMPI##base params __attribute__ ((alias (#orig))); \
	void SEQ##base params  __attribute__ ((alias (#orig))); \
	void PPU##base params  __attribute__ ((alias (#orig))); 
#endif

#if !defined(IS_CELL_MACHINE)
# define INTERFACE_ALIASES_F(base_lo,base_up,orig,params) \
  void mpi##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo##__ params __attribute__ ((alias (#orig))); \
  void seq##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void mpi##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo##_ params __attribute__ ((alias (#orig))); \
  void seq##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo params __attribute__ ((alias (#orig))); \
  void seq##base_lo params  __attribute__ ((alias (#orig))); \
  void MPI##base_up params  __attribute__ ((alias (#orig))); \
  void OMP##base_up params  __attribute__ ((alias (#orig))); \
  void OMPI##base_up params __attribute__ ((alias (#orig))); \
  void SEQ##base_up params  __attribute__ ((alias (#orig)));
#else
# define INTERFACE_ALIASES_F(base_lo,base_up,orig,params) \
  void mpi##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo##__ params __attribute__ ((alias (#orig))); \
  void seq##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void ppu##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void mpi##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo##_ params __attribute__ ((alias (#orig))); \
  void seq##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void ppu##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omp##base_lo params  __attribute__ ((alias (#orig))); \
  void ompi##base_lo params __attribute__ ((alias (#orig))); \
  void seq##base_lo params  __attribute__ ((alias (#orig))); \
  void ppu##base_lo params  __attribute__ ((alias (#orig))); \
  void MPI##base_up params  __attribute__ ((alias (#orig))); \
  void OMP##base_up params  __attribute__ ((alias (#orig))); \
  void OMPI##base_up params __attribute__ ((alias (#orig))); \
  void SEQ##base_up params  __attribute__ ((alias (#orig))); \
  void PPU##base_up params  __attribute__ ((alias (#orig)));
#endif

#endif /* MPI_INTERFACE_H_INCLUDED */
