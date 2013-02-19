/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                   Extrae                                  *
 *              Instrumentation package for parallel applications            *
 *****************************************************************************
 *     ___     This library is free software; you can redistribute it and/or *
 *    /  __         modify it under the terms of the GNU LGPL as published   *
 *   /  /  _____    by the Free Software Foundation; either version 2.1      *
 *  /  /  /     \   of the License, or (at your option) any later version.   *
 * (  (  ( B S C )                                                           *
 *  \  \  \_____/   This library is distributed in hope that it will be      *
 *   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
 *    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
 *                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
 *                                                                           *
 * You should have received a copy of the GNU Lesser General Public License  *
 * along with this library; if not, write to the Free Software Foundation,   *
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
 * The GNU LEsser General Public License is contained in the file COPYING.   *
 *                                 ---------                                 *
 *   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
\*****************************************************************************/

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL$
 | @last_commit: $Date$
 | @version:     $Revision$
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef MPI_INTERFACE_H_INCLUDED
#define MPI_INTERFACE_H_INCLUDED

#include <config.h>

/**** Create synonims of the very same routine using replication of code! ****/

#if !defined(IS_CELL_MACHINE)
# define EXPAND_ROUTINE_WITH_PREFIXES(x) \
   x(OMPItrace); \
   x(MPItrace); \
   x(OMPtrace); \
   x(SEQtrace);
# define EXPAND_F_ROUTINE_WITH_PREFIXES(x) \
   x(ompitrace); \
   x(mpitrace); \
   x(omptrace); \
   x(seqtrace);
#else
# define EXPAND_ROUTINE_WITH_PREFIXES(x) \
   x(OMPItrace); \
   x(MPItrace); \
   x(OMPtrace); \
   x(SEQtrace); \
   x(PPUtrace);
# define EXPAND_F_ROUTINE_WITH_PREFIXES(x) \
   x(ompitrace); \
   x(mpitrace); \
   x(omptrace); \
   x(seqtrace); \
   x(pputrace);
#endif

/**** Create synonims of the very same routine using 'alias' of the same routine (preferred) ****/

#if !defined(IS_CELL_MACHINE)
# define INTERFACE_ALIASES_C(base,orig,params) \
  void MPItrace##base params  __attribute__ ((alias (#orig))); \
  void OMPtrace##base params  __attribute__ ((alias (#orig))); \
  void OMPItrace##base params __attribute__ ((alias (#orig))); \
	void SEQtrace##base params  __attribute__ ((alias (#orig)));
#else
# define INTERFACE_ALIASES_C(base,orig,params) \
  void MPItrace##base params  __attribute__ ((alias (#orig))); \
  void OMPtrace##base params  __attribute__ ((alias (#orig))); \
  void OMPItrace##base params __attribute__ ((alias (#orig))); \
	void SEQtrace##base params  __attribute__ ((alias (#orig))); \
	void PPUtrace##base params  __attribute__ ((alias (#orig))); 
#endif

#if !defined(IS_CELL_MACHINE)
# define INTERFACE_ALIASES_F(base_lo,base_up,orig,params) \
  void extrae##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo##__ params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void extrae##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo##_ params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo params  __attribute__ ((alias (#orig))); \
  void EXTRAE##base_up params  __attribute__ ((alias (#orig))); \
  void MPITRACE##base_up params  __attribute__ ((alias (#orig))); \
  void OMPTRACE##base_up params  __attribute__ ((alias (#orig))); \
  void OMPITRACE##base_up params __attribute__ ((alias (#orig))); \
  void SEQTRACE##base_up params  __attribute__ ((alias (#orig)));
#else
# define INTERFACE_ALIASES_F(base_lo,base_up,orig,params) \
  void extrae##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo##__ params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void pputrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
  void extrae##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo##_ params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void pputrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
  void mpitrace##base_lo params  __attribute__ ((alias (#orig))); \
  void omptrace##base_lo params  __attribute__ ((alias (#orig))); \
  void ompitrace##base_lo params __attribute__ ((alias (#orig))); \
  void seqtrace##base_lo params  __attribute__ ((alias (#orig))); \
  void pputrace##base_lo params  __attribute__ ((alias (#orig))); \
  void EXTRAE##base_up params  __attribute__ ((alias (#orig))); \
  void MPITRACE##base_up params  __attribute__ ((alias (#orig))); \
  void OMPTRACE##base_up params  __attribute__ ((alias (#orig))); \
  void OMPITRACE##base_up params __attribute__ ((alias (#orig))); \
  void SEQTRACE##base_up params  __attribute__ ((alias (#orig))); \
  void PPUTRACE##base_up params  __attribute__ ((alias (#orig)));
#endif

#endif /* MPI_INTERFACE_H_INCLUDED */
