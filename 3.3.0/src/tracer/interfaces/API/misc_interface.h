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

#ifndef MPI_INTERFACE_H_INCLUDED
#define MPI_INTERFACE_H_INCLUDED

#include <config.h>

/**** Create synonims of the very same routine using replication of code! ****/

#define EXPAND_ROUTINE_WITH_PREFIXES(x) \
   x(Extrae); \
   x(OMPItrace); \
   x(MPItrace); \
   x(OMPtrace); \
   x(SEQtrace);
#define EXPAND_F_ROUTINE_WITH_PREFIXES(x) \
   x(extrae); \
   x(ompitrace); \
   x(mpitrace); \
   x(omptrace); \
   x(seqtrace);

/**** Create synonims of the very same routine using 'alias' of the same routine (preferred) ****/

#define INTERFACE_ALIASES_C(base,orig,params,rettype) \
	rettype MPItrace##base params  __attribute__ ((alias (#orig))); \
	rettype OMPtrace##base params  __attribute__ ((alias (#orig))); \
	rettype OMPItrace##base params __attribute__ ((alias (#orig))); \
	rettype SEQtrace##base params  __attribute__ ((alias (#orig)));

#define INTERFACE_ALIASES_F(base_lo,base_up,orig,params,rettype) \
	rettype extrae##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo##__ params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype extrae##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo##_ params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype EXTRAE##base_up params  __attribute__ ((alias (#orig))); \
	rettype MPITRACE##base_up params  __attribute__ ((alias (#orig))); \
	rettype OMPTRACE##base_up params  __attribute__ ((alias (#orig))); \
	rettype OMPITRACE##base_up params __attribute__ ((alias (#orig))); \
	rettype SEQTRACE##base_up params  __attribute__ ((alias (#orig)));
#define INTERFACE_ALIASES_F_REUSE_C(base_lo,base_up,orig,params,rettype) \
	rettype extrae##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo##__ params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo##__ params  __attribute__ ((alias (#orig))); \
	rettype extrae##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo##_ params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo##_ params  __attribute__ ((alias (#orig))); \
	rettype mpitrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype omptrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype ompitrace##base_lo params __attribute__ ((alias (#orig))); \
	rettype seqtrace##base_lo params  __attribute__ ((alias (#orig))); \
	rettype EXTRAE##base_up params  __attribute__ ((alias (#orig))); \
	rettype MPITRACE##base_up params  __attribute__ ((alias (#orig))); \
	rettype OMPTRACE##base_up params  __attribute__ ((alias (#orig))); \
	rettype OMPITRACE##base_up params __attribute__ ((alias (#orig))); \
	rettype SEQTRACE##base_up params  __attribute__ ((alias (#orig))); \
	rettype extrae##base_lo params  __attribute__ ((alias (#orig)));  /* This is the extra to reuse C */

void Extrae_init (void);
void Extrae_fini (void);

#endif /* MPI_INTERFACE_H_INCLUDED */
