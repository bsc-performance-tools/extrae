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

#ifndef DEFINES_DEFINED
#define DEFINES_DEFINED 

#if defined(HAVE_MPI)

# if defined(PMPI_SINGLE_UNDERSCORE)
#  define CtoF77(x) x ## _
# elif defined(PMPI_DOUBLE_UNDERSCORES)
#  define CtoF77(x) x ## __
# elif defined(PMPI_NO_UNDERSCORES)
#  define CtoF77(x) x
# elif defined(PMPI_UNDERSCORE_F_SUFFIX)
#  define CtoF77(x) x ## _f
# else
#  error Do not know how to define CtoF77!
# endif

# if defined(MPICH_NAME)
#  if !defined(MPICH)
#   define MPICH
#  endif
# elif defined(OPEN_MPI)
#  if !defined(OPENMPI)
#   define OPENMPI
#  endif
# endif

#else /* HAVE_MPI */

/* If we don't have MPI, rely on FC_FUNC if web have detected at configure time! */
# if defined(FC_FUNC)
#  define CtoF77(x) FC_FUNC(x,x)
# else
#  error "Error! Not defined FC_FUNC, how do we deal with Fortran symbols?"
# endif

#endif /* HAVE_MPI */

#endif
