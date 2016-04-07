C*****************************************************************************
C*                        ANALYSIS PERFORMANCE TOOLS                         *
C*                                   Extrae                                  *
C*              Instrumentation package for parallel applications            *
C*****************************************************************************
C*     ___     This library is free software; you can redistribute it and/or *
C*    /  __         modify it under the terms of the GNU LGPL as published   *
C*   /  /  _____    by the Free Software Foundation; either version 2.1      *
C*  /  /  /     \   of the License, or (at your option) any later version.   *
C* (  (  ( B S C )                                                           *
C*  \  \  \_____/   This library is distributed in hope that it will be      *
C*   \  \__         useful but WITHOUT ANY WARRANTY; without even the        *
C*    \___          implied warranty of MERCHANTABILITY or FITNESS FOR A     *
C*                  PARTICULAR PURPOSE. See the GNU LGPL for more details.   *
C*                                                                           *
C* You should have received a copy of the GNU Lesser General Public License  *
C* along with this library; if not, write to the Free Software Foundation,   *
C* Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA          *
C* The GNU LEsser General Public License is contained in the file COPYING.   *
C*                                 ---------                                 *
C*   Barcelona Supercomputing Center - Centro Nacional de Supercomputacion   *
C*****************************************************************************

	INTEGER EXTRAE_DISABLE_ALL_OPTIONS 
	PARAMETER (EXTRAE_DISABLE_ALL_OPTIONS=0)

	INTEGER EXTRAE_CALLER_OPTION 
	PARAMETER (EXTRAE_CALLER_OPTION=1)

	INTEGER EXTRAE_HWC_OPTION
	PARAMETER (EXTRAE_HWC_OPTION=2)  

	INTEGER EXTRAE_MPI_HWC_OPTION
	PARAMETER (EXTRAE_MPI_HWC_OPTION=4)

	INTEGER EXTRAE_MPI_OPTION
	PARAMETER (EXTRAE_MPI_OPTION=8)

	INTEGER EXTRAE_OMP_OPTION
	PARAMETER (EXTRAE_OMP_OPTION=16)

	INTEGER EXTRAE_OMP_HWC_OPTION
	PARAMETER (EXTRAE_OMP_HWC_OPTION=32)

	INTEGER EXTRAE_UF_HWC_OPTION
	PARAMETER (EXTRAE_UF_HWC_OPTION=64)

	INTEGER EXTRAE_SAMPLING_OPTION
	PARAMETER (EXTRAE_SAMPLING_OPTION=128)

	INTEGER EXTRAE_ENABLE_ALL_OPTIONS
	PARAMETER (EXTRAE_ENABLE_ALL_OPTIONS=255)

  INTEGER EXTRAE_NOT_INITIALIZED
  PARAMETER (EXTRAE_NOT_INITIALIZED=0)

  INTEGER EXTRAE_INITIALIZED_EXTRAE_INIT
  PARAMETER (EXTRAE_INITIALIZED_EXTRAE_INIT=1)

  INTEGER EXTRAE_INITIALIZED_MPI_INIT
  PARAMETER (EXTRAE_INITIALIZED_MPI_INIT=2)

  INTEGER EXTRAE_INITIALIZED_SHMEM_INIT
  PARAMETER (EXTRAE_INITIALIZED_SHMEM_INIT=3)
C
C
C OLD CONSTANT NAMING - maintained temporary, consider them deprecated!
C
C

	INTEGER MPITRACE_DISABLE_ALL_OPTIONS 
	PARAMETER (MPITRACE_DISABLE_ALL_OPTIONS=0)

	INTEGER MPITRACE_CALLER_OPTION 
	PARAMETER (MPITRACE_CALLER_OPTION=1)

	INTEGER MPITRACE_HWC_OPTION
	PARAMETER (MPITRACE_HWC_OPTION=2)  

	INTEGER MPITRACE_MPI_HWC_OPTION
	PARAMETER (MPITRACE_MPI_HWC_OPTION=4)

	INTEGER MPITRACE_MPI_OPTION
	PARAMETER (MPITRACE_MPI_OPTION=8)

	INTEGER MPITRACE_OMP_OPTION
	PARAMETER (MPITRACE_OMP_OPTION=16)

	INTEGER MPITRACE_OMP_HWC_OPTION
	PARAMETER (MPITRACE_OMP_HWC_OPTION=32)

	INTEGER MPITRACE_UF_HWC_OPTION
	PARAMETER (MPITRACE_UF_HWC_OPTION=64)

	INTEGER MPITRACE_SAMPLING_OPTION
	PARAMETER (MPITRACE_SAMPLING_OPTION=128)

	INTEGER MPITRACE_ENABLE_ALL_OPTIONS
	PARAMETER (MPITRACE_ENABLE_ALL_OPTIONS=255)

