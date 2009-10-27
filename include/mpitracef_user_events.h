C
C*****************************************************************************
C*                        ANALYSIS PERFORMANCE TOOLS                         *
C*                                  MPItrace                                 *
C*              Instrumentation package for parallel applications            *
C*****************************************************************************
C*                                                             ___           *
C*   +---------+     http:// www.cepba.upc.edu/tools_i.htm    /  __          *
C*   |    o//o |     http:// www.bsc.es                      /  /  _____     *
C*   |   o//o  |                                            /  /  /     \    *
C*   |  o//o   |     E-mail: cepbatools@cepba.upc.edu      (  (  ( B S C )   *
C*   | o//o    |     Phone:          +34-93-401 71 78       \  \  \_____/    *
C*   +---------+     Fax:            +34-93-401 25 77        \  \__          *
C*    C E P B A                                               \___           *
C*                                                                           *
C* This software is subject to the terms of the CEPBA/BSC license agreement. *
C*      You must accept the terms of this license to use this software.      *
C*                                 ---------                                 *
C*                European Center for Parallelism of Barcelona               *
C*                      Barcelona Supercomputing Center                      *
C*****************************************************************************
C
C* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
C| @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/include/mpitracef_user_events.h,v $
C| 
C| @last_commit: $Date: 2009/05/25 10:31:02 $
C| @version:     $Revision: 1.4 $
C| 
C| History:
C* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- 
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

	INTEGER MPITRACE_ENABLE_ALL_OPTIONS
	PARAMETER (MPITRACE_ENABLE_ALL_OPTIONS=127)

