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

#include "common.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif
#if HAVE_STDIO_H
# include <stdio.h>
#endif
#if HAVE_STRING_H
# include <string.h>
#endif
#if HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <list>
#include <string>
#include <iostream>
#include <fstream>

using namespace std; 

#include "commonSnippets.h"
#include "ompSnippets.h"

#include <BPatch_function.h>

void InstrumentCUDAruntime (ApplicationType *at, BPatch_image *appImage)
{
	UNREFERENCED_PARAMETER(at);

	wrapRoutine (appImage, "cudaLaunch",
	  "Extrae_cudaLaunch_Enter", "Extrae_cudaLaunch_Exit", 1);
	wrapRoutine (appImage, "cudaConfigureCall",
	  "Extrae_cudaConfigureCall_Enter", "Extrae_cudaConfigureCall_Exit", 4);
	wrapRoutine (appImage, "cudaStreamCreate",
	  "Extrae_cudaStreamCreate_Enter", "Extrae_cudaStreamCreate_Exit", 1);
	wrapRoutine (appImage, "cudaMemcpyAsync",
	  "Extrae_cudaMemcpyAsync_Enter", "Extrae_cudaMemcpyAsync_Exit", 5);
	wrapRoutine (appImage, "cudaMemcpy",
	  "Extrae_cudaMemcpy_Enter", "Extrae_cudaMemcpy_Exit", 4);
	wrapRoutine (appImage, "cudaThreadSynchronize",
	  "Extrae_cudaThreadSynchronize_Enter", "Extrae_cudaThreadSynchronize_Exit", 0);
	wrapRoutine (appImage, "cudaStreamSynchronize",
	  "Extrae_cudaStreamSynchronize_Enter", "Extrae_cudaStreamSynchronize_Exit", 1);
}

