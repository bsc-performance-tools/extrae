
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

#ifndef CUDA_WRAPPER_CUPTI_H_
#define CUDA_WRAPPER_CUPTI_H_

#if CUPTI_API_VERSION > 12
typedef struct cudaConfigureCall_v3020_params_st {
	dim3 gridDim;
	dim3 blockDim;
	size_t sharedMem;
	cudaStream_t stream;
} cudaConfigureCall_v3020_params;

typedef struct cudaStreamDestroy_v3020_params_st {
    cudaStream_t stream;
} cudaStreamDestroy_v3020_params;
#endif /* CUPTI_API_VERSION > 12 */

void Extrae_CUDA_init (int rank);

#endif

