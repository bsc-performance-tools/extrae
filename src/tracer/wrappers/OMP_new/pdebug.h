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

#pragma once

// #define DEBUG

#include <omp.h>
#include "taskid.h"
#include "threadid.h"
#include "common.h"

#define MASTER_OUT(format, args...)                               \
{                                                                 \
  if (TASKID == 0)                                                \
	{                                                               \
		fprintf(stdout, PACKAGE_NAME ": " format,                     \
		   ## args);                                                  \
	}                                                               \
} 

#define MASTER_WARN(format, args...)                              \
{                                                                 \
  if (TASKID == 0)                                                \
	{                                                               \
		fprintf(stderr, PACKAGE_NAME ": [T:%d] %s: WARNING: " format, \
		   TASKID, __func__, ## args);                                \
	}                                                               \
} 


#define MASTER_ERROR(format, args...)                             \
{                                                                 \
  if (TASKID == 0)                                                \
	{                                                               \
		fprintf(stderr, PACKAGE_NAME ": [T:%d] %s: WARNING: " format, \
		   TASKID, __func__, ## args);                                \
	}                                                               \
} 

#define THREAD_DBG(format, args...)             \
{                                               \
  fprintf(stderr, PACKAGE_NAME                  \
          ": [T:%d THD:%d] %s: " format,        \
          TASKID, THREADID, __func__, ## args); \
}

#define THREAD_ERROR(format, args...)           \
{                                               \
  fprintf(stderr, PACKAGE_NAME                  \
          ": [T:%d THD:%d] %s: ERROR: " format, \
          TASKID, THREADID, __func__, ## args); \
}

#define THREAD_WARN(format, args...)              \
{                                                 \
  fprintf(stderr, PACKAGE_NAME                    \
          ": [T:%d THD:%d] %s: WARNING: " format, \
          TASKID, THREADID, __func__, ## args);   \
}

