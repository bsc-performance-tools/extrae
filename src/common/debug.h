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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/debug.h,v $
 | 
 | @last_commit: $Date: 2009/05/25 10:31:02 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef __DEBUG_H__
#define __DEBUG_H__

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

/* To ensure certain conditions at any point in the code */
#define ASSERT(condition, message) {                               \
   if (!(condition)) {                                             \
      fprintf (stderr, "ASSERTION FAILED on %s [%s:%d]\n%s\n%s\n", \
               __FUNCTION__,                                       \
               __FILE__,                                           \
               __LINE__,                                           \
               #condition,                                         \
               message);                                           \
      exit (-1);                                                   \
   }                                                               \
}
                  
#define PRINT_PRETTY_ERROR(severity, message)                      \
{                                                                  \
   fprintf (stderr, "%s on %s [%s:%d]\n%s\n",                      \
            severity,                                              \
            __FUNCTION__,                                          \
            __FILE__,                                              \
            __LINE__,                                              \
            message);                                              \
}

#define ERROR(message)                         \
{                                              \
   PRINT_PRETTY_ERROR("ERROR", message);       \
}

#define PERROR(message)                        \
{                                              \
   ERROR(message);                             \
   perror ("E");                               \
}

#define FATAL_ERROR(message)                   \
{                                              \
   PRINT_PRETTY_ERROR("FATAL ERROR", message); \
   exit (-1);                                  \
}

#define FATAL_PERROR(message)                  \
{                                              \
   PRINT_PRETTY_ERROR("FATAL ERROR", message); \
   perror ("E");                               \
   exit (-1);                                  \
}

#define WARNING(message)                       \
{                                              \
   fprintf(stderr, "WARNING: %s\n", message);  \
}         

#endif /* __DEBUG_H__ */
