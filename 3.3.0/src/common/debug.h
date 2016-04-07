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

#ifndef __DEBUG_H__
#define __DEBUG_H__

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_ERRNO_H
# include <errno.h>
#endif

/* To ensure certain conditions at any point in the code */
#define ASSERT(condition, message) {                               \
   if (!(condition)) {                                             \
      fprintf (stderr, PACKAGE_NAME ": ASSERTION FAILED on %s [%s:%d]\n" \
                       PACKAGE_NAME ": CONDITION:   %s\n" \
                       PACKAGE_NAME ": DESCRIPTION: %s\n", \
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
   fprintf (stderr, PACKAGE_NAME ": %s on %s [%s:%d]\n"            \
                    PACKAGE_NAME ": DESCRIPTION: %s\n",            \
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
   PRINT_PRETTY_ERROR("WARNING", message);     \
}         

#endif /* __DEBUG_H__ */
