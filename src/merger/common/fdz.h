/*****************************************************************************\
 *                        ANALYSIS PERFORMANCE TOOLS                         *
 *                                 MPItrace                                  *
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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/merger/common/fdz.h,v $
 | 
 | @last_commit: $Date: 2009/05/26 14:10:06 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */

#ifndef FDZ_H
#define FDZ_H

#include <config.h>

#ifdef HAVE_ZLIB

# include <zlib.h>

struct fdz_fitxer
{
  FILE *handle;
  gzFile handleGZ;
};
#else
struct fdz_fitxer
{
  FILE *handle;
};
#endif

#ifdef HAVE_ZLIB
# define FDZ_CLOSE(x) \
	(x.handleGZ!=NULL)?gzclose(x.handleGZ):fclose(x.handle)
# define FDZ_WRITE(x,buffer) \
	(x.handleGZ!=NULL)?gzputs(x.handleGZ,buffer):fputs(buffer,x.handle)
# define FDZ_DUMP(x,buffer,size) \
  (x.handleGZ!=NULL)?:write(fileno(x.handle),buffer,size)
# define FDZ_FLUSH(x) \
	(x.handleGZ!=NULL)?gzflush(x.handleGZ,Z_FULL_FLUSH):fflush(x.handle)
# define FDZ_TELL(x) \
	(x.handleGZ!=NULL)?gztell(x.handleGZ):ftell(x.handle)
# define FDZ_SEEK_SET(x,offset) \
	(x.handleGZ!=NULL)?gzseek(x.handleGZ,offset,SEEK_SET):fseek(x.handle,offset,SEEK_SET)
#else
# define FDZ_CLOSE(x) fclose(x.handle)
# define FDZ_WRITE(x,buffer) fputs(buffer,x.handle)
# define FDZ_DUMP(x,buffer,size) write(fileno(x.handle),buffer,size)
# define FDZ_FLUSH(x) fflush(x.handle)
# define FDZ_TELL(x) ftell(x.handle)
# define FDZ_SEEK_SET(x,offset) fseek(x.handle,offset,SEEK_SET)
#endif

#endif
