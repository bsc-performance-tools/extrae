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
