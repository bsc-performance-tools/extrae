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

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include "mpit_header.h"
#include "utils.h"

MPIT_Header_t * new_MPIT_Header()
{
	MPIT_Header_t * new_header = NULL;

	xmalloc(new_header, sizeof(MPIT_Header_t));
	return new_header;
}

void free_MPIT_Header(MPIT_Header_t * header)
{
	xfree(header);
}

void MPIT_Header_Write(int fd, MPIT_Header_t * header)
{
	off_t prev_offset;

	if (header != NULL)
	{
		prev_offset = lseek(fd, 0, SEEK_CUR);
		lseek(fd, 0, SEEK_SET);
		write(fd, header, sizeof(MPIT_Header_t));
		lseek(fd, prev_offset, SEEK_SET);
	}
}

MPIT_Header_t * MPIT_Header_Read(int fd)
{
	off_t prev_offset;
	MPIT_Header_t * header = new_MPIT_Header(); 

	prev_offset = lseek(fd, 0, SEEK_CUR);
	lseek(fd, 0, SEEK_SET);
	read(fd, header, sizeof(MPIT_Header_t));
	lseek(fd, prev_offset, SEEK_SET);

	return header;
}

