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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/src/common/mpit_header.c,v $
 | 
 | @last_commit: $Date: 2009/04/29 15:44:53 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: mpit_header.c,v 1.2 2009/04/29 15:44:53 harald Exp $";

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

