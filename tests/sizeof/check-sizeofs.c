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
 | @file: $Source: /home/paraver/cvs-tools/mpitrace/fusion/tests/sizeof/check-sizeofs.c,v $
 | 
 | @last_commit: $Date: 2008/01/26 11:18:22 $
 | @version:     $Revision: 1.2 $
 | 
 | History:
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
static char rcsid[] = "$Id: check-sizeofs.c,v 1.2 2008/01/26 11:18:22 harald Exp $";

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define PRINT_SIZE(t) \
  printf ("sizeof(%s) = %d\n", #t, sizeof(t));

/* These are the types checked in configure.ac */

int main (int argc, char *argv[])
{
  PRINT_SIZE(long)
  PRINT_SIZE(long long)
  PRINT_SIZE(char)
  PRINT_SIZE(int)
  PRINT_SIZE(off_t)
  PRINT_SIZE(ssize_t)
  PRINT_SIZE(void*)
}

