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

#if defined(OS_LINUX) && defined(ARCH_IA64)

#include "ia64_clock.h"

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_LINUX_MMTIMER_H
# include <linux/mmtimer.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_SYS_MMAN_H
# include <sys/mman.h>
#endif
#ifdef HAVE_ASM_ERRNO_H
# include <asm/errno.h>
#endif
#ifdef HAVE_SYS_IOCTL_H
# include <sys/ioctl.h>
#endif

static unsigned long long proc_timebase_MHz;

static double proc_timebase (void)
{
  FILE *fp;
  char buffer[ 32768 ];
  size_t bytes_read;
  char* match;
  double temp;
  int res;

  fp = fopen( "/proc/cpuinfo", "r" );
  bytes_read = fread( buffer, 1, sizeof( buffer ), fp );
  fclose( fp );

  if (bytes_read == 0)
    return 0ULL;

  buffer[ bytes_read ] = '\0';
  match = strstr( buffer, "itc MHz" );
  if (match == NULL)
    return 0ULL;

  res = sscanf (match, "itc MHz    : %lf", &temp );

  return (res == 1) ? temp : 0.0f;
}

#if !defined(HAVE_LINUX_MMTIMER_H) && !defined(HAVE_MMTIMER_DEVICE)

void ia64_Initialize (void)
{
	proc_timebase_MHz = proc_timebase();
}

iotimer_t ia64_getTime (void)
{
  unsigned long long cycles;
  __asm__ __volatile__ ("mov %0=ar.itc" : "=r" (cycles));
  return (cycles * 1000) / proc_timebase_MHz; 
}

#else /* !defined(HAVE_LINUX_MMTIMER_H) && !defined(HAVE_MMTIMER_DEVICE) */

static unsigned long long mmdev_clicks_per_tick;
static volatile unsigned long *mmdev_timer_addr;

void ia64_Initialize (void)
{
  int fd;
  unsigned long long femtosecs_per_tick = 0;
  int offset;

  if ((fd = open("/dev/mmtimer", O_RDONLY)) == -1)
	{
    fprintf (stderr, PACKAGE_NAME": ERROR! Failed to open MM timer");
    return;
  }
  if ((offset = ioctl(fd, MMTIMER_GETOFFSET, 0)) != 0)
	{
    fprintf (stderr, PACKAGE_NAME": ERROR! Failed to get offset of MM timer");
    return;
  }
  if ((mmdev_timer_addr = mmap(0, getpagesize(), PROT_READ, MAP_SHARED, fd, 0)) == NULL)
	{
    fprintf (stderr, PACKAGE_NAME": ERROR! Failed to mmap MM timer");
    return;
  }

  mmdev_timer_addr += offset;
  ioctl(fd, MMTIMER_GETRES, &femtosecs_per_tick);

#if defined(DEBUG)
  fprintf (stdout, PACKAGE_NAME": MMDEV clock resolution is %llu ns\n", femtosecs_per_tick/1000000);
#endif

	proc_timebase_MHz = proc_timebase();

  mmdev_clicks_per_tick = proc_timebase_MHz * (femtosecs_per_tick / 1000000);

  close(fd);
}

iotimer_t ia64_getTime (void)
{
  return (mmdev_clicks_per_tick*(*mmdev_timer_addr))/proc_timebase_MHz;
}

#endif /* !defined(HAVE_LINUX_MMTIMER_H) && !defined(HAVE_MMTIMER_DEVICE) */

void ia64_Initialize_thread (void)
{
}

#endif
