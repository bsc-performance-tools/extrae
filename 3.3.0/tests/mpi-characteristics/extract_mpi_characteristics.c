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

#include <config.h>

#include <mpi.h>

#if USE_HARDWARE_COUNTERS

# ifdef HAVE_BGL_PERFCTR_EVENTS_H
#  include <bgl_perfctr_events.h>
# endif
# ifdef HAVE_BGL_PERFCTR_H
#  include <bgl_perfctr.h>
# endif

#include <papi.h>
#include "queue.h"
#include "map_clock.h"
#include "wrapper.h"
#include "HardwareCounters.h"

void HWC_PARAVER_Labels (FILE * pcfFD)
{
  int rc, ii;
  long long cnt;
  const PAPI_preset_info_t *myinfo;
  long long HWCEvtType;
  CntQueue *queue;
  CntQueue *ptmp;
  int EvCnt;
  int primer_comptador = 1;

  BGL_PERFCTR_event_num_t event;

  /*
   * Inicialitzacio PAPI 
   */
  rc = PAPI_library_init (PAPI_VER_CURRENT);
  if (rc != PAPI_VER_CURRENT)
  {
    if (rc > 0)
      fprintf (stderr, "PAPI library version mismatch!\n");
    fprintf (stderr, "Can't use hardware counters!\n");
    fprintf (stderr, "  Papi library error: %s\n", PAPI_strerror (rc));
    if (rc == PAPI_ESYS)
      perror ("  The error is:");
    return;
  }
  fprintf (pcfFD, "labels_mpi#ifdef COUNTERS_INFO\n");
  fprintf (pcfFD, "labels_mpistruct counters_info {\n");
  fprintf (pcfFD, "labels_mpi   long long ci_type;\n");
  fprintf (pcfFD, "labels_mpi   char *    ci_descr;\n");
  fprintf (pcfFD, "labels_mpi   char *    ci_label;\n");
  fprintf (pcfFD, "labels_mpi   long long ci_num;\n");
  fprintf (pcfFD, "labels_mpi   int       ci_used;\n");
  fprintf (pcfFD, "labels_mpi};\n");

  fprintf (pcfFD, "labels_mpi#define NUM_HWC %d\n",
           PAPI_MAX_PRESET_EVENTS + BGL_PERFCTR_NUMEVENTS);

  fprintf (pcfFD, "labels_mpi"
           "static struct counters_info cnts_info[NUM_HWC] = {\n");
  myinfo = PAPI_query_all_events_verbose ();
  event = 0;
  if (myinfo != NULL)
  {
    while (event < PAPI_MAX_PRESET_EVENTS)
    {
      fprintf (pcfFD, "labels_mpi{%d,\"%s\",\"%s\",0x%lx,0},",
               HWC_BASE + event,
               myinfo[event].event_descr,
               myinfo[event].event_name, myinfo[event].event_code);
      if (!myinfo[event].avail)
        fprintf (pcfFD, " /* Not available */");
      fprintf (pcfFD, "\n");
      ++event;
    }
  }
  event = 0;
  while (event < BGL_PERFCTR_NUMEVENTS)
  {
    int evcode;
    char *desc;

#if 0
    printf ("Event %d: num %d encodings %d mapping %llx, name %s descr %s\n",
            event,
            BGL_PERFCTR_event_table[event].event_num,
            BGL_PERFCTR_event_table[event].num_encodings,
            BGL_PERFCTR_event_table[event].mapping,
            BGL_PERFCTR_event_table[event].event_name,
            BGL_PERFCTR_event_table[event].event_descr);
#endif
    /*fprintf (pcfFD, "labels_mpi{%d,\"%s\",\"%s\",0x%llx,0},\n", */
    fprintf (pcfFD, "labels_mpi{%d,\"%s\",\"%s\",0x%x,0},\n",
             HWC_BASE + PAPI_MAX_PRESET_EVENTS + event,
             BGL_PERFCTR_event_table[event].event_descr,
             BGL_PERFCTR_event_table[event].event_name,
             BGL_PERFCTR_event_table[event].event_num);
    ++event;
  }
  fprintf (pcfFD, "labels_mpi};\n");
  fprintf (pcfFD, "labels_mpi#endif\n");
}
#endif /* USE_HARDWARE_COUNTERS */

#define BASE_OP_VALUE   50000000

int main (int argc, char *argv[])
{
  printf
    ("labels_mpi/* This file is generated automatically. Do not modify it */\n");
  printf ("labels_mpi#define MPI_MAX %d\n", MPI_MAX);
  printf ("labels_mpi#define MPI_MIN %d\n", MPI_MIN);
  printf ("labels_mpi#define MPI_SUM %d\n", MPI_SUM);
  printf ("labels_mpi#define MPI_PROD %d\n", MPI_PROD);
  printf ("labels_mpi#define MPI_LAND %d\n", MPI_LAND);
  printf ("labels_mpi#define MPI_BAND %d\n", MPI_BAND);
  printf ("labels_mpi#define MPI_LOR %d\n", MPI_LOR);
  printf ("labels_mpi#define MPI_BOR %d\n", MPI_BOR);
  printf ("labels_mpi#define MPI_LXOR %d\n", MPI_LXOR);
  printf ("labels_mpi#define MPI_BXOR %d\n", MPI_BXOR);
  printf ("labels_mpi#define MPI_MAXLOC %d\n", MPI_MAXLOC);
  printf ("labels_mpi#define MPI_MINLOC %d\n", MPI_MINLOC);
  printf ("labels_mpi#define MPI_MINLOC %d\n", MPI_MINLOC);

  printf ("\n");

  printf ("labels_mpi#define MPI_NONROOT_MAX %d\n",
          (int) MPI_MAX + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_MIN %d\n",
          (int) MPI_MIN + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_SUM %d\n",
          (int) MPI_SUM + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_PROD %d\n",
          (int) MPI_PROD + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_LAND %d\n",
          (int) MPI_LAND + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_BAND %d\n",
          (int) MPI_BAND + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_LOR %d\n",
          (int) MPI_LOR + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_BOR %d\n",
          (int) MPI_BOR + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_LXOR %d\n",
          (int) MPI_LXOR + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_BXOR %d\n",
          (int) MPI_BXOR + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_MAXLOC %d\n",
          (int) MPI_MAXLOC + BASE_OP_VALUE);
  printf ("labels_mpi#define MPI_NONROOT_MINLOC %d\n",
          (int) MPI_MINLOC + BASE_OP_VALUE);

  printf ("\n");

  printf ("labels_mpi#define MPI_COMM_WORLD %d\n", MPI_COMM_WORLD);
  printf ("labels_mpi#define MPI_COMM_SELF %d\n", MPI_COMM_SELF);
  printf ("labels_mpi#define MPI_COMM_NULL %d\n", MPI_COMM_NULL);

#if USE_HARDWARE_COUNTERS
  HWC_PARAVER_Labels (stdout);
#endif

  return 0;
}
