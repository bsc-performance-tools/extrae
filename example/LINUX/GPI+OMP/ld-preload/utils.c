#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "utils.h"

#define BACKTRACE_SIZE 256

void
tsuite_do_backtrace (int id, gaspi_rank_t node, FILE * bt_file)
{
  void *array[BACKTRACE_SIZE];
  size_t size, i;
  char **strings;

  size = backtrace (array, BACKTRACE_SIZE);
  strings = backtrace_symbols (array, size);

  fprintf (bt_file,
           "************* BACKTRACE: Pid %d, Rank %d  *************** \n", id,
           node);

  for (i = 0; i < size; i++)
    fprintf (bt_file, "%s\n", strings[i]);

  fflush (bt_file);

  free (strings);               /*  malloced by backtrace_symbols */
}

void
tsuite_sighandler (int signum, siginfo_t * info, void * ptr)
{
  FILE *bt_file;
  pid_t ptid = syscall (__NR_gettid);
  gaspi_number_t initialized;
  gaspi_rank_t nodeRank = 0;

  gaspi_initialized (&initialized);
  if (initialized)
  {
    gaspi_proc_rank (&nodeRank);
  }

  /* Use stdout for now */
  bt_file = stdout;

  fprintf (bt_file, "Pid signal: pid %u signum %d\n", ptid, signum);
  fprintf (bt_file, "Signal %d originates from process %lu (rank %d)\n",
           info->si_signo, (unsigned long) info->si_pid, nodeRank);

  tsuite_do_backtrace (ptid, nodeRank, bt_file);

  exit (-1);
}

void
tsuite_init (int argc, char *argv[])
{
  /* Backtracing for debugging */
  struct sigaction act;

  memset (&act, 0, sizeof (act));
  act.sa_sigaction = tsuite_sighandler;
  act.sa_flags = SA_SIGINFO;

  sigaction (SIGABRT, &act, NULL);
  sigaction (SIGTERM, &act, NULL);
  sigaction (SIGFPE, &act, NULL);
  sigaction (SIGBUS, &act, NULL);
  sigaction (SIGSEGV, &act, NULL);
  sigaction (SIGIO, &act, NULL);
  sigaction (SIGHUP, &act, NULL);

  if (argc > 1)
  {
    gaspi_config_t config;
    ASSERT (gaspi_config_get (&config));

    for (int i = 1; i < argc; i++)
    {
      if (strcmp (argv[i], "GASPI_ETHERNET") == 0)
      {
        config.network = GASPI_ETHERNET;
      }
      if (strcmp (argv[i], "GASPI_IB") == 0)
      {
        config.network = GASPI_IB;
      }
      if (strcmp (argv[i], "GASPI_ROCE") == 0)
      {
        config.network = GASPI_ROCE;
      }
      if (strcmp (argv[i], "STATIC_TOPO") == 0)
      {
        config.build_infrastructure = GASPI_TOPOLOGY_STATIC;
      }
      if (strcmp (argv[i], "DYNAMIC_TOPO") == 0)
      {
        config.build_infrastructure = GASPI_TOPOLOGY_DYNAMIC;
      }
      if (strcmp (argv[i], "NO_TOPO") == 0)
      {
        config.build_infrastructure = GASPI_TOPOLOGY_NONE;
      }
      if (strcmp (argv[i], "SN_PERSIST_TRUE") == 0)
      {
        config.sn_persistent = 1;
      }
      if (strcmp (argv[i], "SN_PERSIST_FALSE") == 0)
      {
        config.sn_persistent = 0;
      }
    }
    ASSERT (gaspi_config_set (config));
  }
}

void
success_or_exit (const char *file, const int line, const gaspi_return_t ec)
{
  if (ec != GASPI_SUCCESS)
  {
    gaspi_printf ("Assertion failed in %s[%i]: Return %d: %s\n", file, line,
                  ec, gaspi_error_str (ec));
    exit (EXIT_FAILURE);
  }
}

void
must_fail (const char *file, const int line, const gaspi_return_t ec)
{
  if (ec == GASPI_SUCCESS || ec == GASPI_TIMEOUT)
  {
    gaspi_printf ("Non-expected success in %s[%i]\n", file, line);

    exit (EXIT_FAILURE);
  }
}

void
must_return_err (const char *file, const int line, const gaspi_return_t ec,
                 const gaspi_return_t err)
{
  if (ec != err)
  {
    gaspi_printf ("Expected %d(%s) but got %d(%s) in %s[%i]\n",
                  err, gaspi_error_str (err),
                  ec, gaspi_error_str (ec), file, line);

    exit (EXIT_FAILURE);
  }
}

void
exit_safely (void)
{
  gaspi_rank_t rank, nprocs, i;

  ASSERT (gaspi_proc_num (&nprocs));
  ASSERT (gaspi_proc_rank (&rank));

  if (rank == 0)
  {
    for (i = 1; i < nprocs; i++)
      ASSERT (gaspi_proc_kill (i, GASPI_BLOCK));
  }

  ASSERT (gaspi_proc_term (GASPI_BLOCK));
}
