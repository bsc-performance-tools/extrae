#include "utils.h"
#include "common.h"

int main()
{
  int i, j, k, t;
  gaspi_rank_t myrank;

  //on numa architectures you have to map this process to the numa
  //node where nic is installed

  //if(gaspi_set_socket_affinity(1) != GASPI_SUCCESS){
  //printf("gaspi_set_socket_affinity failed !\n"); }

  if (start_bench (2) != 0)
  {
    printf ("Initialization failed\n");
    exit (-1);
  }

  // BENCH //
  gaspi_proc_rank (&myrank);

  gaspi_float cpu_freq;

  gaspi_cpu_frequency (&cpu_freq);

  if (myrank == 0)
  {
    printf ("--------------------------------------------------\n");
    printf ("%12s\t%5s\t\t%s\n", "Bytes", "BW", "MsgRate(Mpps)");
    printf ("--------------------------------------------------\n");

    int bytes = 2;

    for (i = 0; i < 15; i++)
    {
      for (j = 0; j < 10; j++)
      {
        stamp[j] = get_mcycles ();
        for (k = 0; k < 1000; k++)
        {
          gaspi_write (0, 0, 1, 0, 0, bytes, 0, GASPI_BLOCK);
        }

        gaspi_wait (0, GASPI_BLOCK);
        stamp2[j] = get_mcycles ();
      }

      for (t = 0; t < 10; t++)
      {
        delta[t] = stamp2[t] - stamp[t];
      }

      qsort (delta, 10, sizeof *delta, mcycles_compare);

      const double div = 1.0 / cpu_freq / (1000.0 * 1000.0);
      const double ts = (double) delta[5] * div;

      const double bw = (double) bytes / ts * 1000.0;
      const double bw_mb = bw / (1024.0 * 1024.0);
      const double rate = (double) 1000.0 / ts;

      printf ("%12d\t%4.2f\t\t%.4f\n", bytes, bw_mb, rate / 1e6);

      bytes <<= 1;
    }
  }

  end_bench ();

  return 0;
}
