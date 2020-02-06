#include <stdlib.h>
#include <stdio.h>

#include <GASPI_Threads.h>

#include "utils.h"

gaspi_rank_t numranks, myrank;

void work(int tid)
{
  gaspi_rank_t rankSend;
  gaspi_offset_t localOff = 81478066;
  gaspi_offset_t remOff   = 81478246;
  gaspi_offset_t size = 1800;
  gaspi_number_t queueSize, qmax;

  ASSERT (gaspi_queue_size_max(&qmax));

  for(rankSend = 0; rankSend < numranks; rankSend++)
    {
      gaspi_printf("thread %d rank to send: %d\n", tid, rankSend);

      gaspi_queue_size(1, &queueSize);
      if (queueSize > qmax - 24)
  	  ASSERT (gaspi_wait(1, GASPI_BLOCK));

      ASSERT (gaspi_write(0, localOff, rankSend, 0,  remOff,  size, 1, GASPI_BLOCK));
      
    }
  ASSERT (gaspi_wait(1, GASPI_BLOCK));

  gaspi_threads_sync();
}

void * thread_fun(void *args)
{
  gaspi_int tid;
  ASSERT(gaspi_threads_register(&tid));
  work(tid);

  return NULL;
}

int
main(int argc, char *argv[])
{
  int i;
  int num_threads = 0;
  gaspi_size_t segSize;

  TSUITE_INIT(argc, argv);
  ASSERT (gaspi_proc_init(GASPI_BLOCK));

  ASSERT(gaspi_threads_init(&num_threads));

  ASSERT (gaspi_proc_num(&numranks));
  ASSERT (gaspi_proc_rank(&myrank));

  ASSERT (gaspi_segment_create(0, _128MB, GASPI_GROUP_ALL, GASPI_BLOCK, GASPI_MEM_INITIALIZED));

  ASSERT( gaspi_segment_size(0, myrank, &segSize));

  for(i = 1; i < num_threads; i++)
    ASSERT(gaspi_threads_run(thread_fun, NULL));

  thread_fun(NULL);

  ASSERT (gaspi_barrier(GASPI_GROUP_ALL, GASPI_BLOCK));
  
  ASSERT (gaspi_proc_term(GASPI_BLOCK));

  return EXIT_SUCCESS;
}
