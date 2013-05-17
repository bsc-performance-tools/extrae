#ifndef __ONLINE_BE_H__
#define __ONLINE_BE_H__

#include "OnlineConfig.h"

//#define ONLINE_DEBUG

#define FRONTEND_RANK(world_size) (world_size - 1) // Last MPI process runs the front-end

#if defined(ONLINE_DEBUG)
# define ONLINE_DBG(msg, args...) \
   fprintf(stderr, "[ONLINE %d%s] " msg, this_BE_rank, (I_am_root ? "R" : ""), ## args);
#else
# define ONLINE_DBG(msg, args...) { ; }
#endif

#define ONLINE_DBG_1 \
  if (I_am_root) ONLINE_DBG

typedef struct 
{
  char resources_file[128];
  char topology_file[128];
  int  num_backends;
  char attach_file[128];
} FE_thread_data_t;

typedef struct
{
  int  my_rank;
  char parent_hostname[128];
  int  parent_port;
  int  parent_rank;
} BE_thread_data_t;

#if defined(__cplusplus)
extern "C" {
#endif

int Online_Start(int rank, int world_size, char **node_list);
int Online_Stop();
int Generate_Topology(int world_size, char **node_list, char *resources_file, char *topology_file);
void * FE_main_loop(void *context);
void * BE_main_loop(void *context);

#if defined(__cplusplus)
}
#endif

#endif /* __ONLINE_BE_H__ */
