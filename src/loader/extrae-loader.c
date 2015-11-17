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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <libgen.h>
#include <string.h>
#include <sys/stat.h>

char *extrae_home = NULL;

void preload_this(char *module)
{
  char *current_env = NULL;
  char new_env[16384];

  current_env = getenv("LD_PRELOAD");
  if (current_env == NULL)
  {
    snprintf(new_env, 16384, "LD_PRELOAD=%s/lib/%s", extrae_home, module);
  }
  else
  {
    snprintf(new_env, 16384, "LD_PRELOAD=%s:%s/lib/%s", current_env, extrae_home, module);
  }
  putenv(new_env);
}

void show_preload()
{
  char *current_env = NULL;
  char *module_list = NULL;
  char *token       = NULL;
  int   num_modules = 0;

  current_env = getenv("LD_PRELOAD");
  if (current_env != NULL)
  {
    module_list = strdup(current_env);
    token = strtok(module_list, ":");
    fprintf(stderr, "extrae-loader: The following modules will be loaded:\n");
    while (token) 
    {
      num_modules ++;
      char *library_name = basename(token);
      fprintf(stderr, "#%d: %s [%s]\n", num_modules, library_name, token);
      token = strtok(NULL, ":");
    }
    free( module_list );
  }
  else
  {
    fprintf(stderr, "extrae-loader: WARNING: Any module loaded!\n");
  }
}

int look_for_symbol(void *handle, char *symbol)
{
  void *symbol_ptr = NULL;
  char *error = NULL;

  fprintf(stderr, "extrae-loader: Looking for '%s'... ", symbol);
  symbol_ptr = dlsym(handle, symbol); 
  if ((error = dlerror()) != NULL)
  {
    fprintf(stderr, "no (%s)\n", error);
    return 0;
  }
  else
  {
    fprintf(stderr, "yes\n");
    return 1;
  }
}

int detect_extrae(void *handle)
{
  return look_for_symbol(handle, "Extrae_init");
}

#define MPI_SYMBOLS 10
int detect_mpi(void *handle)
{
  int i = 0;
  int found = 0;
  char *mpi_symbols[MPI_SYMBOLS] = { 
    "MPI_Init", "mpi_init", "mpi_init_", "mpi_init__", "MPI_INIT",  
    "MPI_Init_thread", "mpi_init_thread", "mpi_init_thread_", "mpi_init_thread__", "MPI_INIT_THREAD",  
  };

  while ((i<MPI_SYMBOLS) && (!found))
  {
    found = look_for_symbol(handle, mpi_symbols[i]);
    i ++;
  }
  return found;
}

#define OPENMP_SYMBOLS 3
int detect_openmp(void *handle)
{
  int i = 0;
  int found = 0;

  char *openmp_symbols[OPENMP_SYMBOLS] = {
    "_xlsmpParallelDoSetup_TPO", "__kmpc_fork_call", "GOMP_parallel_start" 
  };

  while ((i<OPENMP_SYMBOLS) && (!found))
  {
    found = look_for_symbol(handle, openmp_symbols[i]);
    i ++;
  }
  return found;
}

int detect_pthreads(void *handle)
{
  return look_for_symbol(handle, "pthread_create");
}

int detect_cuda(void *handle)
{
  return look_for_symbol(handle, "cudaLaunch");
}

int detect_opencl(void *handle)
{
  return look_for_symbol(handle, "clCreateBuffer");
}

void print_help()
{
  fprintf(stdout, "\nSYNTAX\n");
  fprintf(stdout, "  extrae-loader [OPTIONS] <binary> [args ...]\n\n");
  fprintf(stdout, "OPTIONS\n");

  fprintf(stdout, "\n");
}

int main(int argc, char *argv[])
{
  void  *handle      = NULL;
  char  *app         = NULL;
  struct stat sb;
  pid_t  child_pid;
  int    child_status;
  int    have_extrae   = 0;
  int    have_mpi      = 0;
  int    have_openmp   = 0;
  int    have_pthreads = 0;
  int    have_cuda     = 0;
  int    have_opencl   = 0;

  extrae_home = getenv("EXTRAE_HOME");
  if (extrae_home == NULL)
  {
    fprintf(stderr, "extrae-loader: ERROR: Environment variable EXTRAE_HOME is not set!\n");
    exit(EXIT_FAILURE);
  }

  if (argc != 2)
  {
    print_help();
    exit(EXIT_FAILURE);
  }

  app = argv[1];

  if (stat(app, &sb) != 0)
  {
    fprintf(stderr, "extrae-loader: ERROR: Can not find binary '%s'\n", app);
    exit(EXIT_FAILURE);
  }
  else if (!(sb.st_mode & S_IXUSR))
  {
    fprintf(stderr, "extrae-loader: ERROR: File '%s' is not an executable\n", app);
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "extrae-loader: Opening application binary '%s'... ", app);
  handle = dlopen(app, RTLD_LAZY);
  if (!handle)
  {
    fprintf(stderr, "error!\n%s\n", dlerror());
    exit(EXIT_FAILURE);
  }
  fprintf(stderr, "ok!\n");
  dlerror(); /* Clear any existing error */

  have_extrae = detect_extrae(handle);
  have_mpi = detect_mpi(handle);
  have_openmp = detect_openmp(handle);
  if (!have_openmp)
  {
    have_pthreads = detect_pthreads(handle); 
  }
  have_cuda = detect_cuda(handle);
  have_opencl = detect_opencl(handle);

  if (have_extrae == 0)
  {
    fprintf(stderr, "extrae-loader: Extrae core not detected\n");
    preload_this("libextrae-core.so");
  }
  if (have_mpi == 1)
  {
    fprintf(stderr, "extrae-loader: MPI detected\n");
    preload_this("libextrae-mpi.so");
  }
  if (have_openmp == 1)
  {
    fprintf(stderr, "extrae-loader: OpenMP detected\n");
    preload_this("libextrae-openmp.so");
  }

  show_preload();

  child_pid = fork();
  if (child_pid == 0) 
  {
    /* This is done by the child process. */
    execv ("/home/bsc41/bsc41127/tests/single-lib-extrae/app", NULL);
    /* If execv returns, it must have failed. */
    fprintf(stderr, "Unknown command\n");
    exit(0);
  }
  else 
  {
    pid_t tpid;
    /* This is run by the parent.  Wait for the child to terminate. */
    do 
    {
      tpid = wait(&child_status);
    } while(tpid != child_pid);

    return child_status;
  }
  return 0;
}
