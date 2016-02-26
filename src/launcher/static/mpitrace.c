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

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_SYS_WAIT_H
# include <sys/wait.h>
#endif
#ifdef HAVE_GETOPT_H
#include <getopt.h>
#endif

int mpi_counters_on = 0;
char *mpi_flush_signal = "";
char **shargs;
char *nom_aplicacio;
char variable_entorn[300];


#define NOM_PER_DEFECTE "Trace"

#ifdef OS_LINUX
static struct option long_options[] = {
  {"arch", 1, 0, 0},
  {"machine", 1, 0, 0},
  {"machinefile", 1, 0, 0},
  {"np", 1, 0, 0},
  {"nolocal", 0, 0, 0},
  {"stdin", 1, 0, 0},
  {"dbx", 0, 0, 0},
  {"gdb", 0, 0, 0},
  {"xxgdb", 0, 0, 0},
  {"tv", 0, 0, 0},
  {"batch", 0, 0, 0},
  {"stdout", 1, 0, 0},
  {"stderr", 1, 0, 0},
  {"nexuspg", 1, 0, 0},
  {"nexusdb", 1, 0, 0},
  {"pg", 0, 0, 0},
  {"leave_pg", 0, 0, 0},
  {"p4pg", 1, 0, 0},
  {"tcppg", 1, 0, 0},
  {"p4ssport", 1, 0, 0},
  {"mvhome", 0, 0, 0},
  {"mvback", 1, 0, 0},
  {"maxtime", 1, 0, 0},
  {"nopoll", 0, 0, 0},
  {"mem", 1, 0, 0},
  {"cpu", 1, 0, 0},
  {"cac", 1, 0, 0},
  {"paragontype", 1, 0, 0},
  {"paragonname", 1, 0, 0},
  {"paragonpn", 1, 0, 0},
  {0, 0, 0, 0}
};
#endif /* OS_LINUX */


void process_arguments (int argc, char *argv[])
{
#ifdef OS_DEC
  nom_aplicacio = NOM_PER_DEFECTE;

#elif defined (OS_LINUX)
  int c, option_index = 0;
  int i;
	      	
  c = getopt_long_only (argc, argv, "+htve", long_options, &option_index);
  while (c != -1)
  {
    // No cal consultar els arguments. Nomes cal processar-los per arribar
    // a obtenir el primer parametre que no es una opcio del mpirun. Es a dir,
    // el nom de l'aplicacio.
    c = getopt_long_only (argc, argv, "+htve", long_options, &option_index);
  }

  // S'agafa el nom de l'aplicacio a no ser que hi hagi hagut algun problema
  if (argc < optind + 1)
    nom_aplicacio = NOM_PER_DEFECTE;
  else
  {
    nom_aplicacio = argv[optind];
    // S'ha d'agafar nomes el nom, no els directoris
    i = strlen (nom_aplicacio) - 1;
    while ((i > 0) && (nom_aplicacio[i - 1] != '/'))
      i--;
    nom_aplicacio = &nom_aplicacio[i];
  }
#endif /* OS_LINUX */
}


void process_mpitrace_arguments (int argc, char *argv[])
{
	int i, index = 0, no_options = 0, parsed_parameter = 1;

	while (parsed_parameter)
	{
		printf ("parsing %s\n", argv[index]);
		parsed_parameter = 0;

		if (strcmp (argv[index], "-counters:mpi") == 0)
		{
			mpi_counters_on = 1;
			parsed_parameter = 1;
		}
		else if (strcmp (argv[index], "-counters:nompi") == 0)
		{
			mpi_counters_on = 0;
			parsed_parameter = 1;
		}
		else if (strcmp (argv[index], "-flush-signal:usr1") == 0)
		{
			mpi_flush_signal = "USR1";
			parsed_parameter = 1;
		}
		else if (strcmp (argv[index], "-flush-signal:usr2") == 0)
		{
			mpi_flush_signal = "USR2";
			parsed_parameter = 1;
		}

		index += parsed_parameter;
		no_options += parsed_parameter;
	}

	printf ("number of options = %d\n", no_options);

	shargs = &argv[no_options];

	for (i = index; i < argc; i++)
		printf ("%d - %s\n", i, argv[i]);

  process_arguments (argc-no_options, &argv[no_options]);
}


void setup_environment ()
{
  // S'indica que cal tracejar
  putenv ("EXTRAE_ON=1");

  sprintf (variable_entorn, "EXTRAE_PROGRAM_NAME=%s", nom_aplicacio);
  // Es dona el nom de l'aplicacio
  putenv (variable_entorn);

  // S'indica si cal generar comptadors a les rutines mpi
  if (mpi_counters_on)
    putenv ("EXTRAE_MPI_COUNTERS_ON=1");

	if (strcmp(mpi_flush_signal, "") != 0)
	{
		char *duplicat = malloc ((strlen(mpi_flush_signal)+32+1)*sizeof(char));
		sprintf (duplicat, "EXTRAE_SIGNAL_FLUSH_TERMINATE=%s", mpi_flush_signal);
		printf ("%s\n", duplicat);
		putenv (duplicat);
	}
}


void launch (char **shargs)
{
  int status;

  if (!vfork ())
  {
    execvp (shargs[0], shargs);
    perror (PACKAGE_NAME);
    exit (1);
  }
  wait (&status);
}


/*
 * Steps:
 * - look for envvars
 * - set defaults
 */

int main (int argc, char *argv[])
{
  /*
   * all the arguments are intended to feed the real shell 
   */
  if (argc > 1)
    shargs = &(argv[1]);
  else
  {
    // Cal mostrar l'ajuda i sortir
    fprintf (stderr, PACKAGE_NAME": Arguments needed!\n");
    shargs = NULL;
    exit (1);
  }

//  process_arguments(argc,argv);
  process_mpitrace_arguments (argc - 1, &argv[1]);
  
  setup_environment ();
	  
  launch (shargs);

  return 0;
}
