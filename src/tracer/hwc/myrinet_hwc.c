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

#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_DLFCN_H
# include <dlfcn.h>
#endif
#include "myrinet_hwc.h"
#include "wrapper.h"

static void Generate_Myrinet_HWC_Labels(driver_t driver);

int Myrinet_Counters_Enabled = 0;
int Myrinet_Counters_Count   = 0;
int Myrinet_Routes_Enabled   = 0;
int Myrinet_Routes_Count     = 0;

driver_t Myrinet_Driver;

void (*Myrinet_start_counters)  (void);
int  (*Myrinet_num_counters)    (void);
void (*Myrinet_reset_counters)  (void);
int  (*Myrinet_read_counters)   (int, uint32_t *);
int  (*Myrinet_counters_labels) (char ***);

int  (*Myrinet_num_routes)  (void);
int  (*Myrinet_read_routes) (int, int, uint32_t *);

void Myrinet_HWC_Initialize(void)
{
#if !defined(IS_BG_MACHINE)
	const char * error, * error2;
	void * handle = NULL; 
	char * env_mpich_type;
	int num_loaded_libraries = 0;

	if (tracejant_network_hwc)
	{
		char libmyrinet_counters[1024];

		env_mpich_type = getenv("GMPI_ID");
		if (env_mpich_type != NULL)
		{
			fprintf(stdout, PACKAGE_NAME": Initializing Myrinet (GM) counters.\n");
			sprintf(libmyrinet_counters, "%s/lib/libgm_counters.so", trace_home);
			handle = dlopen (libmyrinet_counters, RTLD_LAZY);
			if (!handle)
			{
				fprintf (stderr, PACKAGE_NAME": Error! %s\n", dlerror());
				return;
			}
			Myrinet_Driver = GM;
			num_loaded_libraries ++;
		}

		env_mpich_type = getenv("MXMPI_ID");
		if (env_mpich_type != NULL) 
		{
			fprintf (stdout, PACKAGE_NAME": Initializing Myrinet (MX) counters.\n");
			sprintf (libmyrinet_counters, "%s/lib/libmx_counters.so", trace_home);
			handle = dlopen (libmyrinet_counters, RTLD_LAZY); 
			if (!handle) 
			{
				error = dlerror();
				if (strstr(error, "mxmpi") != NULL)
				{
					fprintf (stderr, PACKAGE_NAME": Error! Did you link the application with the flag -rdynamic or --export-dynamic?\n");
				}
				return;
			}
			Myrinet_Driver = MX;
			num_loaded_libraries ++;
		}
      if (num_loaded_libraries > 1)
      {
         fprintf(stderr, PACKAGE_NAME": Only one of GMMPI_ID or MXMPI_ID variables should be defined.\n");
         fprintf(stderr, PACKAGE_NAME": If you want GM counters unset MXMPI_ID, and vice-versa.\n");         return;
      }

		if (!handle) 
		{
			/* Neither MX or GM detected, disable counters */
			fprintf(stdout, PACKAGE_NAME": Unknown network. Network counters disabled.\n");
			return;
		}

		Myrinet_start_counters = (void(*)()) dlsym(handle, "MYRINET_start_counters");
		if ((error = dlerror()) != NULL)  
		{
            fprintf (stderr, PACKAGE_NAME": Error! %s\n", error);
            exit(1);
		}
		Myrinet_num_counters = (int(*)()) dlsym(handle, "MYRINET_num_counters");
		if ((error = dlerror()) != NULL)
		{
            fprintf (stderr, PACKAGE_NAME": Error! %s\n", error);
            exit(1);
		}
		Myrinet_reset_counters = (void(*)()) dlsym(handle, "MYRINET_reset_counters");
		if ((error = dlerror()) != NULL)  
		{
			fprintf (stderr, PACKAGE_NAME": Error! %s\n", error);
			exit(1);
		}
		Myrinet_read_counters = (int(*)(int,uint32_t *)) dlsym(handle, "MYRINET_read_counters");
		if ((error = dlerror()) != NULL)
		{
			fprintf (stderr, "%s\n", error);
			exit(1);
		}
		Myrinet_counters_labels = (int(*)(char ***)) dlsym(handle, "MYRINET_counters_labels");
		if ((error = dlerror()) != NULL)
		{
            fprintf (stderr, PACKAGE_NAME": Error! %s\n", error);
            exit(1);
		}

		if (Myrinet_Driver == MX)
		{
			Myrinet_num_routes = (int(*)()) dlsym(handle, "MYRINET_num_routes");
			error = dlerror();	
			Myrinet_read_routes = (int(*)(int,int,uint32_t *)) dlsym(handle, "MYRINET_read_routes");
			error2 = dlerror();
			if ((error == NULL) && (error2 == NULL))
			{
				Myrinet_Routes_Enabled = 1;
			}
			else
			{
				/* Routes counters not available */
			}
		}

		(*Myrinet_start_counters)();
		(*Myrinet_reset_counters)();

		Myrinet_Counters_Count = (*Myrinet_num_counters)();

		if (Myrinet_Routes_Enabled)
		{
			Myrinet_Routes_Count = (*Myrinet_num_routes)();
		}
		
		if (TASKID == 0)
		{
			Generate_Myrinet_HWC_Labels (Myrinet_Driver);
		}
		Myrinet_Counters_Enabled = 1;
	}
#endif /* IS_BG_MACHINE */
}

static void Generate_Myrinet_HWC_Labels(driver_t driver) 
{
#if !defined(IS_BG_MACHINE)
	char FileName[1024];
	int i;
	int num_labels;
	char ** labels;
	FILE * file;

	sprintf (FileName, "%s/myrinet_counters.pcf", final_dir);
	file = fopen (FileName, "w");
	if (file == NULL) 	
	{
		fprintf(stderr, PACKAGE_NAME": Error! Could not generate labels file for Myrinet counters.\n");
		return;
	}
	fprintf(file, "EVENT_TYPE\n");

	num_labels = (*Myrinet_counters_labels) (&labels);
	for (i=0; i<num_labels; i++)
	{
		if (strstr(labels[i], (const char *)"(Port 1)"))
		{
			continue;	
		}
		fprintf(file, "0   %d   %s\n", MYRINET_BASE_EV + Myrinet_Driver*100000 + i, labels[i]);
	}

	if (file) 
	{
		fclose(file);
	}
#endif /* IS_BG_MACHINE */
}
