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
#ifdef HAVE_BFD_H
# include <bfd.h>
#endif
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif

// #define DEBUG

#define SHORT_STRING_PREFIX 8
#define SHORT_STRING_SUFFIX 8
#define SHORT_STRING_INFIX  ".."

#include "utils.h"

#if defined(NEED_ERRNO_LOCATION_PATCH)
/* On some machines (BG/L for instance) this symbol is undefined but required by
   BFD library. */
int *__errno_location(void)
{
	return 0;
}   
#endif

#include "object_tree.h"
#include "bfd_manager.h"
#include "events.h"
#include "addr2info.h"
#include "labels.h"
#include "misc_prv_semantics.h"
#include "addr2info_hashcache.h"
#include "options.h"

static void AddressTable_Initialize (void);
static int  AddressTable_Insert (UINT64 address, int event_type,
	char *module, char *funcname, char *filename, int line);
#if defined(HAVE_BFD)
static int AddressTable_Insert_MemReference (int addr_type,
	const char *module, const char *staticname, const char *filename, int line);
#endif
#if defined(HAVE_BFD)
static void Translate_Address (UINT64 address, unsigned ptask, unsigned task,
	char **module, char ** funcname, char ** filename, int * line);
static void Translate_Address_Data (UINT64 address, unsigned ptask, unsigned task,
	char **symbol);
#endif

static struct address_table  * AddressTable  [COUNT_ADDRESS_TYPES]; /* Addresses translation table     */
static struct function_table * FunctionTable [COUNT_ADDRESS_TYPES]; /* Function name translation table */

static struct address_object_table_st AddressObjectInfo;

/* Addresses will be translated into function, file and line if set to TRUE */
static int Tables_Initialized = FALSE;
static int Translate_Addresses = FALSE;

/* Determine whether to write the translation tables into the PCF file */
#define A2I_MPI      0
#define A2I_OMP      1
#define A2I_UF       2
#define A2I_SAMPLE   3
#define A2I_CUDA     4
#define A2I_OTHERS   5
#define A2I_LAST     6
int Address2Info_Labels[A2I_LAST];

static int Address2Info_Sort_routine(const void *p1, const void *p2)
{
	struct address_info *a1 = (struct address_info*) p1;
	struct address_info *a2 = (struct address_info*) p2;

	/* Sort by filename, line and address */
	if (strcmp (a1->file_name, a2->file_name) == 0)
	{
		if (a1->line == a2->line)
		{
			if (a1->address == a2->address)
				return 0;
			else if (a1->address < a2->address)
				return -1;
			else
				return 1;
		}
		else if (a1->line < a2->line)
			return -1;
		else
			return 1;
	}
	else 
		return strcmp (a1->file_name, a2->file_name);
}

void Address2Info_Sort (int unique_ids)
{
	/* Sort identifiers and also skip UNRESOLVED and NOT_FOUND */
	if (unique_ids)
	{
		void *base = (void*) &(AddressTable[UNIQUE_TYPE]->address[2]);

		qsort (base, AddressTable[UNIQUE_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);
	}
	else
	{
		void *base = (void*) &(AddressTable[OUTLINED_OPENMP_TYPE]->address[2]);
		qsort (base, AddressTable[OUTLINED_OPENMP_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[MPI_CALLER_TYPE]->address[2]);
		qsort (base, AddressTable[MPI_CALLER_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[SAMPLE_TYPE]->address[2]);
		qsort (base, AddressTable[SAMPLE_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[USER_FUNCTION_TYPE]->address[2]);
		qsort (base, AddressTable[USER_FUNCTION_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[OTHER_FUNCTION_TYPE]->address[2]);
		qsort (base, AddressTable[OTHER_FUNCTION_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[CUDAKERNEL_TYPE]->address[2]);
		qsort (base, AddressTable[CUDAKERNEL_TYPE]->num_addresses-2,
			sizeof(struct address_info), Address2Info_Sort_routine);
	}

	/* Cached entries are now invalid as everything gets resorted */
	Addr2Info_HashCache_Clean();
}

/** Address2Info_Initialized
 *
 * Check if the translation system is correctly initialized.
 * 
 * @param None
 *
 * @return Returns if Address2Info_Initialize has been succesfully called.
 */
int Address2Info_Initialized (void)
{
	return Translate_Addresses;
}

/** Address2Info_Initialize
 *
 * Initialize the memory addresses translation module.
 * 
 * @param binary
 *
 * @return No return value.
 */
void Address2Info_Initialize (char * binary)
{
	int type;

	Translate_Addresses = FALSE;

	/* Initialize the memory addresses translation table */
	AddressTable_Initialize();

	/* Add Unresolved and Not Found references now, so they always get
	   the same ID */
	for (type = 0; type < COUNT_ADDRESS_TYPES; type++)
	{
		/* Add fake address 0x0 to link "Unresolved" functions */
		AddressTable_Insert (UNRESOLVED_ID, type, NULL, ADDR_UNRESOLVED, ADDR_UNRESOLVED, 0);

		/* Add fake address 0x1 to link "Address not found" functions */
		AddressTable_Insert (NOT_FOUND_ID, type, NULL, ADDR_NOT_FOUND, ADDR_NOT_FOUND, 0);
	}

#if defined(HAVE_BFD)
	/* Initialize BFD libraries */
	BFDmanager_init();

	if (binary != NULL)
		BFDmanager_loadDefaultBinary (binary);

	AddressTable_Insert_MemReference (MEM_REFERENCE_DYNAMIC, "", "",
	  ADDR_UNRESOLVED, 0);
	AddressTable_Insert_MemReference (MEM_REFERENCE_STATIC, "", ADDR_UNRESOLVED,
	  "", 0);
#else
	UNREFERENCED_PARAMETER(binary);
#endif

	/* Initialize the hash cache */
	Addr2Info_HashCache_Initialize();

	/* The addresses translation module has been successfully initialized */
	Translate_Addresses = TRUE;
}

/** AddressTable_Initialize
 *
 * Initialize the addresses translation table structures.
 *
 * @return No return value.
 */
static void AddressTable_Initialize (void)
{
	int type;

	for (type=0; type < COUNT_ADDRESS_TYPES; type++)
	{
		AddressTable[type] = (struct address_table *)malloc(sizeof(struct address_table));
		if (AddressTable[type] == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for AddressTable[type=%d]\n", type);
			exit (-1);
		}
		AddressTable[type]->address = NULL;
		AddressTable[type]->num_addresses = 0;

		FunctionTable[type] = (struct function_table *)malloc(sizeof(struct function_table));
		if (FunctionTable[type] == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for FunctionTable[type=%d]\n", type);
			exit (-1);
		}
		FunctionTable[type]->function = NULL;
		FunctionTable[type]->address_id = NULL;
		FunctionTable[type]->num_functions = 0;
	}

	AddressObjectInfo.objects = NULL;
	AddressObjectInfo.num_objects = 0;

	Tables_Initialized = TRUE;
}

/** Address2Info_AddSymbol
 *
 * Add a symbol (address, name, filename and line tuple) in the translation
 * table of the specified address_type 
 */
void Address2Info_AddSymbol (UINT64 address, int addr_type, char * funcname,
	char * filename, int line)
{
	int found = FALSE;
	int i;

	for (i = 0; i < AddressTable[addr_type]->num_addresses && !found; i++)
		found = AddressTable[addr_type]->address[i].address == address;

	if (!found)
		AddressTable_Insert (address, addr_type, NULL, strdup(funcname), strdup(filename), line);
}

/** Address2Info_Translate_MemReference
 *
 * 
 * Si l'adreca demanada no estava ja a la taula d'adreces, la tradueix pel nom
 * de la funcio i el numero de linea de codi corresponents, li assigna un nou
 * identificador i afegeix una nova entrada a la taula
 * 
 * @param address
 * @param query
 *
 * @return
 */
UINT64 Address2Info_Translate_MemReference (unsigned ptask, unsigned task, UINT64 address,
	int query, UINT64 *calleraddresses)
{
#if !defined(HAVE_BFD)
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(address);
	UNREFERENCED_PARAMETER(query);
	UNREFERENCED_PARAMETER(calleraddresses);

	return address;
#else

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": Address2Info_TranslateMemreference (%u, %u, %lx, %d);\n",
	  ptask, task, address, query);
#endif

	if (query == MEM_REFERENCE_DYNAMIC)
	{
		int line;
		char * sname;
		char * filename;
		char * module;
		char buffer[2048];
		char tmp[1024];
		int i;

		snprintf (buffer, sizeof(buffer), "");

		/* Trim head and tail for callers that can't be translated */
		for (i = 0; i < MAX_CALLERS; i++)
			if (calleraddresses[i] != 0)
			{
				Translate_Address (calleraddresses[i], ptask, task, &module,
				  &sname, &filename, &line);
				if (!strcmp (filename, ADDR_UNRESOLVED) || !strcmp (filename, ADDR_NOT_FOUND))
					calleraddresses[i] = 0;
				else
					break;
			}

		for (i = MAX_CALLERS-1; i >= 0; i--)
			if (calleraddresses[i] != 0)
			{
				Translate_Address (calleraddresses[i], ptask, task, &module,
				  &sname, &filename, &line);
				if (!strcmp (filename, ADDR_UNRESOLVED) || !strcmp (filename, ADDR_NOT_FOUND))
					calleraddresses[i] = 0;
				else
					break;
			}

		for (i = 0; i < MAX_CALLERS; i++)
			if (calleraddresses[i] != 0)
			{
				Translate_Address (calleraddresses[i], ptask, task, &module,
				  &sname, &filename, &line);
				if (strlen(buffer) > 0)
					snprintf (tmp, sizeof(tmp), " > %s:%d", filename, line);
				else
					snprintf (tmp, sizeof(tmp), "%s:%d", filename, line);
				strncat (buffer, tmp, sizeof(buffer));
			}

		return 1+AddressTable_Insert_MemReference (query, module, "",
		  strdup(buffer), 0);

	}
	else if (query == MEM_REFERENCE_STATIC)
	{
		char *varname;

		Translate_Address_Data (address, ptask, task, &varname);

		return 1+AddressTable_Insert_MemReference (query, "", varname, "", 0);
	}
	else
		return address;

#endif
}

/** Address2Info_Translate
 *
 * 
 * Si l'adreca demanada no estava ja a la taula d'adreces, la tradueix pel nom
 * de la funcio i el numero de linea de codi corresponents, li assigna un nou
 * identificador i afegeix una nova entrada a la taula
 * 
 * @param address
 * @param query
 *
 * @return
 */
UINT64 Address2Info_Translate (unsigned ptask, unsigned task, UINT64 address,
	int query, int uniqueID)
{
#if !defined(HAVE_BFD)
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(address);
	UNREFERENCED_PARAMETER(query);
	UNREFERENCED_PARAMETER(uniqueID);

	return address;
#else
	UINT64 caller_address;
	int addr_type;
	int i;
	int already_translated;
	int line_id = 0;
	int function_id = 0;
	UINT64 result;
	int found;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": Address2Info_Translate (%u, %u, %lx, %d, %d);\n",
	  ptask, task, address, query, uniqueID);
#endif

/* address es la direccion de retorno por donde continuara ejecutandose el
 * codigo despues de cada CALL a una rutina MPI. En arquitecturas x86, despues
 * de la llamada se desapilan los parametros y se recogen los resultados. Asi
 * pues, la @ de retorno apunta una instruccion de codigo maquina que se refiere
 * a la misma instruccion de invocacion del codigo de usuario, p.e:
 *
 *           mi_funcion(1,2,3);  =====>  add ... (apilar parametros)
 *                                       call 0x... (invocar funcion)
 *                     @ RETORNO ----->  sub ... (despilar parametros)
 *                                       ... (recoger resultado)
 *
 * Al traducir la direccion de cualquiera de las instrucciones maquinas de una misma
 * instruccion del codigo fuente, la linea / fichero obtenida es la misma. Por tanto, al
 * traducir directamente la @ obtenida por el backtrace, identificamos la linea de
 * la llamada en el codigo. EXCEPCION: Cuando la funcion no recibe ni recoge parametros,
 * la @ de retorno apunta directamente a la siguiente instruccion del codigo (p.e. MPI_Finalize)
 *
 * En arquitecturas POWER4, los parametros se pasan por REGISTRO. Despues de la llamada
 * a la funcion no se realizan mas operaciones, de modo que la @ de retorno apunta a una
 * instruccion maquina que se refiere a la siguiente instruccion del codigo fuente. Al
 * traducir esta direccion, obtenemos la SIGUIENTE linea donde se encuentra la llamada
 * MPI.
 *
 * Restando 1 a la direccion de retorno, pasamos a apuntar los bits intermedios de
 * la instruccion anterior, que es el CALL. El mecanismo de traduccion permite indicar
 * un octeto intermedio de una instruccion, y de esta forma obtendremos la linea correcta en ambos casos.
 *
 */

	if (address == 0 || !Translate_Addresses)
		return address;

	/* Enable vars to generate labels in the PCF */
	switch (query)
	{
		case ADDR2MPI_FUNCTION:
		case ADDR2MPI_LINE:
			Address2Info_Labels[A2I_MPI] = TRUE;
			caller_address = address - 1;
			addr_type = uniqueID?UNIQUE_TYPE:MPI_CALLER_TYPE;
			break;
		case ADDR2OMP_FUNCTION:
		case ADDR2OMP_LINE:
			Address2Info_Labels[A2I_OMP] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE;
			break;
		case ADDR2CUDA_FUNCTION:
		case ADDR2CUDA_LINE:
			Address2Info_Labels[A2I_CUDA] = TRUE;
			caller_address = address-1;
			addr_type = uniqueID?UNIQUE_TYPE:CUDAKERNEL_TYPE;
			break;
		case ADDR2UF_FUNCTION:
		case ADDR2UF_LINE:
			Address2Info_Labels[A2I_UF] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:USER_FUNCTION_TYPE;
			break;
		case ADDR2SAMPLE_FUNCTION:
		case ADDR2SAMPLE_LINE:
			Address2Info_Labels[A2I_SAMPLE] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:SAMPLE_TYPE;
			break;
		case ADDR2OTHERS_FUNCTION:
		case ADDR2OTHERS_LINE:
			Address2Info_Labels[A2I_OTHERS] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:OTHER_FUNCTION_TYPE;
			break;
		default:
			return address;
	}

/* Traduciremos caller_address para obtener la linea exacta del codigo, pero
 * escribiremos address en el PCF.
 * En un objdump y utilidades similares, no podemos buscar
 * la caller_address, que apunta a un octeto intermedio de una instruccion,
 * mientras que si encontraremos address, que es una direccion base (de la siguiente
 * instruccion).
 *
 * Por el momento no intentamos obtener la direccion base de la instruccion de call
 * por la diversidad de tamanyo de instrucciones en las distintas maquinas
 */

	found = Addr2Info_HashCache_Search (address, &line_id, &function_id);

	if (!found)
	{
		/* Search if address has been translated (search [1]) */
		already_translated = FALSE;
		for (i = 0; i < AddressTable[addr_type]->num_addresses; i++)
		{
			if (AddressTable[addr_type]->address[i].address == address)
			{
				already_translated = TRUE;
				function_id = AddressTable[addr_type]->address[i].function_id;
				line_id = i;
				break;
			}
		}
	}
	else
		already_translated = TRUE;

	if (!already_translated) 
	{
		int line;
		char * funcname;
		char * filename;
		char * module;

		Translate_Address (caller_address, ptask, task, &module, &funcname, &filename, &line);

		/* Samples can be taken anywhere in the code. It can happen that two
		   samples at the different addresses refer to the same combination of
		   function/file/line... We search again if this combination has been
		   already translated. */
		if (ADDR2SAMPLE_FUNCTION == query || ADDR2SAMPLE_LINE == query)
			for (i = 0; i < AddressTable[addr_type]->num_addresses; i++)
				if (AddressTable[addr_type]->address[i].line == line &&
				    strcmp (AddressTable[addr_type]->address[i].file_name, filename) == 0)
				{
					already_translated = TRUE;
					function_id = AddressTable[addr_type]->address[i].function_id;
					line_id = i;
					break;
				}

		if (funcname == NULL || filename == NULL)
		{
			function_id = UNRESOLVED_ID;
			line_id = UNRESOLVED_ID;
			already_translated = TRUE;
		}
		else if (!strcmp (ADDR_UNRESOLVED, funcname) ||
		         !strcmp (ADDR_UNRESOLVED, filename))
		{
			function_id = UNRESOLVED_ID;
			line_id = UNRESOLVED_ID;
			already_translated = TRUE;
		}
		else if (!strcmp(ADDR_NOT_FOUND, funcname) ||
		         !strcmp (ADDR_NOT_FOUND, filename))
		{
			function_id = NOT_FOUND_ID;
			line_id = NOT_FOUND_ID;
			already_translated = TRUE;
		}

		/* Again, if not found, insert into the translation table */
		if (!already_translated)
		{
			line_id = AddressTable_Insert (address, addr_type, module, funcname,
			  filename, line);
			function_id = AddressTable[addr_type]->address[line_id].function_id;
		}
	}

	if (!found)
		Addr2Info_HashCache_Insert (address, line_id, function_id);

	result = 0;
	switch(query)
	{
		case ADDR2CUDA_FUNCTION:
		case ADDR2SAMPLE_FUNCTION:
		case ADDR2MPI_FUNCTION:
		case ADDR2UF_FUNCTION:
		case ADDR2OTHERS_FUNCTION:
		case ADDR2OMP_FUNCTION:
			result = function_id + 1;
			break;
		case ADDR2CUDA_LINE:
		case ADDR2UF_LINE:
		case ADDR2OTHERS_LINE:
		case ADDR2SAMPLE_LINE:
		case ADDR2OMP_LINE:
		case ADDR2MPI_LINE:
		case MEM_REFERENCE_DYNAMIC:
		case MEM_REFERENCE_STATIC:
			result = line_id + 1;
			break;
	}

	return result;
#endif /* HAVE_BFD */
}

/** AddressTable_Insert_MemReference
 *
 * @param address
 * @param funcname
 * @param filename
 * @param line
 * 
 * @return 
 */
#if defined(HAVE_BFD)
static int AddressTable_Insert_MemReference (int addr_type,
	const char *module, const char *staticname, const char *filename, int line) 
{
	int i; 
	int found;

#if defined(DEBUG) 
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s, %s, %d)\n",
	  addr_type, module, staticname, filename, line);
#endif

	int nObjects = AddressObjectInfo.num_objects;

	/* Search if it exists */
	for (i = 0; i < nObjects; i++)
	{
		struct address_object_info_st * object = &(AddressObjectInfo.objects[i]);
		if (addr_type == MEM_REFERENCE_STATIC && object->is_static)
			found = !strcmp (staticname, object->name);
		else if (addr_type == MEM_REFERENCE_DYNAMIC && !object->is_static)
			found = !strcmp (filename, object->file_name);
		if (found)
		{
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s, %s, %d) -> %d\n",
			  addr_type, module, staticname, filename, line, i);
#endif
			return i;
		}
	}

	/* If we're here, we haven't find it! */
	AddressObjectInfo.objects = (struct address_object_info_st*) realloc (
	  AddressObjectInfo.objects,
	  (AddressObjectInfo.num_objects+1)*sizeof(struct address_object_info_st));
	if (NULL == AddressObjectInfo.objects)
	{
		fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for memory object identifiers\n");
		exit (-1);
	}

	i = AddressObjectInfo.num_objects;
	AddressObjectInfo.objects[i].is_static = addr_type == MEM_REFERENCE_STATIC;
	AddressObjectInfo.objects[i].line = line;
	AddressObjectInfo.objects[i].file_name = filename;
	AddressObjectInfo.objects[i].name = staticname;
	AddressObjectInfo.objects[i].module = module;
	AddressObjectInfo.num_objects++;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s, %s, %d) -> %d\n",
	  addr_type, module, staticname, filename, line, i);
#endif

	return i;
}
#endif /* HAVE_DEFINED(BFD) */

/** AddressTable_Insert
 *
 * @param address
 * @param funcname
 * @param filename
 * @param line
 * 
 * @return 
 */
static int AddressTable_Insert (UINT64 address, int addr_type, char *module,
	char *funcname, char *filename, int line) 
{
	int i; 
	int found;
	int new_address_id;
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int function_id = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert (%lx, %d, %s, %s, %s, %d)\n",
	  address, addr_type, module, funcname, filename, line);
#endif

	AddrTab = AddressTable[addr_type];
	FuncTab = FunctionTable[addr_type];

	new_address_id = AddrTab->num_addresses ++;
	AddrTab->address = (struct address_info *) realloc (
		AddrTab->address,
		AddrTab->num_addresses * sizeof(struct address_info)
	);
	if (NULL == AddrTab->address)
	{
		fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for AddressTable\n");
		exit (-1);
	}
	
	AddrTab->address[new_address_id].address = address;
	AddrTab->address[new_address_id].file_name = filename;
	AddrTab->address[new_address_id].module = module;
	AddrTab->address[new_address_id].line = line;

	/* Search for the function id */
	found = FALSE;
	for (i=0; i < FuncTab->num_functions; i++) 
		if (!strcmp(funcname, FuncTab->function[i])) 
		{
			found = TRUE;
			function_id = i;
			break;
		}

	if (!found) 
	{
		function_id = FuncTab->num_functions ++;
		FuncTab->function = (char **) realloc (
			FuncTab->function, 
			FuncTab->num_functions * sizeof(char*)
		);
		if (NULL == FuncTab->function)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for function-identifiers table in FuncTab\n");
			exit (-1);
		}
		FuncTab->address_id = (UINT64*) realloc (
			FuncTab->address_id,
			FuncTab->num_functions * sizeof (UINT64)
		);
		if (NULL == FuncTab->address_id)
		{
			fprintf (stderr, "mpi2prv: Error! Cannot reallocate memory for address-identifiers table in FuncTab\n");
			exit (-1);
		}
		FuncTab->function[function_id] = funcname;
		FuncTab->address_id[function_id] = new_address_id;
	}
	AddrTab->address[new_address_id].function_id = function_id;

	return new_address_id;
}

#if defined(HAVE_BFD)
static void Translate_Address_Data (UINT64 address, unsigned ptask, unsigned task,
	char **symbol)
{
	*symbol = ADDR_UNRESOLVED;

	if (!Translate_Addresses) 
		return;

	ObjectTable_GetSymbolFromAddress (address, ptask, task, symbol);
}
#endif /* HAVE_BFD */

#if defined(HAVE_BFD)
static void Translate_Address (UINT64 address, unsigned ptask, unsigned task,
	char **module, char ** funcname, char ** filename, int * line)
{
	binary_object_t *obj;
	int found = FALSE;
	char *translated_function = NULL;
	char *translated_filename = NULL;
	int translated_line = 0;

	*funcname = ADDR_UNRESOLVED;
	*filename = ADDR_UNRESOLVED;
	*line = 0;

	if (!Translate_Addresses) 
		return;

	obj = ObjectTable_GetBinaryObjectAt (ptask, task, address);

# if defined(DEBUG)
	if (obj)
	{
		printf ("\naddress %llx is at file %s [%llx - %llx] - abfd = %p asym = %p\n", address,
		  obj->module, obj->start_address, obj->end_address, obj->bfdImage, obj->bfdSymbols);
	}
	else
		printf ("obj = NULL for address %llx\n", address);
# endif

	if (obj)
	{
		found = BFDmanager_translateAddress (obj->bfdImage, obj->bfdSymbols,
		  (void*) address, &translated_function, &translated_filename, &translated_line);

		/* If we didn't find the address, then the function is possibly in a shared
		   library. Substract base address and retry.
		   If we found it, just ensure we don't get the module name for the main binary */
		if (!found)
		{
			found = BFDmanager_translateAddress (obj->bfdImage, obj->bfdSymbols,
			  (void*) (address - obj->start_address), &translated_function,
			  &translated_filename, &translated_line);
		}
	}
	else
	{
		found = BFDmanager_translateAddress (BFDmanager_getDefaultImage(),
		  BFDmanager_getDefaultSymbols(), (void*) address, &translated_function,
		  &translated_filename, &translated_line);
	}

	if (!found) 
	{
		/* The address has not been found in the binary */
		*filename = ADDR_NOT_FOUND;
		*funcname = ADDR_NOT_FOUND;
		*line = 0;
	}
	else 
	{
		*line = translated_line;

		if (translated_function != NULL) 
		{
			char buffer[1024];
			unsigned count = 0;
			char *ptr;

			/* Remove CUDA prefixes */
			if ((ptr = strstr (translated_function, "__device_stub__Z")) != NULL)
			{
				ptr += strlen ("__device_stub__Z");
				while (ptr[0]>='0' && ptr[0]<='9')
				{
					count = count * 10 + (ptr[0] - '0');
					ptr++;
				}
				/* Add +1 in the translated_function name for additional \0 in C */
				snprintf (buffer, MIN(count+1,sizeof(buffer)), "%s", ptr);
				COPY_STRING(buffer, *funcname);
			}
			else
			{
				COPY_STRING(translated_function, *funcname);
			}
		}
		else
			*funcname = ADDR_UNRESOLVED;

		if (translated_filename != NULL) 
		{
			char *file_basename = basename((char *)translated_filename);
			COPY_STRING(file_basename, *filename);
		}
		else
			*filename = ADDR_UNRESOLVED;
	}

	*module = NULL;
	if (obj != NULL)
		if (obj->module != NULL)
			*module = strdup (basename ((char*) obj->module));
}
#endif /* HAVE_BFD */

UINT64 Address2Info_GetLibraryID (unsigned ptask, unsigned task, UINT64 address)
{
#if !defined(HAVE_BFD)
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(address);
#else
	binary_object_t *obj = ObjectTable_GetBinaryObjectAt (ptask, task, address);
	if (obj)
		return obj->index;
#endif
	return 0;
}

void Address2Info_Write_LibraryIDs (FILE *pcf_fd)
{
#if !defined(HAVE_BFD)
	UNREFERENCED_PARAMETER(pcf_fd);
#else
	if (BFDmanager_numLoadedBinaries() > 0 && 
	    get_option_merge_EmitLibraryEvents())
	{
		unsigned b;
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", LIBRARY_EV, LIBRARY_LBL);
		fprintf (pcf_fd, "%s\n", VALUES_LABEL);
		fprintf (pcf_fd, "0    Unknown\n");
		for (b = 0; b < BFDmanager_numLoadedBinaries(); b++)
		{
			loadedModule_t *m = BFDmanager_getLoadedModule (b);
			fprintf(pcf_fd, "%d    %s\n", b+1, m->module);
		}
		LET_SPACES(pcf_fd);
	}
#endif
}

void Address2Info_Write_MPI_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];

	if (Address2Info_Labels[A2I_MPI] /*Address2Info_Need_MPI_Labels*/) 
	{
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		if (MPI_Caller_Multiple_Levels_Traced) 
		{
			if (MPI_Caller_Labels_Used == NULL) 
				for(i=1; i<=MAX_CALLERS; i++) 
					fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_EV+i, CALLER_LVL_LBL, i);
			else 
				for(i=1; i<=MAX_CALLERS; i++) 
					if (MPI_Caller_Labels_Used[i-1] == TRUE) 
						fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_EV+i, CALLER_LVL_LBL, i);
		}
		else 
			fprintf(pcf_fd, "0    %d    %s\n", CALLER_EV, CALLER_LBL);

		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i=0; i<FuncTab->num_functions; i++) 
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		if (MPI_Caller_Multiple_Levels_Traced) 
		{
			if (MPI_Caller_Labels_Used == NULL) 
				for(i=1; i<=MAX_CALLERS; i++) 
					fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_LINE_EV+i, CALLER_LINE_LVL_LBL, i);
			else 
				for(i=1; i<=MAX_CALLERS; i++) 
					if (MPI_Caller_Labels_Used[i-1] == TRUE) 
						fprintf(pcf_fd, "0    %d    %s %d\n", CALLER_LINE_EV+i, CALLER_LINE_LVL_LBL, i);
		}
		else 
			fprintf(pcf_fd, "0    %d    %s\n", CALLER_LINE_EV, CALLER_LINE_LBL);

		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_MemReferenceCaller_Labels (FILE * pcf_fd)
{
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	if (Address2Info_Initialized())
	{
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		fprintf(pcf_fd, "0    %d    %s\n", SAMPLING_ADDRESS_ALLOCATED_OBJECT_EV,
		  SAMPLING_ADDRESS_ALLOCATED_OBJECT_LBL);

		if (AddressObjectInfo.num_objects > 0)
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

		for (i = 0; i < AddressObjectInfo.num_objects; i++)
		{
			struct address_object_info_st *obj = &(AddressObjectInfo.objects[i]);

			if (obj->is_static)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, obj->name);
				if (!shortened)
					fprintf (pcf_fd, "%d %s\n", i+1, obj->name);
				else
					fprintf (pcf_fd, "%d %s [%s]\n", i+1, short_label, obj->name);
			}
			else
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, obj->file_name);
				if (!shortened)
					fprintf (pcf_fd, "%d (%s)\n", i+1, obj->file_name);
				else
					fprintf (pcf_fd, "%d (%s) [%s]\n", i+1, short_label,
					  obj->file_name);
			}
		}

		if (AddressObjectInfo.num_objects > 0)
			LET_SPACES(pcf_fd);
	}
}

void Address2Info_Write_OMP_Labels (FILE * pcf_fd, int eventtype,
	char *eventtype_description, int eventtype_line,
	char *eventtype_line_description, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];

	if (Address2Info_Labels[A2I_OMP] /*Address2Info_Need_OMP_Labels*/) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", eventtype, eventtype_description);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", eventtype_line, eventtype_line_description);
		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_CUDA_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];

	if (Address2Info_Labels[A2I_CUDA]) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDAFUNC_EV, "CUDA kernel");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}

			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDAFUNC_LINE_EV, "CUDA kernel source code line");
		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_UF_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];

	if (Address2Info_Labels[A2I_UF] /*Address2Info_Need_UF_Labels*/) 
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_EV, "User function");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_LINE_EV, "User function line");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_OTHERS_Labels (FILE * pcf_fd, int uniqueid, int nlabels,
	codelocation_label_t *labels)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:OTHER_FUNCTION_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:OTHER_FUNCTION_TYPE];

	if (Address2Info_Labels[A2I_OTHERS] && nlabels > 0)
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		for (i = 0; i < nlabels; i++)
			if (labels[i].type == CODELOCATION_FUNCTION)
				fprintf (pcf_fd, "0    %d    %s\n", labels[i].eventcode, labels[i].description);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		for (i = 0; i < nlabels; i++)
			if (labels[i].type == CODELOCATION_FILELINE)
				fprintf (pcf_fd, "0    %d    %s\n", labels[i].eventcode, labels[i].description);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_Sample_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;
	char short_label[1+SHORT_STRING_PREFIX+SHORT_STRING_SUFFIX+strlen(SHORT_STRING_INFIX)];

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];

	if (Address2Info_Labels[A2I_SAMPLE] /*Address2Info_Need_Sample_Labels*/) 
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", SAMPLING_EV, "Sampled functions");
		if (Sample_Caller_Labels_Used != NULL)
			for (i = 1; i <= MAX_CALLERS; i++)
				if (Sample_Caller_Labels_Used[i-1])
					fprintf (pcf_fd, "0    %d    Sampled functions (depth %d)\n", SAMPLING_EV+i, i);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, FuncTab->function[i]);
				if (shortened)
					fprintf(pcf_fd, "%d %s [%s]\n", i + 1, short_label, FuncTab->function[i]);
				else
					fprintf(pcf_fd, "%d %s\n", i + 1, FuncTab->function[i]);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", SAMPLING_LINE_EV, "Sampled line functions (depth 0)");
		if (Sample_Caller_Labels_Used != NULL)
			for (i = 1; i <= MAX_CALLERS; i++)
				if (Sample_Caller_Labels_Used[i-1])
					fprintf (pcf_fd, "0    %d    Sampled lines functions (depth %d)\n", SAMPLING_LINE_EV+i, i);

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0   %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				int shortened = ExtraeUtils_shorten_string (SHORT_STRING_PREFIX,
				  SHORT_STRING_SUFFIX, SHORT_STRING_INFIX,
				  sizeof(short_label), short_label, AddrTab->address[i].file_name);
				if (shortened)
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s) [%d (%s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
						    AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s) [%d (%s, %s)]\n", 
							i + 1, AddrTab->address[i].line, short_label,
							AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
				else
				{
					if (AddrTab->address[i].module == NULL)
						fprintf(pcf_fd, "%d %d (%s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name);
					else
						fprintf(pcf_fd, "%d %d (%s, %s)\n", 
							i + 1, AddrTab->address[i].line, AddrTab->address[i].file_name,
					    	AddrTab->address[i].module);
				}
			}
			LET_SPACES(pcf_fd);
		}
	}
}

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

void Share_Callers_Usage (void)
{
	int MPI_used[MAX_CALLERS], SAMPLE_used[MAX_CALLERS];
	int A2I_tmp[A2I_LAST];
	int i, res;

	res = MPI_Reduce (Address2Info_Labels, A2I_tmp, A2I_LAST, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK (res, MPI_Reduce, "Sharing information about address<->info labels");
	for (i = 0; i < A2I_LAST; i++)
		Address2Info_Labels[i] = A2I_tmp[i];

	if (MPI_Caller_Labels_Used == NULL)
	{
		MPI_Caller_Labels_Used = malloc(sizeof(int)*MAX_CALLERS);
		if (MPI_Caller_Labels_Used == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for used MPI Caller labels\n");
			exit (-1);
		}
		for (i = 0; i < MAX_CALLERS; i++)
			MPI_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (MPI_Caller_Labels_Used, MPI_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about MPI address<->info");
	for (i = 0; i < MAX_CALLERS; i++)
		MPI_Caller_Labels_Used[i] = MPI_used[i];

	if (Sample_Caller_Labels_Used == NULL)
	{
		Sample_Caller_Labels_Used = malloc(sizeof(int)*MAX_CALLERS);
		if (Sample_Caller_Labels_Used == NULL)
		{
			fprintf (stderr, "mpi2prv: Fatal error! Cannot allocate memory for used sample Caller labels\n");
			exit (-1);
		}
		for (i = 0; i < MAX_CALLERS; i++)
			Sample_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (Sample_Caller_Labels_Used, SAMPLE_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about sampling address<->info");
	for (i = 0; i < MAX_CALLERS; i++)
		Sample_Caller_Labels_Used[i] = SAMPLE_used[i];

	res = MPI_Reduce (&MPI_Caller_Multiple_Levels_Traced, &i, 1, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about multiple address<->info labels");
	MPI_Caller_Multiple_Levels_Traced = i;
}
#endif /* PARALLEL_MERGE */

