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
#ifdef HAVE_STRING_H
# include <string.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_LIBGEN_H
# include <libgen.h>
#endif
#ifdef HAVE_SYS_TIME_H
# include <sys/time.h>
#endif
# include "addr2line.h"
# include "maps.h"

// #define DEBUG
// #define LEGACY_BEHAVIOR

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
#include "events.h"
#include "addr2info.h"
#include "labels.h"
#include "misc_prv_semantics.h"
#include "addr2info_hashcache.h"
#include "options.h"
#include "xalloc.h"

static void AddressTable_Initialize (void);
static void AddressTable_Insert (UINT64 address, int event_type, 
                                 char *module, char *funcname, char *filename, int line,
                                 int *function_id, int *line_id);
static int  AddressTable_Insert_MemReference (int addr_type, 
                                              const char *data_reference, 
                                              const char *mapping_name);

#if defined(HAVE_LIBADDR2LINE)
static void translate_function_address (UINT64 address, unsigned ptask, unsigned task, 
                                        char **module, char ** funcname, char ** filename, int * line);
static void translate_data_address (UINT64 address, unsigned ptask, unsigned task, 
                                    char **mapping_name, char **symbol_name);
#endif

static int Tables_Initialized = FALSE;
static int Translation_Module_Initialized = FALSE;

/**
 * AddressTable
 * 
 * Array of translation tables, one for each address type, correlating 
 * addresses to their corresponding line, file, module and function identifier.
 */
static struct address_table  *AddressTable[COUNT_ADDRESS_TYPES];

/**
 * FunctionTable
 * 
 * Array of translation tables, one for each address type, correlating
 * address identifiers (index in AddressTable) to their corresponding
 * function name.
 */
static struct function_table *FunctionTable[COUNT_ADDRESS_TYPES];

/**
 * AddressObjectInfo
 * 
 * Translation table for data objects (from PEBS samples), correlating
 * addresses to their corresponding line, file, module and variable name.
 * 
 * TODO: Possibly merge this with AddressTable with a new address category for data objects.
 *       AddressTable's function_id could be generalised into object_id, and 
 *       FunctionTable into ObjectTable, to hold both functions and data objects.
 */
static struct address_object_table_st AddressObjectInfo;

/*
 * Categories to dump the translation tables into the PCF file
 * TODO: Feels a bit redundant to the enum for address types up to COUNT_ADDRESS_TYPES, only excluding UNIQUE_TYPE
 *       Maybe UNIQUE_TYPE should not be an address type, just a flag to control the id's assigned to the addresses
 */
#define A2I_MPI      0
#define A2I_OMP      1
#define A2I_UF       2
#define A2I_SAMPLE   3
#define A2I_CUDA     4
#define A2I_HIP      5
#define A2I_OTHERS   6
#define A2I_LAST     7

// Boolean to indicate if labels should be generated per address type
int Address2Info_Labels[A2I_LAST];

/** 
 * Address2Info_Initialize
 *
 * Initialize the memory addresses translation module.
 * This function prepares the tables to store the translation of addresses.
 * These tables are categorized by the type of address (MPI, OpenMP, CUDA, HIP, etc.).
 * When the flag -unique-caller-id is set, all the addresses are stored in the 
 * same table under category 'UNIQUE_TYPE', regardless of the type of address.
 * 
 * Formerly, this function also initialized the translation back-end with binutils, 
 * but now it is done lazily in the translate_function_address function the first time 
 * a translation is requested. See Initialize_Translation_Backend for details.
 * 
 * TODO: This routine is called unconditionally, so the tables are ready to 
 * store user addresses that were emitted manually through the API call
 * Extrae_register_function_address, and may be discovered after parsing 
 * the SYM file. However, this prepares all the tables for any kind of
 * address, plus the structures for PEBS samples (*_MemReference), as well 
 * as the hot-addresses caches (*_HashCache_*), and all this might be 
 * for nothing if the trace does not contain automatically captured 
 * addresses to translate. It would be nice to:
 * 1) Defer some of these initializations until (if) they are actually required.
 * 2) Merge the AddressTable_* and AddressTable_*_MemRefence API. Instead of
 *    duplicating all methods, we can just consider PEBS samples as a different
 *    category of addresses (new entries MEM_REFERENCE_DYNAMIC_TYPE and 
 *    MEM_REFERENCE_STATIC_TYPE in corresponding enum at addr2info.h).
 * 3) Generalise the hot-addresses cache (*_HashCache_*), currently it is 
 *    only used to accelerate translations for non-data symbols, i.e., 
 *    for callstack, CUDA kernels, HIP kernels, OpenMP outlineds, etc.
 */
void Address2Info_Initialize (void)
{
	int type;

	Translation_Module_Initialized = FALSE;

	// Initialize the addresses translation tables
	AddressTable_Initialize();

	// Add fake entries in the translation tables to link "unknown" translations to ID 0, which will be 1st entry in PCF
	for (type = 0; type < COUNT_ADDRESS_TYPES; type++)
	{
		int function_id = 0, line_id = 0;
		AddressTable_Insert (0x0, type, UNKNOWN_MAPPING, UNKNOWN_ADDRESS, UNKNOWN_ADDRESS, 0, &line_id, &function_id); 
	}
	// TODO: Either unify the two "unknowns" into a single entry, or make explicit which one is dynamic and which is static.
	AddressTable_Insert_MemReference (MEM_REFERENCE_DYNAMIC, UNKNOWN_SYMBOL, UNKNOWN_MAPPING);
	AddressTable_Insert_MemReference (MEM_REFERENCE_STATIC, UNKNOWN_SYMBOL, UNKNOWN_MAPPING);

	// Initialize the hash cache for the frequently referenced function addresses (not for data object addresses)
	Addr2Info_HashCache_Initialize();

	// Mark the address translation module has been successfully initialized
	Translation_Module_Initialized = TRUE;
}

/** 
 * Address2Info_Initialized
 *
 * Checks if the translation system is correctly initialized.
 *
 * @return Returns 1 if Address2Info_Initialize was called and returned successfully
 */
int Address2Info_Initialized (void)
{ 
	return Translation_Module_Initialized;
}


/*******************************************\
 *                                         *
 *   API TO TRANSLATE FUNCTION ADDRESSES   *
 *                                         *
\*******************************************/

/** 
 * AddressTable_Initialize
 *
 * Initializes the translation table structures for function addresses.
 */
static void AddressTable_Initialize (void)
{
	int type;

	for (type=0; type < COUNT_ADDRESS_TYPES; type++)
	{
		AddressTable[type] = (struct address_table *)xmalloc(sizeof(struct address_table));
		AddressTable[type]->address = NULL;
		AddressTable[type]->num_addresses = 0;

		FunctionTable[type] = (struct function_table *)xmalloc(sizeof(struct function_table));
		FunctionTable[type]->function = NULL;
		FunctionTable[type]->address_id = NULL;
		FunctionTable[type]->num_functions = 0;
	}

	AddressObjectInfo.objects = NULL;
	AddressObjectInfo.num_objects = 0;

	Tables_Initialized = TRUE;
}

/**
 * AddressTable_Search
 *
 * Searches for an address in the full translation table of the specified address_type.
 * If found, it returns the line and function identifiers.
 *
 * @param addr_type The type of address (MPI, OpenMP, etc.)
 * @param address The address to search for
 * @param line_id[out] The line identifier
 * @param function_id[out] The function identifier
 * @return 1 if the address was found, 0 otherwise
 */
static int AddressTable_Search(int addr_type, UINT64 address, int *line_id, int *function_id)
{
	int i = 0;

	for (i = 0; i < AddressTable[addr_type]->num_addresses; i++)
	{
		if (AddressTable[addr_type]->address[i].address == address)
		{
			*function_id = AddressTable[addr_type]->address[i].function_id;
			*line_id = i;
			return TRUE;
		}
	}
	return FALSE;
}

/**
 * AddressTable_Insert
 *
 * This function inserts the debug information of a translated 
 * function address into two tables:
 * - AddressTable: stores the address, its corresponding mapping name,
 *                 the source file name, the line number, and the 
 *                 identifier of the function in the FuncTab table.
 * - FunctionTable: stores the function name and the identifier of the
 *                  corresponding entry in the AddrTab table.
 *
 * This separation enables to assign an unique identifier to each
 * source code location <module, filename, line>, and an unique
 * identifier to each function name, that may be shared by
 * multiple source code locations.
 * FIXME: These tables are lists, this is very inefficient for searching.
 *        We should use hashes indexed by address, and then pointing
 *        to a dictionary of unique triplets of <mapping, file, line>; 
 *        and a dictionary of unique function names.
 * 
 * There are two cases where the address is not inserted in the tables:
 * 1) The address is partially unresolved
 * 2) A sample address that shares the same <mapping, file, line> than another
 * FIXME: This is confusing, because the tables do not keep all the 
 *        addresses that we have translated before, only those that 
 *        result in a different translation; and forces new samples 
 *        to search for the same translation across the whole list.
 *
 * @param address The translated address
 * @param funcname The name of the function corresponding to the address
 * @param filename The name of the source file corresponding to the address
 * @param line The line number corresponding to the address
 * @param unique_line_id[out] The identifier of the unique address (unique source code line) in the AddrTab table
 * @param unique_function_id[out] The identifier of the unique function in the FuncTab table
 */
static void AddressTable_Insert (UINT64 address, int addr_type, 
                                char *module, char *funcname, char *filename, int line,
							    int *unique_line_id, int *unique_function_id)
{
	int i = 0; 
	int found = 0;
	int new_address_id = 0;
	struct address_table  *AddrTab = NULL;
	struct function_table *FuncTab = NULL;
	int function_id = 0;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert (%lx, %d, %s, %s, %s, %d)\n",
	         address, addr_type, module, funcname, filename, line);
#endif

	// Select the proper translation tables on the address category
	AddrTab = AddressTable[addr_type];
	FuncTab = FunctionTable[addr_type];

	if (AddrTab->num_addresses > 0)
	{
#if defined(LEGACY_BEHAVIOR)
		// Force any partially unresolved addresses to fallback to id 0
		if (!strcmp(funcname, UNKNOWN_ADDRESS) || !strcmp(filename, UNKNOWN_ADDRESS))
#else
		// Allow partially unresolved addresses to be stored in the translation tables
		if (!strcmp(funcname, UNKNOWN_ADDRESS) && !strcmp(filename, UNKNOWN_ADDRESS) && line == 0)
#endif /* LEGACY_BEHAVIOR */
		{
			// "Unresolved" value in the PCF (1st entry inserted during initialization)
			*unique_line_id = UNRESOLVED_ID;
			*unique_function_id = AddrTab->address[UNRESOLVED_ID].function_id;
			return;
		}
	}

	/*    
	 * Two samples with different addresses may correspond to the same combination
	 * of source file and line. We check whether this combination has already been 
	 * translated. If so, we reuse the identifier [See DONTUSEBASENAMES].
	 * NOTE: I have added the check for the module name, because the same source file
	 *       could be compiled into different libraries that are loaded at the same time.
	 * FIXME: This is extremely inefficient, it forces any sample to check the whole table!
	 */
	if (addr_type == SAMPLE_TYPE) 
	{
		for (i = 0; i < AddrTab->num_addresses; i++) 
		{
			if ((AddrTab->address[i].line == line) &&
				(strcmp (AddrTab->address[i].file_name, filename) == 0)
#if !defined(LEGACY_BEHAVIOR)
				&& (strcmp (AddrTab->address[i].module, module) == 0)
#endif
			) {
				function_id = AddrTab->address[i].function_id;

				// Return the identifiers of the entries that were already present in the tables
				*unique_line_id = i;
				*unique_function_id = function_id;
				return; 
				/* 
				 * FIXME: This return prevents the current sample address, that resolves to the same 
				 *        <mapping, file, line> than another sample, to be inserted in the tables. 
				 */
			}
		}
	}

	// Insert a new entry in the AddrTab for this source code location identified by 'new_address_id'
	new_address_id = AddrTab->num_addresses ++;
	AddrTab->address = (struct address_info *) xrealloc (
		AddrTab->address,
		AddrTab->num_addresses * sizeof(struct address_info)
	);
	AddrTab->address[new_address_id].address = address;
	AddrTab->address[new_address_id].file_name = filename;
	AddrTab->address[new_address_id].module = module;
	AddrTab->address[new_address_id].line = line;

	/* 
	 * Search if the function already has an entry in the FuncTab table,
	 * If not, add it a new entry identified by 'function_id'
	 */
	found = FALSE;
	for (i=0; i < FuncTab->num_functions; i++) {
		if (!strcmp(funcname, FuncTab->function[i]))
		{
			found = TRUE;
			function_id = i;
			break;
		}
	}
	if (!found) 
	{
		function_id = FuncTab->num_functions ++;
		FuncTab->function = (char **) xrealloc (
			FuncTab->function, 
			FuncTab->num_functions * sizeof(char*)
		);
		FuncTab->address_id = (UINT64*) xrealloc (
			FuncTab->address_id,
			FuncTab->num_functions * sizeof (UINT64)
		);
		FuncTab->function[function_id] = funcname;
		FuncTab->address_id[function_id] = new_address_id;
	}
	AddrTab->address[new_address_id].function_id = function_id;

	// Return the identifier of the new entry in the AddrTab table, and the identifier of the corresponding function
	*unique_line_id = new_address_id;
	*unique_function_id = function_id;
}

/** 
 * Address2Info_AddSymbol
 *
 * Add a symbol (address, name, filename and line tuple) in the translation
 * table of the specified address_type.
 * 
 * This is specific to function addresses (not data objects), and is used
 * to manually emit address translations through the API call
 * Extrae_register_function_address.
 * 
 * @param address The translated address
 * @param addr_type The type of address (MPI, OpenMP, etc.)
 * @param funcname The function name
 * @param filename The source file name
 * @param line The line number in the source file
 */
void Address2Info_AddSymbol (UINT64 address, int addr_type,
                             char * funcname, char * filename, int line)
{
	int found = FALSE;
	int i = 0;

	// Check if the address is already in the table
	for (i = 0; i < AddressTable[addr_type]->num_addresses && !found; i++) {
		found = AddressTable[addr_type]->address[i].address == address;
	}

	// If not, add it to the tables
	if (!found) {
		int function_id = 0, line_id = 0;
		AddressTable_Insert (address, addr_type, NULL, strdup(funcname), strdup(filename), line, &line_id, &function_id);
	}
}

#if defined(HAVE_LIBADDR2LINE)

static void initialize_translation_backend(unsigned ptask, unsigned task)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);
	char *main_binary = ObjectTree_getMainBinary(ptask, task);

	/*
	 * We prioritize using /proc/self/maps if available, otherwise use /proc/self/exe, as a last resort use the user input binary.
	 * The selection between /proc/self/exe or the user input binary is done at ObjectTree_getMainBinary.
	 */
	if (task_info->proc_self_maps != NULL)
	{
		task_info->addr2line = addr2line_init_maps(task_info->proc_self_maps, OPTION_CLEAR_PRELOAD);
	}
	else if (main_binary != NULL)
	{
		task_info->addr2line = addr2line_init_file(main_binary, OPTION_CLEAR_PRELOAD);
	}
} 

static void translate_function_address (UINT64 address, unsigned ptask, unsigned task,
	char **module, char ** funcname, char ** filename, int * line)
{
	code_loc_t code_loc;
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);

	if (!Translation_Module_Initialized) 
	{
		*funcname = UNKNOWN_ADDRESS;
		*filename = UNKNOWN_ADDRESS;
		*line = 0;
		return;
	}

	if (!task_info->addr2line) initialize_translation_backend(ptask, task);

	addr2line_translate(task_info->addr2line, (void *)address, &code_loc);

	*filename = code_loc.file;
	*line = code_loc.line;
	*funcname = code_loc.function;
 	*module = code_loc.mapping_name;

	/*
	 * After translating, we used to do an extra parsing on the function name to remove CUDA decorations '__device_stub__Z' from the kernel names.
	 * We have removed that temporarily after the change from binutils to elfutils (libaddr2line), just in case 
	 * the demangle of elfutils already does this for us. We may need to reintroduce this.
	 * Also, we used to truncate the absolute filename path into its basename, this is also removed to avoid conflicts
	 * between different paths that end with the same basename, when looking for previous translations [DONTUSEBASENAMES].
	 */
}

#endif /* HAVE_LIBADDR2LINE */

/**
 * Address2Info_Translate
 *
 * If the requested address was not already in the address table, it is translated
 * to the corresponding function name, line number and file name, assigned a new 
 * identifier, and a new entry is added to the translation table. 
 *
 * FIXME: The trace contains separate events: one to represent the function name,
 * and another for the line number and source file. This function, after a successful
 * translation, returns either the function identifier or the line number/source file 
 * identifier, depending on the query type (i.e., which kind of event is being translated). 
 * This design is inefficient because these two events typically appear consecutively in the trace.
 * On the first call, we translate the address, update the relevant tables and caches, and return
 * the function identifier. On the second call, we look up the same address again, just to return
 * the line number and source file identifier. This redundant lookup could be avoided if 
 * both identifiers were returned at once.
 *
 * @param ptask The application identifier
 * @param task The task identifier
 * @param address The address to translate 
 * @param query Indicates the category of the address being translated,
 *              as well as whether the translation should return the function name
 *              or the source file and line number.
 * @param uniqueID A boolean indicating whether addresses from different categories
 *                 should be assigned unique identifiers ('true'), or whether the 
 *                 identifier may appear across different categories ('false'). 
 *                 This behavior is controlled via the `-unique-caller-id` flag.
 * @return The translated function identifier, or the line number and file name identifier 
 *         (depending on the 'query' parameter), or the original address if translation fails.
 */
UINT64 Address2Info_Translate (unsigned ptask, unsigned task, UINT64 address, int query, int uniqueID)
{
#if !defined(HAVE_LIBADDR2LINE)

	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(address);
	UNREFERENCED_PARAMETER(query);
	UNREFERENCED_PARAMETER(uniqueID);

	return address;

#else /* HAVE_LIBADDR2LINE */

	UINT64 caller_address;
	int addr_type;
	int already_translated;
	int line_id = 0;
	int function_id = 0;
	UINT64 result = 0;
	int found_in_cache = 0;

# if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": Address2Info_Translate (%u, %u, %lx, %d, %d);\n",
	         ptask, task, address, query, uniqueID);
# endif

	/*
	 * 'address' is the return address where execution will continue after each
	 * CALL to an MPI routine. On x86 architectures, after the call, parameters
	 * are popped off the stack and the result is retrieved. Therefore, the return
	 * address points to a machine instruction that still belongs to the same
	 * high-level source instruction that performed the call. For example:
	 *
	 *           my_function(1,2,3);  =====> add ...    (push parameters)
	 *                                       call 0x... (invoke function)
	 *               RETURN @ ------->       sub ...    (pop parameters)
	 *                                       ...        (retrieve result)
	 *
	 * When translating the address of any machine instruction belonging to a single
	 * source-level instruction, the resulting file/line information is the same.
	 * Therefore, by translating the return address obtained via backtrace, we can 
	 * correctly identify the line in the source code where the call occurs.
	 * 
	 * EXCEPTION: When a function neither takes parameters nor returns values
	 * (e.g., `MPI_Finalize`), the return address points directly to the next 
	 * instruction in the code.
	 *
	 * On POWER4 architectures, parameters are passed via registers.
	 * No additional operations are performed after the function call,
	 * so the return address points to a machine instruction that belongs to
	 * the **next** line in the source code after the MPI call.
	 *
	 * By subtracting 1 from the return address, we effectively point to a middle
	 * byte of the previous instruction, which is the actual call.
	 * The translation mechanism supports identifying the correct source line
	 * even when given a byte inside an instruction, allowing us to retrieve
	 * the accurate line number in both cases.
	 */

	if (address == 0 || !Translation_Module_Initialized) return address;

	// Mark that for the category of the address being translated, labels need to be generated in the PCF
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
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:CUDAKERNEL_TYPE;
			break;
		case ADDR2HIP_FUNCTION:
		case ADDR2HIP_LINE:
			Address2Info_Labels[A2I_HIP] = TRUE;
			caller_address = address;
			addr_type = uniqueID?UNIQUE_TYPE:HIPKERNEL_TYPE;
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

	/*
	 * We will translate 'caller_address' to obtain the exact line of code,
	 * but we will write 'address' into the PCF.
	 * In objdump and similar tools, we cannot look up 'caller_address',
	 * as it may point to a middle byte within an instruction.
	 * However, 'address' can be found, as it refers to the base address
	 * of the following instruction.
	 *
	 * We do not attempt to retrieve the base address of the call instruction itself
	 * due to the varying instruction sizes across different machines.
	 * 
	 * FIXME: Would be better just to work with 'address' and substract 1 to the 'line' for callstack @'s?
	 */

	// First search in the cache for frequently referenced function addresses (quick)
	found_in_cache = Addr2Info_HashCache_Search (address, &line_id, &function_id);
	if (!found_in_cache)
	{
		// If not in cache, search in the full table of translated addresses (slow)
		already_translated = AddressTable_Search(addr_type, address, &line_id, &function_id);
	}
	else already_translated = TRUE;

	// If we couldn't find a previous translation, we need to translate it now and update the table and cache
	if (!already_translated) 
	{
		int line;
		char * funcname;
		char * filename;
		char * module;

		// Get the translation from libaddr2line
		translate_function_address (caller_address, ptask, task, &module, &funcname, &filename, &line);

		// Update the translation tables
		AddressTable_Insert (address, addr_type, module, funcname, filename, line,
		                     &line_id, &function_id);
	}

	// Update the cache with the current address translation
	if (!found_in_cache) Addr2Info_HashCache_Insert (address, line_id, function_id);

	/* 
	 * Returns the translated function identifier, 
	 * or the line number and file name identifier, 
	 * depending on the event being translated.
	 */
	switch(query)
	{
		case ADDR2CUDA_FUNCTION:
		case ADDR2HIP_FUNCTION:
		case ADDR2SAMPLE_FUNCTION:
		case ADDR2MPI_FUNCTION:
		case ADDR2UF_FUNCTION:
		case ADDR2OTHERS_FUNCTION:
		case ADDR2OMP_FUNCTION:
			result = function_id + 1;
			break;
		case ADDR2CUDA_LINE:
		case ADDR2HIP_LINE:
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

#endif /* HAVE_LIBADDR2LINE */
}


/**
 * Address2Info_Sort_routine
 * 
 * This function is used by qsort to sort the addresses by line and file name.
 */
static int Address2Info_Sort_routine(const void *p1, const void *p2)
{
	struct address_info *a1 = (struct address_info*) p1;
	struct address_info *a2 = (struct address_info*) p2;

	/* Sort by filename, line and address */
	if (strcmp (a1->file_name, a2->file_name) == 0)
	{
		if (a1->line == a2->line)
		{
			if (a1->address == a2->address) return 0;
			else if (a1->address < a2->address) return -1;
			else return 1;
		}
		else if (a1->line < a2->line) return -1;
		else return 1;
	}
	else return strcmp (a1->file_name, a2->file_name);
}

/**
 * Address2Info_Sort
 *
 * Sort the addresses in the translation tables.
 * There's a bunch of "address[1]" and "num_addresses-1" to skip the 
 * first entry in the table, which is a fake entry for the unresolved
 * addresses. 
 *
 * This operation is done when the flag -sort-addresses is set,
 * and applies to all addresses collected both via manual calls to the 
 * Extrae_register_function_address API, and via automatic instrumentation,
 * including OpenMP outlined functions, CUDA kernels, HIP kernels, call stack, etc.
 * However, it does not apply to data object addresses collected via 
 * PEBS samples. 
 */
void Address2Info_Sort (int unique_ids)
{
	if (unique_ids)
	{
		void *base = (void*) &(AddressTable[UNIQUE_TYPE]->address[1]);

		qsort (base, AddressTable[UNIQUE_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);
	}
	else
	{
		void *base = (void*) &(AddressTable[OUTLINED_OPENMP_TYPE]->address[1]);
		qsort (base, AddressTable[OUTLINED_OPENMP_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[MPI_CALLER_TYPE]->address[1]);
		qsort (base, AddressTable[MPI_CALLER_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[SAMPLE_TYPE]->address[1]);
		qsort (base, AddressTable[SAMPLE_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[USER_FUNCTION_TYPE]->address[1]);
		qsort (base, AddressTable[USER_FUNCTION_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[OTHER_FUNCTION_TYPE]->address[1]);
		qsort (base, AddressTable[OTHER_FUNCTION_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[CUDAKERNEL_TYPE]->address[1]);
		qsort (base, AddressTable[CUDAKERNEL_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);

		base = (void*) &(AddressTable[HIPKERNEL_TYPE]->address[1]);
		qsort (base, AddressTable[HIPKERNEL_TYPE]->num_addresses-1,
		       sizeof(struct address_info), Address2Info_Sort_routine);
	}

	// Cached entries are now invalid as everything gets re-sorted
	Addr2Info_HashCache_Clean();
}


/**********************************************\
 *                                            *
 *   API TO TRANSLATE DATA OBJECT ADDRESSES   *
 *                                            *
\**********************************************/

#if defined(HAVE_LIBADDR2LINE)

#if defined(LEGACY_BEHAVIOR)
// Set to 1 to retain legacy behavior: only translate PEBS sample addresses for the main binary
# define TRANSLATE_DATA_ADDRESSES_FOR_MAIN_BINARY_ONLY 1
#else
// Set to 0 to attempt translation of PEBS sample addresses for any mapping
# define TRANSLATE_DATA_ADDRESSES_FOR_MAIN_BINARY_ONLY 0
#endif

/**
 * get_mapping_name_and_offset_for_address
 * 
 * This function retrieves the maps file associated with the given (ptask, task),
 * uses libmaps to search for the mapping entry that contains the given address, 
 * and returns the mapping name and the relative offset in the mapping.
 * 
 * @param ptask The ptask identifier
 * @param task The task identifier
 * @param address The address to search for
 * @param[out] mapping_name The name of the mapping if found, or UNKNOWN_MAPPING otherwise
 * @param[out] offset The relative offset in the mapping if found, or the absolute address otherwise
 * @return 1 if the mapping was found, 0 otherwise
 */
static int get_mapping_name_and_offset_for_address(unsigned ptask, unsigned task, UINT64 address, char **mapping_name, UINT64 *offset)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);
	// First find the mapping entry that contains the address
	maps_entry_t *mapping = search_in_exec_mappings(task_info->proc_self_maps, address);

	*mapping_name = strdup(mapping_path(mapping));

	if (mapping == NULL) {
		*offset = address;
		return 0;
	}
	else {
		// If found, get the offset
		*offset = absolute_to_relative(mapping, address);
		return 1;
	}
}

/**
 * translate_data_address
 * 
 * Translates a data address (i.e. PEBS sample) to a symbol name.
 * This function finds the mapping entry for the address in the task's memory mappings.
 * Then it checks if the mapping entry corresponds to the main binary or not, and whether we want to translate for this mapping.
 * If so, it uses the symbol table to find the symbol name corresponding to the address. 
 *
 * In case translation for data symbols is disabled, 
 * mapping_name is equally resolved, but symbol_name is set to the symbol address.
 *
 * @param address The address to translate
 * @param ptask The ptask identifier
 * @param task The task identifier
 * @param mapping_name[out] The name of the mapping entry that contains the address
 * @param symbol_name[out] The name of the static symbol corresponding to the address taken from its mapping's symbol table
 */
static void translate_data_address (UINT64 address, unsigned ptask, unsigned task, char **mapping_name, char **symbol_name)
{
	*mapping_name = UNKNOWN_MAPPING;
	*symbol_name = UNKNOWN_SYMBOL;

	if (!Translation_Module_Initialized) return;

	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);

	// Find the mapping entry for the address
	maps_entry_t *entry = search_in_exec_mappings(task_info->proc_self_maps, address);
	if (entry != NULL)
	{
		char *main_binary = ObjectTree_getMainBinary(ptask, task);
		// Check if we should only translate addresses for the main binary
		if ((!TRANSLATE_DATA_ADDRESSES_FOR_MAIN_BINARY_ONLY) || (main_binary && strcmp(mapping_path(entry), main_binary) == 0))
		{
			if (get_option_merge_TranslateDataAddresses())
			{
#if defined(HAVE_ELFUTILS)
				*symbol_name = symtab_translate(entry->symtab, address);
#endif
			}
			else
			{
				char caddress[32];

				snprintf(caddress, sizeof(caddress), "%p", address);
				*symbol_name = strdup(caddress);
			}
			*mapping_name = mapping_path(entry);
		}
	}
}

#endif /* HAVE_LIBADDR2LINE */

/**
 * Address2Info_Translate_MemReference
 *
 * This function is analogous to Address2Info_Translate, but it is used to translate
 * data object addresses (PEBS samples) instead of function addresses.
 *
 * The behavior of this function differs depending on the value of the 'query' parameter:
 * - If 'query' == MEM_REFERENCE_DYNAMIC, the sampled 'address' fell within a heap-allocated
 *                                        memory region (e.g., obtained via malloc or a similar 
 *                                        allocation function). In this case, the function translates
 *                                        the address as the call-path of the function that allocated 
 *                                        the data object. 
 * - If 'query' == MEM_REFERENCE_STATIC,  the sampled 'address' fell within a static memory region
 *                                        (e.g., a global variable). In this case, the function translates
 *                                        the address as the name of the variable accessed, taken from
 *                                        the corresponding mapping's symbol table.
 *
 * In both cases, the translation is stored in a list through AddressTable_Insert_MemReference 
 * that correlates the address' translation (callpath or variable name) to its unique identifier.
 * 
 * NOTE: Different address samples that fall on the same memory region (either static or dynamic allocation)
 *       will translate into the same memory reference, example (for dynamic allocation):
 * 
 *            0xbaseaddr    addr1            addr2         0xbaseaddr+1000
 *                V           V                V                 V
 * malloc(1000) = [                                              ]
 *    ^
 *  main:15 > utils:30 (this unique callpath identifies where malloc was called and gets id: 1)	
 *
 * Because multiple samples may translate into the same callpath (or variable name for static allocations),
 * we need to get the translation of the address first, and then check if it is already in the list of translations
 * to reuse the identifier. 
 * 
 * FIXME: This function is lacking a cache for an already translated 'address'.
 *        Before attempting to translate the whole call-stack again, we should 
 *        check if the address is already in the cache.
 *
 * @param ptask The application identifier
 * @param task The task identifier
 * @param address The address to translate
 * @param query Indicates the category of the address being translated
 *              (either MEM_REFERENCE_DYNAMIC or MEM_REFERENCE_STATIC)
 * @param calleraddresses An array of addresses representing the call stack
 *                        of the function that allocated the data object.
 * @return An unique identifier for the translated address, or the original address if the translation fails.
 */
UINT64 Address2Info_Translate_MemReference (unsigned ptask, 
                                            unsigned task,
                                            UINT64 address,
                                            int query,
                                            UINT64 *calleraddresses)
{
#if !defined(HAVE_LIBADDR2LINE)
	UNREFERENCED_PARAMETER(ptask);
	UNREFERENCED_PARAMETER(task);
	UNREFERENCED_PARAMETER(address);
	UNREFERENCED_PARAMETER(query);
	UNREFERENCED_PARAMETER(calleraddresses);

	return address;
#else /* HAVE_LIBADDR2LINE */
	char * module = NULL;

# if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": Address2Info_TranslateMemreference (%u, %u, %lx, %d);\n",
	         ptask, task, address, query);
# endif

	if (query == MEM_REFERENCE_DYNAMIC)
	{
		// The 'address' to translate refers belongs to heap-allocated memory region (e.g., obtained via malloc or a similar allocation function).
		int line;
		char * sname = NULL;
		char * filename = NULL;
		char buffer[2048];
		char tmp[1024];
		int i;

		if (get_option_merge_TranslateDataAddresses())
		{
			/*
			 * When the flag -translate-data-addresses is set, the sampled address will be translated
			 * as the the call-stack of the function that allocated the memory region to which it belongs,
			 * i.e., accesses to dynamically allocated objects are represented by their malloc or similar call.
			 * TODO: In this case, the 'address' argument is not used at all, maybe we should refactor this.
			 */
			snprintf (buffer, sizeof(buffer), "");

			// Iterate from the start to remove any leading "unresolved" frames
			for (i = 0; i < MAX_CALLERS; i++)
			{
				if (calleraddresses[i] != 0)
				{
					translate_function_address (calleraddresses[i], ptask, task, &module,
					  &sname, &filename, &line);
					if (!strcmp (filename, UNKNOWN_ADDRESS)) calleraddresses[i] = 0;
					else break;
				}
			}
			// Iterate from the end to remove any trailing "unresolved" frames
			for (i = MAX_CALLERS-1; i >= 0; i--)
			{
				if (calleraddresses[i] != 0)
				{
					translate_function_address (calleraddresses[i], ptask, task, &module,
					  &sname, &filename, &line);
					if (!strcmp (filename, UNKNOWN_ADDRESS)) calleraddresses[i] = 0;
					else break;
				}
			}
			// Build a human-readable call-stack string, skipping any "unresolved" frames on both ends
			for (i = 0; i < MAX_CALLERS; i++)
			{
				if (calleraddresses[i] != 0)
				{
					translate_function_address (calleraddresses[i], ptask, task, &module,
					  &sname, &filename, &line);

					// Format each frame as "filename:line" and separate frames with token " > "
					snprintf(tmp, sizeof(tmp), "%s%s:%d", (strlen(buffer) > 0 ? " > " : ""), filename, line);
					strncat (buffer, tmp, sizeof(buffer));
				}
				// Example output for 'buffer': main.c:42 > util.c:128 > io.c:75
			}
		}
		else
		{
			/*
			 * If the -translate-data-addresses option is disabled, the sampled address still gets translated
			 * as the call-stack of the function that allocated the memory region to which it belongs,
			 * but instead of printing each frame as a reference to the sources with "filename:line",
			 * we emit it as a sequence of "mapping_name+relative_offset" pairs, where:
			 * - mapping_name is the binary or shared library name from /proc/self/maps
			 * - offset       is the relative address within that mapping
			 */
			buffer[0] = (char)0;
			for (i = 0; i < MAX_CALLERS; i++)
			{
				if (calleraddresses[i] != 0)
				{
					UINT64 offset;

					// 'module' keeps the last valid mapping name in the callstack 
					free(module);
					// Find the memory mapping to which the sampled address belongs
					get_mapping_name_and_offset_for_address(ptask, task, calleraddresses[i], &module, &offset);

					// Format each frame as "mapping_name+offset" and separate frames with token " > "
					snprintf(tmp, sizeof(tmp), "%s%s+%08lx", (strlen(buffer) > 0 ? " > " : ""), module, offset);
					strncat (buffer, tmp, sizeof(buffer));
					// Example output for 'buffer': main.exe+0x0015A2 > libutil.so+0x00F3B8 > libio.so+0x0A1C4F
				}
			}
		}
		/* 
		 * FIXME: ASK: At this point 'module' contains the last valid mapping name in the callstack for a dynamic sample
		 *             Is this really used by the HetMem group? Ask mjorda about this.
		 *             If not, I'm inclined to set it to a constant such as "<HeapBlock>" to indicate
		 *             that this sample fell in a dynamic memory region, or look for it in the mapping table
		 *             to find the actual [heap] section (if this section is not anonymous).
		 */

		// Get an unique ID for the dynamic memory reference (i.e., the call-stack string in 'buffer')
		return 1+AddressTable_Insert_MemReference (query, strdup(buffer), module);
	}
	else if (query == MEM_REFERENCE_STATIC)
	{
		/*
 		 * The address to translate belongs to a static allocation (e.g., a global variable).
		 * FIXME: It could belong to an untracked malloc that was filtered because the size allocated was too small!
		 *        See Sampling_Address_Event at misc_prv_semantics.c where the distinction between static and dynamic is made. 
		 */
		char *varname = NULL;

		// Lookup the symbol table to translate the address to a variable name (e.g. 'global_array')
		translate_data_address (address, ptask, task, &module, &varname);
		
		// Get an unique ID for the static memory reference (i.e., the variable name in 'varname')
		return 1+AddressTable_Insert_MemReference (query, varname, module);
	}
	else return address;

#endif /* HAVE_LIBADDR2LINE */
}

/**
 * AddressTable_Insert_MemReference
 *
 * This function converts a translated memory reference (i.e. the call-stack string 
 * of the function that allocates the memory for a sample that falls on a dynamically allocated region,
 * or the symbol name for a sample that falls on a statically allocated region), 
 * into an unique identifier.
 *
 * The memory reference is stored in the AddressObjectInfo structure, which serves as
 * a translation list for data samples. Before inserting a new entry, the function checks
 * if the memory reference already exists in the list. If it does, it returns the index
 * of the existing entry. If it doesn't, it creates a new entry and returns its index.
 *
 * Both static and dynamic memory references are stored in the same list, 
 * FIXME: which is extremely inefficient for searching.
 * 
 * @param addr_type The type of memory reference (static or dynamic)
 * @param data_reference This is the name of the variable for samples accessing static memory regions,
 *                       or the call-stack string of the function that allocated a dynamic memory region.
 * @param mapping_name The path of the mapping that contains the sampled address
 * @return The index of the memory reference in the AddressObjectInfo list
 */
static int AddressTable_Insert_MemReference (int addr_type,
	const char *data_reference,
	const char *mapping_name)
{
	int i = 0; 
	int found = FALSE;

#if defined(DEBUG) 
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s)\n",
	         addr_type, data_reference, mapping_name);
#endif

	int nObjects = AddressObjectInfo.num_objects;

	// First search if this memory reference is already in the list
	for (i = 0; i < nObjects; i++)
	{
		struct address_object_info_st * object = &(AddressObjectInfo.objects[i]);
		if (addr_type == object->static_or_dynamic) {
			found = (!strcmp(data_reference, object->data_reference) && !strcmp(mapping_name, object->module));
		}
		if (found) {
#if defined(DEBUG)
			fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s) -> %d\n",
			         addr_type, data_reference, mapping_name, i);
#endif
			return i;
		}
	}

	// Add a new entry to the list if it was not found
	AddressObjectInfo.objects = (struct address_object_info_st*) xrealloc (
	  AddressObjectInfo.objects,
	  (AddressObjectInfo.num_objects+1)*sizeof(struct address_object_info_st));

	i = AddressObjectInfo.num_objects;
	AddressObjectInfo.objects[i].static_or_dynamic = addr_type;
	AddressObjectInfo.objects[i].data_reference = data_reference;
	AddressObjectInfo.objects[i].module = mapping_name;
	AddressObjectInfo.num_objects++;

#if defined(DEBUG)
	fprintf (stderr, PACKAGE_NAME": AddressTable_Insert_MemReference (%d, %s, %s) -> %d\n",
	         addr_type, data_reference, mapping_name, i);
#endif

	return i;
}


/**************************************************\
 *                                                *
 *   API RELATED TO OPTION -emit-library-events   *
 *                                                *
\**************************************************/

#if defined(HAVE_LIBADDR2LINE)

// List of unique library names from all ranks
library_id_t unifiedLibraryIDs = { NULL, 0 };

/**
 * find_global_library_id
 *
 * Find the global identifier of the given library name
 *
 * @param look_for_library The name of the library to look for
 * @return int The global identifier of the library
 */
static int find_global_library_id (const char *look_for_library)
{
	int i = 0;

	for (i = 0; i < unifiedLibraryIDs.num_libraries; ++i)
	{
		if (strcmp(look_for_library, unifiedLibraryIDs.library_names[i]) == 0) return i;
	}
	return -1;
}

/**
 * insert_global_library
 *
 * Insert a new library name into the global list of libraries
 *
 * @param library_name The name of the library to insert
 */
static void insert_global_library (const char *library_name)
{
	int index = unifiedLibraryIDs.num_libraries;

	unifiedLibraryIDs.library_names = (char **) xrealloc (
		unifiedLibraryIDs.library_names,
		(index + 1) * sizeof (char *)
	);
	unifiedLibraryIDs.library_names[index] = strdup(library_name);
	unifiedLibraryIDs.num_libraries ++;
}

/** 
 * unify_library_ids
 * 
 * Iterate over all (ptask, task), get the pathname of each mapping entry, and build a list of unique libraries.
 * The index of the library in the list will be the library id. 
 * Store the list at the level of ptask not to mix libraries from different applications.
 */
static void unify_library_ids()
{
	int current_app = 0;
	int nptasks = ObjectTree_getNumPtasks();
	for (current_app = 1; current_app <= nptasks; ++current_app)
	{
		int current_task = 0;
		int ntasks = ObjectTree_getNumTasks(current_app);
		for (current_task = 1; current_task <= ntasks; ++current_task)
		{
			task_t *task_info = ObjectTree_getTaskInfo(current_app, current_task);

			maps_entry_t *current_exe = exec_mappings(task_info->proc_self_maps);
			while (current_exe != NULL)
			{
				// current_exe->pathname is the library name, we need to check if it is already in the list
				// The list is stored at the level of the ptask to unify duplicates from different tasks
				int found = (find_global_library_id(current_exe->pathname) != -1);
				if (!found)
				{
					// Add the library to the list
					insert_global_library(current_exe->pathname);
				}
				current_exe = next_exec_mapping(current_exe);
			}
		}
	}
}

/** 
 * Address2Info_GetLibraryID
 *
 * Gets the unique identifier of the library that contains the given address.
 *
 * @param ptask The application identifier
 * @param task The task identifier
 * @param address The address to search for
 * @return The unique identifier of the library that contains the address; 0 if not found.
 */
UINT64 Address2Info_GetLibraryID (unsigned ptask, unsigned task, UINT64 address)
{
	task_t *task_info = ObjectTree_getTaskInfo(ptask, task);

	// First find the mapping entry that contains the address
	maps_entry_t *exe = search_in_exec_mappings(task_info->proc_self_maps, address);
	if (exe) {
		// Find the global ID of this mapping
		return find_global_library_id(exe->pathname) + 1; // +1 because 0 is reserved for Unknown	
	}
	return 0;
}

void Address2Info_Write_LibraryIDs (FILE *pcf_fd)
{
	int i = 0;
	fprintf (pcf_fd, "%s\n", TYPE_LABEL);
	fprintf (pcf_fd, "0    %d    %s\n", LIBRARY_EV, LIBRARY_LBL);
	fprintf (pcf_fd, "%s\n", VALUES_LABEL);
	fprintf (pcf_fd, "0    Unknown\n");

	for (i = 0; i < unifiedLibraryIDs.num_libraries; ++i)
	{
		fprintf (pcf_fd, "%d    %s\n", i + 1, unifiedLibraryIDs.library_names[i]);
	}
	LET_SPACES(pcf_fd);
}

#endif /* HAVE_LIBADDR2LINE */


/**********************\
 *                    *
 *   PCF GENERATION   *
 *                    *
\**********************/

static void prettify_function(char *function, char **short_label, char **long_label)
{
	char *short_str = strdup(function);
	char *long_str = NULL;

	char *open_template = strchr(short_str, '<');
	if (open_template) *open_template = '\0';
	else
	{
		char *open_bracket = strchr(short_str, '(');
		if (open_bracket) *open_bracket = '\0';
	}
	if (strcmp(short_str, function) != 0) long_str = strdup(function);

	*short_label = short_str;
	*long_label = long_str;
}

static void prettify_file_line(char *module, char *file, int line, char **short_label, char **long_label)
{
	char *file_copy = strdup(file);
	char *file_basename = strdup(basename(file_copy));
	free(file_copy);

	char short_str[1024];
	snprintf(short_str, sizeof(short_str), "%s:%d", file_basename, line);
	free(file_basename);

	char long_str[4096];
	snprintf(long_str, sizeof(long_str), "%s:%d, %s", file, line, module);

	*short_label = strdup(short_str);
	*long_label = strdup(long_str);
}

void Address2Info_Write_MPI_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:MPI_CALLER_TYPE];

	if (Address2Info_Labels[A2I_MPI]) 
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
			fprintf(pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i=0; i<FuncTab->num_functions; i++) 
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n", 
				        i + 1, 
					short_label, 
					(long_label != NULL ? "[" : ""),
					(long_label != NULL ? long_label : ""), 
					(long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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
			fprintf(pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_OMP_Labels (FILE * pcf_fd, int eventtype,
	char *eventtype_description, int eventtype_line,
	char *eventtype_line_description, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:OUTLINED_OPENMP_TYPE];

	if (Address2Info_Labels[A2I_OMP]) 
	{
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		fprintf(pcf_fd, "0    %d    %s\n", eventtype, eventtype_description);

		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		fprintf(pcf_fd, "0    %d    %s\n", eventtype_line, eventtype_line_description);
		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:CUDAKERNEL_TYPE];

	if (Address2Info_Labels[A2I_CUDA]) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDA_KERNEL_EXEC_EV, "CUDA kernel execution");
		fprintf (pcf_fd, "0    %d    %s\n", CUDA_KERNEL_INST_EV, "CUDA kernel instantiation");

		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", CUDA_KERNEL_EXEC_LINE_EV, "CUDA kernel execution source code line");
		fprintf (pcf_fd, "0    %d    %s\n", CUDA_KERNEL_INST_LINE_EV, "CUDA kernel instantiation source code line");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);

			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_HIP_Labels (FILE * pcf_fd, int uniqueid)
{
	struct address_table  * AddrTab;
	struct function_table * FuncTab;
	int i;

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:HIPKERNEL_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:HIPKERNEL_TYPE];

	if (Address2Info_Labels[A2I_HIP]) 
	{
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", HIP_KERNEL_INST_EV, "HIP kernel instantiation");
		fprintf (pcf_fd, "0    %d    %s\n", HIP_KERNEL_EXEC_EV, "HIP kernel execution");

		if (Address2Info_Initialized())
		{
			fprintf(pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", HIP_KERNEL_INST_LINE_EV, "HIP kernel instantiation source code line");
		fprintf (pcf_fd, "0    %d    %s\n", HIP_KERNEL_EXEC_LINE_EV, "HIP kernel execution source code line");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);

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

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:USER_FUNCTION_TYPE];

	if (Address2Info_Labels[A2I_UF]) 
	{
		/* First dump functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_EV, "User function");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}

		/* Then dump line-functions */
		fprintf (pcf_fd, "%s\n", TYPE_LABEL);
		fprintf (pcf_fd, "0    %d    %s\n", USRFUNC_LINE_EV, "User function line");

		if (Address2Info_Initialized())
		{
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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

	AddrTab = AddressTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];
	FuncTab = FunctionTable[uniqueid?UNIQUE_TYPE:SAMPLE_TYPE];

	if (Address2Info_Labels[A2I_SAMPLE]) 
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
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < FuncTab->num_functions; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_function(FuncTab->function[i], &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
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
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);
			for (i = 0; i < AddrTab->num_addresses; i ++)
			{
				char *short_label = NULL, *long_label = NULL;
				prettify_file_line(AddrTab->address[i].module, AddrTab->address[i].file_name, AddrTab->address[i].line, &short_label, &long_label);
				fprintf(pcf_fd, "%d %s %s%s%s\n",
				        i + 1,
				        short_label,
				        (long_label != NULL ? "[" : ""),
				        (long_label != NULL ? long_label : ""),
				        (long_label != NULL ? "]" : ""));
				free(short_label);
				free(long_label);
			}
			LET_SPACES(pcf_fd);
		}
	}
}

void Address2Info_Write_MemReferenceCaller_Labels (FILE * pcf_fd)
{
	int i;

	if (Address2Info_Initialized())
	{
		fprintf(pcf_fd, "%s\n", TYPE_LABEL);
		fprintf(pcf_fd, "0    %d    %s\n", SAMPLING_ADDRESS_ALLOCATED_OBJECT_EV,
		  SAMPLING_ADDRESS_ALLOCATED_OBJECT_LBL);
		fprintf(pcf_fd, "0    %d    %s\n", SAMPLING_ADDRESS_ALLOCATED_OBJECT_ALLOC_EV,
		  SAMPLING_ADDRESS_ALLOCATED_OBJECT_ALLOC_LBL);

		if (AddressObjectInfo.num_objects > 0)
			fprintf (pcf_fd, "%s\n0 %s\n", VALUES_LABEL, EVT_END_LBL);

		for (i = 0; i < AddressObjectInfo.num_objects; i++)
		{
			struct address_object_info_st *obj = &(AddressObjectInfo.objects[i]);

			fprintf (pcf_fd, "%d %s\n", i+1, obj->data_reference);
		}

		if (AddressObjectInfo.num_objects > 0)
			LET_SPACES(pcf_fd);
	}
}

#if defined(HAVE_LIBADDR2LINE_LIB_symtab)

/**
 * Address2Info_writeDataObjects
 * 
 * Writes the symbol table of the main binary to the output PCF file.
 * This is currently limited to the main binary of the first application and task,
 * but could be extended to all applications and tasks, as well as to all other libraries.
 * The information produced is only used by Folding, at it is not clear for what purpose, 
 * so we won't extend it for now.  
 * 
 * @param fd File descriptor of the PCF file to write the information to
 * @param eventstart Event number to start writing the information from
 */
void Address2Info_writeDataObjects (FILE *fd, unsigned eventstart)
{
	unsigned _ptask, _task;

	// Loop over all apps & tasks, currently limited to the first one
	for (_ptask = 1; _ptask <= 1 /* ApplicationTable.nptasks */; _ptask++)
	{
		for (_task = 1; _task <= 1 /* ApplicationTable.ptasks[_ptask].ntasks */; _task++)
		{
			char *main_binary = ObjectTree_getMainBinary(_ptask, _task);

			/* 
			 * Here we read the symtab directly from the main binary.
			 * Alternatively, we could reuse the symtab from the mapping entry that corresponds to the main binary,
			 * which is already dumped in task_info->proc_self_maps (field symtab of mapping_entry_t structure). 
			 * 
			 * For this, we would need to iterate over the executable mapping entries (exec_mappings(task_info->proc_self_maps))
			 * and select the one that matches the main binary by name (strcmp(mapping->pathname, main_binary) == 0). 
			 * We stick with the first approach for now, as it is simpler, although not faster.
			 */
			symtab_t *main_binary_symbols = symtab_read(main_binary);
			int i = 0, num_symbols = symtab_count(main_binary_symbols);
			if (num_symbols > 0)
			{
				fprintf (fd, "EVENT_TYPE\n");
				fprintf (fd, "0 %u Symbol table for task %u.%u\n", eventstart++, _ptask, _task);
				fprintf (fd, "VALUES\n");
				for (i = 0; i < num_symbols; ++i)
				{
					symtab_entry_t *se = symtab_get_entry(main_binary_symbols, i);

					fprintf (fd, "%u %s [0x%08lx-0x%08lx)\n",
						i+1,
						se->name,
						se->start,
						se->end); /* This was end-1, but libsymtab searches are already non-inclusive of the end address.
								   * It is now expressed in math notation [start_addr, end_addr), and more consistent with C/ELF conventions
								   */
				}
				fprintf (fd, "\n");
			}
		}
	}
}

#endif /* HAVE_LIBADDR2LINE_LIB_symtab */


/*******************\
 *                 *
 *   UNIFICATION   *
 *                 *
\*******************/

#if defined(PARALLEL_MERGE)

#include <mpi.h>
#include "mpi-aux.h"

/**
 * Share_Callers_Usage
 */
void Share_Callers_Usage (void)
{
	int MPI_used[MAX_CALLERS], SAMPLE_used[MAX_CALLERS];
	int A2I_tmp[A2I_LAST];
	int i, res;

	res = MPI_Reduce (Address2Info_Labels, A2I_tmp, A2I_LAST, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK (res, MPI_Reduce, "Sharing information about address <-> info labels");
	for (i = 0; i < A2I_LAST; i++)
		Address2Info_Labels[i] = A2I_tmp[i];

	if (MPI_Caller_Labels_Used == NULL)
	{
		MPI_Caller_Labels_Used = xmalloc(sizeof(int)*MAX_CALLERS);
		for (i = 0; i < MAX_CALLERS; i++)
			MPI_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (MPI_Caller_Labels_Used, MPI_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about MPI address <-> info");
	for (i = 0; i < MAX_CALLERS; i++)
		MPI_Caller_Labels_Used[i] = MPI_used[i];

	if (Sample_Caller_Labels_Used == NULL)
	{
		Sample_Caller_Labels_Used = xmalloc(sizeof(int)*MAX_CALLERS);
		for (i = 0; i < MAX_CALLERS; i++)
			Sample_Caller_Labels_Used[i] = FALSE;
	}
	res = MPI_Reduce (Sample_Caller_Labels_Used, SAMPLE_used, MAX_CALLERS, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about sampling address <-> info");
	for (i = 0; i < MAX_CALLERS; i++)
		Sample_Caller_Labels_Used[i] = SAMPLE_used[i];

	res = MPI_Reduce (&MPI_Caller_Multiple_Levels_Traced, &i, 1, MPI_INT, MPI_BOR, 0, MPI_COMM_WORLD);
	MPI_CHECK(res, MPI_Reduce, "Sharing information about multiple address <-> info labels");
	MPI_Caller_Multiple_Levels_Traced = i;
}

#endif /* PARALLEL_MERGE */

/**
 * Addr2Info_Unify
 * 
 * This routine combines the information from all the processes into the merger main process.
 * Also performs the unification of the identifiers assigned to the translations.
 * This is done by sending all the addresses seen while processing events to the merger main process,
 * then this only process does all the translations by itself. 
 * FIXME: We really need to change this into a more scalable approach.
 * 
 * @param numtasks Number of merger tasks
 * @param taskid Rank identifier of the current merger task
 * @param CollectedAddresses Pointer to the structure that contains the addresses collected while parsing the mpit files
 */
void Address2Info_Unify(int numtasks, int taskid, struct address_collector_t *CollectedAddresses)
{
	if (get_option_merge_SortAddresses())
	{
#if defined(PARALLEL_MERGE)
		AddressCollector_GatherAddresses (numtasks, taskid, CollectedAddresses);
#endif
		/* Address translation and sorting is only done by the master */
		if (taskid == 0)
		{
			UINT64 *buffer_addresses = AddressCollector_GetAllAddresses (CollectedAddresses);
			int *buffer_types = AddressCollector_GetAllTypes (CollectedAddresses);
			unsigned *buffer_ptasks = AddressCollector_GetAllPtasks (CollectedAddresses);
			unsigned *buffer_tasks = AddressCollector_GetAllTasks (CollectedAddresses);

			for (int i = 0; i < AddressCollector_Count(CollectedAddresses); i++)
				Address2Info_Translate (buffer_ptasks[i], buffer_tasks[i],
				  buffer_addresses[i], buffer_types[i], get_option_merge_UniqueCallerID());

			Address2Info_Sort (get_option_merge_UniqueCallerID());
		}
	}

#if defined(HAVE_LIBADDR2LINE)
	if (get_option_merge_EmitLibraryEvents()) {
		unify_library_ids();
	}
#endif
}
