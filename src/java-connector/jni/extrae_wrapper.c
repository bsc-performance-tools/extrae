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
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#include <wrapper.h>

#include "es_bsc_cepbatools_extrae_Wrapper.h"
#include "extrae_user_events.h"

static int __TASKID = 0;
static int __NUMTASKS = 1;

jclass activityClass;
jobject activityObj;

static unsigned int get_task_id(void)
{ return __TASKID; }

static unsigned int get_num_tasks(void)
{ return __NUMTASKS; }


JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetTaskID(
	JNIEnv *env, jclass jc, jint id)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	__TASKID = id;
	Extrae_set_taskid_function (get_task_id);
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetTaskID(
	JNIEnv *env, jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return get_task_id();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetNumTasks(
	JNIEnv *env, jclass jc, jint numtasks)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	__NUMTASKS = numtasks;
	Extrae_set_numtasks_function (get_num_tasks);
	Extrae_Allocate_Task_Bitmap (numtasks);
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetNumTasks(
	JNIEnv *env, jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return get_num_tasks();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Init(JNIEnv *env,
	jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_init();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Fini (JNIEnv *env,
	jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_fini();
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetPID(JNIEnv *env,
	jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return getpid();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Event (JNIEnv *env,
	jclass jc, jint id, jlong val)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_event ((extrae_type_t)id, (extrae_value_t)val);
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Eventandcounters (
	JNIEnv *env, jclass jc, jint id, jlong val)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_eventandcounters ((extrae_type_t)id, (extrae_value_t)val);
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_nEvent (JNIEnv *env,
	jclass jc, jintArray types, jlongArray values)
{
	jint countTypes = 0;
	jint countValues = 0;

	UNREFERENCED(jc);

	jint *typesArray = (*env)->GetIntArrayElements(env, types, NULL);
	jlong *valuesArray = (*env)->GetLongArrayElements(env, values, NULL);

	if (typesArray != NULL && valuesArray != NULL)
	{
		countTypes = (*env)->GetArrayLength(env, types);
		countValues = (*env)->GetArrayLength(env, values);
		if (countTypes == countValues)
			Extrae_nevent (countTypes,
			  (extrae_type_t *)typesArray, (extrae_value_t *)valuesArray);
		(*env)->ReleaseIntArrayElements(env, types, typesArray, JNI_ABORT);
		(*env)->ReleaseLongArrayElements(env, values, valuesArray, JNI_ABORT);
	}
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_nEventandcounters (
	JNIEnv *env, jclass jc, jintArray types, jlongArray values)
{
	jint countTypes = 0;
	jint countValues = 0;

	UNREFERENCED(jc);

	jint *typesArray = (*env)->GetIntArrayElements(env, types, NULL);
	jlong *valuesArray = (*env)->GetLongArrayElements(env, values, NULL);

	if (typesArray != NULL && valuesArray != NULL)
	{
		countTypes = (*env)->GetArrayLength(env, types);
		countValues = (*env)->GetArrayLength(env, values);
		if (countTypes == countValues)
			Extrae_neventandcounters (countTypes,
			  (extrae_type_t *)typesArray, (extrae_value_t *)valuesArray);
		(*env)->ReleaseIntArrayElements(env, types, typesArray, JNI_ABORT);
		(*env)->ReleaseLongArrayElements(env, values, valuesArray, JNI_ABORT);
	}
}


JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Comm (JNIEnv *env,
	jclass jc, jboolean send, jint tag, jint size, jint partner, jlong id)
{
	struct extrae_UserCommunication comm;
	struct extrae_CombinedEvents events;

	UNREFERENCED(env);
	UNREFERENCED(jc);

 	Extrae_init_UserCommunication(&comm);
	Extrae_init_CombinedEvents(&events);

	if(send)
		comm.type = EXTRAE_USER_SEND;
	else
		comm.type = EXTRAE_USER_RECV;

	comm.tag=tag;
	comm.size=size;
	comm.partner=partner;
	comm.id=id;

	events.nCommunications=1;	
	events.Communications=&comm;	
	events.nEvents=0;

	Extrae_emit_CombinedEvents(&events);
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetOptions(JNIEnv *env,
	jclass jc, jint options)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_set_options (options);
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Shutdown(JNIEnv *env,
	jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_shutdown();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_Restart (JNIEnv *env,
	jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_restart();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_defineEventType
	(JNIEnv *env, jclass jc, jint type, jstring description,
	jlongArray values, jobjectArray descriptionValues)
{
	char *cDescr;
	jint countValues, countDescriptionValues;

	UNREFERENCED(jc);

	if (values != NULL)
		countValues = (*env)->GetArrayLength(env, values);
	else
		countValues = 0;

	if (descriptionValues != NULL)
		countDescriptionValues = (*env)->GetArrayLength(env, descriptionValues);
	else
		countDescriptionValues = 0;

	cDescr = (char*)(*env)->GetStringUTFChars(env, description, NULL);
	if (cDescr != NULL)
	{
		if (countValues == countDescriptionValues && countValues > 0)
		{
			jlong *valuesArray = NULL;
			char *valDesc[countValues];

			valuesArray = (*env)->GetLongArrayElements(env, values, NULL);
			if (valuesArray != NULL)
			{
				jlong i;
				for (i = 0; i < countValues; i++)
				{
					valDesc[i] = (char*)(*env)->GetStringUTFChars(env,
					  (*env)->GetObjectArrayElement(env, descriptionValues, i),
					  NULL);
					if (valDesc[i] == NULL)
						return;
				}
				Extrae_define_event_type ((extrae_type_t *)&type, cDescr,
				  (unsigned *) &countValues, (extrae_value_t *)valuesArray,
			 	 valDesc);

				(*env)->ReleaseLongArrayElements(env, values, valuesArray,
				  JNI_ABORT);
				for (i = 0; i < countValues; i++)
				{
					jstring elem = (jstring)(*env)->GetObjectArrayElement (env,
					  descriptionValues, i);
					(*env)->ReleaseStringUTFChars(env, elem, valDesc[i]);
				}
			}
		}
		else
		{
			unsigned zero = 0;
			Extrae_define_event_type ((extrae_type_t*)&type, cDescr, &zero,
			  NULL, NULL);
		}
		(*env)->ReleaseStringUTFChars(env, description, cDescr);
	}
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_resumeVirtualThread
  (JNIEnv * env, jclass jc, jlong vthread)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_resume_virtual_thread((unsigned) vthread);
}

/*
 * Class:     es_bsc_cepbatools_extrae_Wrapper
 * Method:    SuspendVirtualThread
 * Signature: ()V
 */
JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_suspendVirtualThread
  (JNIEnv * env, jclass jc)
{
	
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_suspend_virtual_thread();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_registerFunctionAddres (
  JNIEnv *env, jclass jc, jlong id, jstring funcname, jstring modname, jint line)
{
	UNREFERENCED(jc);

	char *fname = (char*) (*env)->GetStringUTFChars (env, funcname, NULL) ;
	char *mname = (char*) (*env)->GetStringUTFChars (env, modname, NULL) ;

	Extrae_register_function_address ((void*) id, fname, mname, (unsigned) line);

	(*env)->ReleaseStringUTFChars(env, modname, mname);
	(*env)->ReleaseStringUTFChars(env, funcname, fname);
}


JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_functionEventFromAddress (
	JNIEnv *env, jclass jc, jlong address)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	Extrae_function_from_address (USRFUNC_EV, (void*) address);
}

