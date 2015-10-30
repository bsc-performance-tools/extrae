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

/* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- *\
 | @file: $HeadURL: https://svn.bsc.es/repos/ptools/extrae/branches/2.4/src/tracer/wrappers/API/wrapper.c $
 | @last_commit: $Date: 2013-11-26 10:30:20 +0100 (Tue, 26 Nov 2013) $
 | @version:     $Revision: 2336 $
\* -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- */
#include "common.h"

static char UNUSED rcsid[] = "$Id: wrapper.c 2336 2013-11-26 09:30:20Z harald $";

#ifdef HAVE_STDIO_H
# include <stdio.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif

#include <es_bsc_cepbatools_extrae_Wrapper.h>
#include <extrae_user_events.h>
#include <extrae_types.h>

int THREADID = 0;
int NUMTHREADS = 1;
int TASKID = 0;
int NUMTASKS = 1;

JavaVM* javaVM = NULL;
jclass activityClass;
jobject activityObj;

static unsigned int get_thread_id(void)
{ return THREADID; }

static unsigned int get_num_threads(void)
{ return NUMTHREADS; }

static unsigned int get_task_id(void)
{ return TASKID; }

static unsigned int get_num_tasks(void)
{ return NUMTASKS; }


JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetTaskID(
	JNIEnv *env, jclass jc, jint id)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	TASKID = id;
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
	JNIEnv *env, jclass jc, jint numthreads)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	NUMTHREADS = numthreads;
	Extrae_set_numtasks_function (get_num_tasks);
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetNumTasks(
	JNIEnv *env, jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return get_num_tasks();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetThreadID(
	JNIEnv *env, jclass jc, jint id)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	THREADID = id;
	Extrae_set_threadid_function (get_thread_id);
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetThreadID(
	JNIEnv *env, jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return get_thread_id();
}

JNIEXPORT void JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_SetNumThreads(
	JNIEnv *env, jclass jc, jint numthreads)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	NUMTHREADS = numthreads;
	Backend_ChangeNumberOfThreads (numthreads);
	Extrae_set_numthreads_function (get_num_threads);
}

JNIEXPORT jint JNICALL Java_es_bsc_cepbatools_extrae_Wrapper_GetNumThreads(
	JNIEnv *env, jclass jc)
{
	UNREFERENCED(env);
	UNREFERENCED(jc);

	return get_num_threads();
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
			Extrae_nevent (countTypes, (extrae_type_t *)typesArray, (extrae_value_t *)valuesArray);
		(*env)->ReleaseIntArrayElements(env, types, typesArray, JNI_ABORT);
		(*env)->ReleaseLongArrayElements(env, values, valuesArray, JNI_ABORT);
	}
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
