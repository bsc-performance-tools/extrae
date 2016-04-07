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
#ifdef HAVE_STRING_H
# include <string.h>
#endif

#undef UNUSED

#include <jni.h>
#include <jvmti.h>

#include "threadid.h"
#include "threadinfo.h"
#include "java_probe.h"

#define CHECK_JVMTI_ERROR(x,call) \
	{ if (x != JVMTI_ERROR_NONE) { fprintf (stderr, PACKAGE_NAME": Error during %s in %s:%d\n", #call, __FILE__, __LINE__); } }

/* Global static data */
static jvmtiEnv     *jvmti;
static jrawMonitorID ExtraeJ_AgentLock;

/* Callback for JVMTI_EVENT_GARBAGE_COLLECTION_START */
static void JNICALL Extraej_cb_GarbageCollector_begin (jvmtiEnv* jvmti_env)
{
	UNREFERENCED_PARAMETER(jvmti_env);

	Extrae_Java_GarbageCollector_begin();
}

/* Callback for JVMTI_EVENT_GARBAGE_COLLECTION_FINISH */
static void JNICALL Extraej_cb_GarbageCollector_end(jvmtiEnv* jvmti_env)
{
	UNREFERENCED_PARAMETER(jvmti_env);

	Extrae_Java_GarbageCollector_end();
}

#if 0
/* Callback for JVMTI_EVENT_VM_OBJECT_ALLOC */
static void JNICALL Extraej_cb_ObjectAlloc (jvmtiEnv *jvmti, JNIEnv *env,
	jthread thread, jobject object, jclass object_klass, jlong size)
{
	UNREFERENCED_PARAMETER(object_klass);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(env);

	jvmtiError r = (*jvmti)->SetTag(jvmti, object, 0x1234);
	CHECK_JVMTI_ERROR(r, SetTag);
	Extrae_Java_Object_Alloc (size);
}

/* Callback for JVMTI_EVENT_OBJECT_FREE */
static void JNICALL Extreaj_cb_ObjectFree (jvmtiEnv *jvmti, jlong tag)
{
	UNREFERENCED_PARAMETER(jvmti);
	UNREFERENCED_PARAMETER(tag);

	Extrae_Java_Object_Free ();
}
#endif

static void JNICALL Extraej_cb_Exception (jvmtiEnv *jvmti_env, JNIEnv* jni_env,
	jthread thread, jmethodID method, jlocation location, jobject exception,
	jmethodID catch_method, jlocation catch_location)
{
	UNREFERENCED_PARAMETER(jvmti_env);
	UNREFERENCED_PARAMETER(jni_env);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(method);
	UNREFERENCED_PARAMETER(location);
	UNREFERENCED_PARAMETER(exception);
	UNREFERENCED_PARAMETER(catch_method);
	UNREFERENCED_PARAMETER(catch_location);

	Extrae_Java_Exception_begin();
}

static void JNICALL Extraej_cb_ExceptionCatch (jvmtiEnv *jvmti_env,
	JNIEnv* jni_env, jthread thread, jmethodID method, jlocation location,
	jobject exception)
{
	UNREFERENCED_PARAMETER(jvmti_env);
	UNREFERENCED_PARAMETER(jni_env);
	UNREFERENCED_PARAMETER(thread);
	UNREFERENCED_PARAMETER(method);
	UNREFERENCED_PARAMETER(location);
	UNREFERENCED_PARAMETER(exception);

	Extrae_Java_Exception_end();
}

static void JNICALL Extraej_cb_ThreadStart (jvmtiEnv *jvmti_env,
	JNIEnv* jni_env, jthread thread)
{
/*
	jvmtiThreadInfo ti;
	jvmtiError r;
	UNREFERENCED_PARAMETER(jni_env);

	if (thread != NULL)
	{
		r = (*jvmti_env)->GetThreadInfo(jvmti_env, thread, &ti);
		if (r == JVMTI_ERROR_NONE)
			Extrae_set_thread_name (THREADID, ti.name);
	}
*/
}

JNIEXPORT jint JNICALL Agent_OnLoad(JavaVM *vm, char *options, void *reserved)
{
    jint                rc;
    jvmtiError          r;
    jvmtiCapabilities   capabilities;
    jvmtiEventCallbacks callbacks;

	UNREFERENCED_PARAMETER(options);
	UNREFERENCED_PARAMETER(reserved);

    /* Get JVMTI environment */
    rc = (*vm)->GetEnv(vm, (void **)&jvmti, JVMTI_VERSION);
    if (rc != JNI_OK)
	{
        fprintf (stderr, PACKAGE_NAME": Error!: Unable to create jvmtiEnv, rc=%d\n", rc);
        return -1;
    }

    /* Get/Add JVMTI capabilities */
    memset(&capabilities, 0, sizeof(capabilities));
    capabilities.can_generate_garbage_collection_events = 1;
	capabilities.can_generate_exception_events = 1;
	capabilities.can_tag_objects = 1;
#if 0
	capabilities.can_generate_vm_object_alloc_events = 1;
	capabilities.can_generate_object_free_events = 1;
#endif
    r = (*jvmti)->AddCapabilities(jvmti, &capabilities);
	CHECK_JVMTI_ERROR(r, AddCapabilities);

    /* Set callbacks and enable event notifications */
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.GarbageCollectionStart  = &Extraej_cb_GarbageCollector_begin;
    callbacks.GarbageCollectionFinish = &Extraej_cb_GarbageCollector_end;
	callbacks.Exception               = &Extraej_cb_Exception;
	callbacks.ExceptionCatch          = &Extraej_cb_ExceptionCatch;
#if 0
    callbacks.VMObjectAlloc           = &Extraej_cb_ObjectAlloc;
    callbacks.ObjectFree              = &Extreaj_cb_ObjectFree;
#endif
	callbacks.ThreadStart             = &Extraej_cb_ThreadStart;

    r = (*jvmti)->SetEventCallbacks(jvmti, &callbacks, sizeof(callbacks));
	CHECK_JVMTI_ERROR(r, SetEventCallbacks);

	/* Garbage collector events */
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE,
	  JVMTI_EVENT_GARBAGE_COLLECTION_START, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE,
	  JVMTI_EVENT_GARBAGE_COLLECTION_FINISH, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);

#if 0
	/* VM alloc/free events */
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE, 
	  JVMTI_EVENT_VM_OBJECT_ALLOC, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE, 
	  JVMTI_EVENT_OBJECT_FREE, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);
#endif

	/* Exception events */
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE, 
	  JVMTI_EVENT_EXCEPTION, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);
    r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE, 
	  JVMTI_EVENT_EXCEPTION_CATCH, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);

	/* Thread start */
	r = (*jvmti)->SetEventNotificationMode(jvmti, JVMTI_ENABLE,
	  JVMTI_EVENT_THREAD_START, NULL);
    CHECK_JVMTI_ERROR(r, SetEventNotificationMode);

    /* Create the necessary raw monitor ?? really, necessary? */
    r = (*jvmti)->CreateRawMonitor(jvmti, "ExtraeJ_AgentLock", &ExtraeJ_AgentLock);
    CHECK_JVMTI_ERROR(r, CreateRawMonitor);
    return 0;
}

JNIEXPORT void JNICALL Agent_OnUnload(JavaVM *vm)
{
	UNREFERENCED_PARAMETER(vm);
}
