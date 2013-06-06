# AX_OPENCL
# --------------------
AC_DEFUN([AX_OPENCL],
[
	AC_ARG_WITH(opencl,
		AC_HELP_STRING(
			[--with-opencl@<:@=DIR@:>@],
			[Enable support for tracing OpenCL]
		),
		[opencl_path="${withval}"],
		[opencl_path="none"]
	)

	enable_opencl="no"

	if test "${opencl_path}" != "none" ; then
		AC_MSG_CHECKING([for OpenCL])
		if test -d "${opencl_path}" ; then
			if test -r ${opencl_path}/include/CL/cl.h ; then
				if test -r ${opencl_path}/lib/libOpenCL.so -o -r ${opencl_path}/lib${BITS}/libOpenCL.so ; then
					enable_opencl="yes"
					AC_MSG_RESULT(${opencl_path})
				else
					AC_MSG_ERROR([Cannot find the necessary library files for OpenCL])
				fi
			else
				AC_MSG_ERROR([Cannot find the necessary header files in the OpenCL path])
			fi
		else
			AC_MSG_ERROR([The specified path for OpenCL does not exist])
		fi
		AX_FLAGS_SAVE()
		AX_FIND_INSTALLATION([OPENCL], [${opencl_path}], [opencl])
		AX_FLAGS_RESTORE()
		enable_opencl="yes"
	fi

	if test "${enable_opencl}" = "yes" ; then

   AC_MSG_CHECKING([for OpenCL supported version])

		AX_FLAGS_SAVE()
		CFLAGS="${CFLAGS} -I${OPENCL_INCLUDES}"
		AC_LANG_SAVE()
		AC_LANG([C])
		AC_TRY_COMPILE(
			[#include <CL/cl.h>],
			[
			 #if CL_VERSION_1_2
			  return 1;
			 #else
			  #error "OpenCL does not support 1.2"
			 #endif
			],
			[OpenCL_version="1.2"],
			[OpenCL_version="no"]
		)

		if test "${OpenCL_version}" = "no" ; then
			AC_TRY_COMPILE(
				[#include <CL/cl.h>],
				[
				 #if CL_VERSION_1_1
				  return 1;
				 #else
				  #error "OpenCL does not support 1.1"
				 #endif
				],
				[OpenCL_version="1.1"],
				[OpenCL_version="no"]
			)
    fi

		if test "${OpenCL_version}" = "no" ; then
			AC_TRY_COMPILE(
				[#include <CL/cl.h>],
				[
				 #if CL_VERSION_1_0
				  return 1;
				 #else
				  #error "OpenCL does not support 1.0"
				 #endif
				],
				[OpenCL_version="1.0"],
				[OpenCL_version="no"]
			)
    fi

		if test "${OpenCL_version}" = "no" ; then
			AC_MSG_ERROR([Unable to detect the version of the OpenCL system])
		fi
	
		AX_FLAGS_RESTORE()
		AC_LANG_RESTORE()
		AC_MSG_RESULT([${OpenCL_version}])
		AX_FLAGS_RESTORE()
	fi

	AM_CONDITIONAL(WANT_OPENCL, test "${enable_opencl}" = "yes")
])

# AX_OPENCL_SHOW_CONFIGURATION
# --------------------
AC_DEFUN([AX_OPENCL_SHOW_CONFIGURATION],
[
	if test "${enable_opencl}" = "yes" ; then
		echo OpenCL instrumentation: yes, through LD_PRELOAD
		echo OpenCL version: ${OpenCL_version}
		echo -e \\\tOpenCL home: ${opencl_path}
	else
		echo OpenCL instrumentation: no
  fi
])
