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

	AM_CONDITIONAL(WANT_OPENCL, test "${enable_opencl}" = "yes")

])

