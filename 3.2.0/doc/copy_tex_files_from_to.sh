#!/bin/bash

if test $# -ne 2 ; then
  echo "Invalid number of parameters. Expecting: <origin> <destination>"
fi

READLINK=`which readlink`

if test "${READLINK}" != "" ; then
	FULL_ORIGIN=`$READLINK -f $1`
	FULL_DESTINATION=`$READLINK -f $1`

	if test ! -d "${FULL_ORIGIN}" ; then
		echo "Error! Given origin ${FULL_ORIGIN} is not a directory"
		exit
	fi

	if test ! -d "${FULL_DESTINATION}" ; then
		echo "Error! Given destination ${FULL_DESTINATION} is not a directory"
		exit
	fi

	if test "${FULL_ORIGIN}" != "${FULL_DESTINATION}" ; then
		cp -r ${FULL_ORIGIN}/doc/*.tex ${FULL_ORIGIN}/doc/XML ${FULL_ORIGIN}/doc/images ${FULL_DESTINATION}
	else
		echo "Skipping copy..."
	fi
else
	echo "Cannot determine location for readlink tool"
fi
