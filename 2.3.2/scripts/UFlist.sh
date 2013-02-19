#!/bin/bash

IS_32_BIT_BINARY ()
{
  eval "file $1 | grep 32-bit >& /dev/null"
  return $?
}

IS_64_BIT_BINARY ()
{
  eval "file $1 | grep 64-bit >& /dev/null"
  return $?
}

if [ $# -ne 1 ]; then
	echo "$0: Running method:"
	echo "        $0 <file>"
	exit 1
fi

FILE=$1

if [ ! ${TMPDIR} ]; then
	echo "$0: Undefined \${TMPDIR}. Using \${PWD} as a temporal directory."
	TMPDIR=${PWD}
fi

if [ ! -d ${TMPDIR} ]; then
	echo "$0: Can't locate \${TMPDIR} as a valid directory. Exiting!"
	exit 1
fi

TMPDIR=$PWD

TMPFILE1=`mktemp -q ${TMPDIR}/Taddress.XXXXXX`
if [ $? -ne 0 ]; then
	echo "$0: Can't create temporal file on \${TMPDIR}. Exiting!"
	exit 1
fi

TMPFILE2=`mktemp -q ${TMPDIR}/Daddress.XXXXXX`
if [ $? -ne 0 ]; then
	echo "$0: Can't create temporal file on \${TMPDIR}. Exiting!"
	exit 1
fi

OSNAME=`uname -s`
PROCTYPE=`uname -p`

if [ ${OSNAME} = "GNU/Linux" ]; then
	OSNAME="Linux"
fi

if [ ${PROCTYPE} = "ppc64" ]; then
	PROCTYPE="ppc"
fi

if [ ${OSNAME} = "Linux" -a ${PROCTYPE} = "ppc" ]; then

  if IS_32_BIT_BINARY ${FILE} ; then

    nm --defined-only ${FILE} | awk -F" " '$2 == "T" { print $1" # "$3 ; }'

  elif IS_64_BIT_BINARY ${FILE} ; then

    # Obtain the routines in the binary (skip the routines defined in .SOs)
    nm --defined-only ${FILE} | awk -F" " '$2 == "T"  { if  (index(".", $3) == 0) print substr($3,2) ; }' > ${TMPFILE1}

    # Obtain the routines in the binary (-- in PPC seems to be trampolines --)
    nm --defined-only ${FILE} | awk -F" " '$2 == "D"  { print $1" # "$3 ; }' > ${TMPFILE2}

    # Search for each routine inside the trampolines
    while read line
    do
      awk -F" " -v fname=$line ' { if ( fname == $3 ) print $0 ; } ' < ${TMPFILE2}
    done < ${TMPFILE1}

  else

    echo "$0: Error: ${FILE} is not a binary?"

  fi

else

  echo "Sorry machines ${OSNAME} / ${PROCTYPE} are not handled"

fi

rm -f ${TMPFILE1} ${TMPFILE2}
