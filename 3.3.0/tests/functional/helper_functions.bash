
#
# Die with a message. die ("This was erroneous!")
#

die () { echo "$@" 1>&2; exit 1; }

#
# NumberEntriesInPRV (file.prv, type, value)
#
NumberEntriesInPRV ()
{
	if [[ $# -ne 3 ]]; then
		die "Error! Invalid number of parameters in NumberEntriesInPRV"
	fi

	local n
	if [[ -f ${1} ]]; then
		n=`grep :${2}:${3} ${1} | wc -l`
	else
		n=0
	fi

	return $n
}

#
# CheckEntryIn (file.pcf, string)
#
CheckEntryInPCF ()
{
	if [[ $# -ne 2 ]]; then
		die "Error! Invalid number of parameters in NumberEntriesInPCF"
	fi

	local n
	if [[ -f ${1} ]]; then
		n=`grep "${2}" ${1} | wc -l`
	else
		n=0
	fi

	if [[ ${n} -eq 0 ]]; then
		die "Error! Could not find '${2}'."
	fi
}


