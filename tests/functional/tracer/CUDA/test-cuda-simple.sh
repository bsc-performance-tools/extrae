#!/bin/bash

RET_VALUE=""

function isNumber() {
    local str=$1
    if [[ "$str" =~ ^[0-9]+$ ]]; then
        return true
    else
        return false
    fi
}

function getEventType() {
    RET_VALUE=""
    local string=$1
    local file=${2/prv/pcf}
    local output=$(awk -v string="$string" '{ if(match($0,string)){ print $2 } } ' $file)
    RET_VALUE=$output
}

function getEventValueForType() {
    RET_VALUE=""
    local evt_type=$1
    local value_string=$2
    local file=${3/prv/pcf}
    local output=$(awk -v value_string="$value_string" -v evt_type="$evt_type" 'BEGIN {interest=0} { if(match($0,evt_type)){ interest=1 }; if(interest==1 && match($0,value_string)){print $1; interest=0} } ' $file)
    RET_VALUE=$output
}

function checkInTrace() {
    RET_VALUE=""
    local evt_type=$1
    local evt_value=$2
    local prvFile=${3/pcf/prv}
    local pcfFile=${3/prv/pcf}
    getEventType "$evt_type$" "$pcfFile"
    num_evt_type=$RET_VALUE
    getEventValueForType "$evt_type$" "$evt_value$" "$prvFile"
    num_evt_value=$RET_VALUE
    echo -n "Grep \"$evt_type\" : \"$evt_value\" -> $num_evt_type:$num_evt_value $prvFile - "
    grep -m 1 "$num_evt_type:$num_evt_value" $prvFile > /dev/null
    local RES=$?
    if [ $RES == 0 ]; then
        echo "OK"
    else
        echo "KO"
        exit -1
    fi
}

function checkInPCF() {
    local string=$1
    local file=$2
    echo -n "Grep $string to $file - "
    grep $string $file > /dev/null
    local RES=$?
    if [ $RES == 0 ]; then
        echo "OK"
    else
        echo "KO"
        exit -1
    fi
}

# The compilation part of the check is inserted here due to the difficulties of change the compiler to nvcc

# Try first with -cudart shared (recent nvcc compilers requires this)
nvcc -g -cudart shared hello.cu -o hello
if [[ ! -x hello ]]; then
	nvcc -g hello.cu -o hello
fi

./trace.sh ./hello

# Check tests
getEventType "Flushing Traces" hello.prv
checkInPCF "CUDA" hello.pcf 
checkInPCF "cudaLaunch" hello.pcf 
checkInPCF "cudaConfigureCall" hello.pcf 
checkInPCF "cudaMemcpy" hello.pcf 
checkInPCF "cudaThreadSynchronize" hello.pcf 
#checkInPCF "cudaStreamSynchronize" hello.pcf 
checkInPCF "helloWorld" hello.pcf 

checkInTrace "CUDA kernel" helloWorld hello.prv
checkInTrace "CUDA library call" cudaThreadSynchronize hello.prv

exit 0
