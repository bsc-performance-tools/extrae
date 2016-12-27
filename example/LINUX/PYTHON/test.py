#!/usr/bin/python

### Load the Extrae python module in your own python application
import pyextrae.sequential as pyextrae

def a():
    print 'Called function a()!'
    return 1

def b():
    print 'Called function b()!'
    return 1

def c():
    print 'Called function c()!'
    return 1

### To automatically instrument your user functions, enable 
### the section user-functions in the extrae.xml configuration: 
###
###  <user-functions enabled="yes" list="function-list" exclude-automatic-functions="no">
###
### And list the functions you want to trace in the file "function-list".

for i in range(0, 10):
  a()
  b()
  c()

### You can emit punctual events anywhere in your code by calling:
pyextrae.eventandcounters(1000, 1)

