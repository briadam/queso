#!/bin/sh
ret=$(./test_uqEnvironment 2>&1)
if [ ret != 0 ]
then
  exit 0
fi

if [ ret == 0 ]
then
  exit 1
fi
