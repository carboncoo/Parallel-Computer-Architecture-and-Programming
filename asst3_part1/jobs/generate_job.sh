#/usr/bin/env bash

ncores=$1

if [ -z $1 ]; then
  echo "Usage: $0 <ncores>"
else
  curdir=`pwd`
  curdir=${curdir%/jobs}
  if [ $ncores -lt 16 ]; then
    sed "s:ROUNDCORES:16:g" example.job > tmp.job
  else
    sed "s:ROUNDCORES:$ncores:g" example.job > tmp.job
  fi
  sed "s:PROGDIR:$curdir:g" tmp.job > tmp1.job
  sed "s:NCORES:$ncores:g" tmp1.job > $USER\_$ncores.job
  rm -f tmp.job tmp1.job
fi
