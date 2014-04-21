CC=g++
FLAGS=-c -Wall -I. -iquote /afs/cs/academic/class/15418-s13/public/include

all: LK

LK: LK.o
	$(CC) LKParallel.cpp -o LK

LK.o: LKParallel.cpp
	$(CC) $(FLAGS) LKParallel.cpp

clean:
	rm -rf *o LK
