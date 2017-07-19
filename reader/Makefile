 #
 # This file is part of an experimental software implementation of the
 # Dreyfus-Wagner algorithm for solving the Steiner problem in graphs.
 #
 # This software was developed as part of my master thesis work 
 # "Scalable Parameterised Algorithms for two Steiner Problems" at Aalto
 # University, Finland.
 #
 # The source code is configured for a gcc build for Intel
 # microarchitectures. Other builds are possible but require manual
 # configuration of the 'Makefile'.
 #
 # The source code is subject to the following license.
 #
 # Copyright (c) 2017 Suhas Thejaswi
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 #

MAKE = make
CC = gcc 
CFLAGS = -Wall -march=native -std=c99 -fopenmp

SOURCE = reader-dw.c

EXE = reader-dw \
	READER_KER1 \
	READER_KER2

all: $(EXE)

reader-dw: $(SOURCE)
	$(CC) $(CFLAGS) -DDEFAULT -o $@ $< -lm

READER_KER1: reader-dw.c
	$(CC) $(CFLAGS) -g -DDEBUG -DBIN_HEAP -DDW_KERNEL -DTRACK_OPTIMAL -DBUILD_PARALLEL -o $@ $< -lm

READER_KER2: reader-dw.c
	$(CC) $(CFLAGS) -g -DDEBUG -DBIN_HEAP -DDW_KERNEL2 -DTRACK_OPTIMAL -o $@ $< -lm


.PHONY: $(EXE)

clean:  
	rm -f *.o *.a *~ 
	rm -f $(EXE)
