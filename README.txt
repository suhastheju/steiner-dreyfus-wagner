Overview
--------
This software repository contains a parallel implementation of the 
Dreyfus-Wagner algorithm for solving the Steiner problem in graphs, and by
reduction, the group Steiner problemi in graphs. The software is written in 
C programming language and OpenMP API for parallelisation.

This software was developed as part of my master thesis work 
"Scalable Parameterised Algorithms for two Steiner Problems" at Aalto
University, Finland.

The source code is configured for a gcc build for Intel
microarchitectures. Other builds are possible but require manual
configuration of the 'Makefile'.

The source code is subject to MIT license, see 'LICENSE' for details.

Building
--------
Use GNU make to build the software.

Testing
-------
For testing use 'verify.py' script provided along with the software. The script 
is written in Python Version 3.0 release and it uses 'networkx' python package.
The test instances used for the purpose  of testing this software are available
in the 'testset' directory.

Use the command-line options to verify the optimal cost and optimal solution 
of the test instances.

Optimal cost: ./verify.py -bt reader-dw
Optimal solution: ./verify.py -bt reader-dw --list

Usage
-----
./reader-dw -in <input graph> <arguments>

arguments:
        -seed : seed value
        -dw   : Dreyfus-Wagner algorithm
        -list : list a solution
        -h    : help

./reader-dw -dw -in b01.stp -list
invoked as: ./reader-dw -dw -in b01.stp -list
no random seed given, defaulting to 123456789
random seed = 123456789
input: n = 50, m = 63, k = 9, cost = 82 [0.21 ms] {peak: 0.00GiB} {curr: 0.00GiB}
terminals: 48 49 22 35 27 12 37 34 24
root build: [zero: 1.89 ms] [pos: 0.02 ms] [adj: 0.02 ms] [term: 0.01 ms] done. [2.20 ms] {peak: 0.00GiB} {curr: 0.00GiB}
dreyfus: [zero: 0.35 ms] [init: 0.65 ms] [kernel: 7.32 ms] [total: 7.98 ms] [traceback: 0.00 ms] done. [8.45 ms] [cost: 82] {peak: 0.00GiB} {curr: 0.00GiB}
solution: ["24 22", "22 12", "22 41", "41 49", "41 37", "22 20", "20 48", "20 35", "20 27", "27 34"]
command done [8.64 ms]
grand total [14.55 ms] {peak: 0.00GiB}
host: cs-119
list solution: true
num threads: 4
compiler: gcc 5.4.0

Input graphs
------------
The input graphs should be in DIMACS STP format. Our implementation accepts the
STP file only in ASCII file format and the characters must be in lower-case.
