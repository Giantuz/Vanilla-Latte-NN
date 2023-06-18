# -*- Makefile -*-

all: 
	clang main.c MLP_init.c print_debug.c -o main -lm


clear:
	rm main