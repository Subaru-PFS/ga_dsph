# 
#Makefile
#

MF= Makefile

EXE= MCMC.exe

SRC= main.cu Gauss.cu Integrand.cu utils.cu


CC=	nvcc
CFLAGS= -O3 -std=c++11
#CFLAGS= -O3 -g -arch=sm_75 -std=c++11
#same speed with or without sm_75, but can also be run on nekoya without it


.SUFFIXES:
.SUFFIXES: .cu  .o

OBJ=	$(SRC:.cu=.o)


.cu.o:
	$(CC) $(CFLAGS) -dc -c $<

default: $(EXE)

$(EXE):	$(OBJ) 
	$(CC) $(CFLAGS) -o $@ $(OBJ) 

$(OBJ):	$(MF) *.h

clean:
	rm -f *.o *.exe
