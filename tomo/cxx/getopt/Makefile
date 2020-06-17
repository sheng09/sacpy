SRC  = getopt.cpp 
OBJ  = $(foreach src, ${SRC},  $(subst .cpp,.o,${src}))
LIB = libgetopt.a

DEBUG = -O3 -std=c++11
CXX = g++

all: ${OBJ} demo


${OBJ}:%.o:%.cpp getopt.cpp
	${CXX} -c -o $@ ${INC} ${DEBUG} $<

demo: demo.cc getopt.o
	${CXX} -o $@ ${INC} ${DEBUG} demo.cc getopt.o

${LIB}: ${OBJ}
	ar cr $@ ${OBJ}

clean:
	rm ${OBJ} ${LIB}
