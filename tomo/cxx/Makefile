export CPLUS_INCLUDE_PATH = /usr/include/python3.5m/
export DEBUG = -O3 -fopenmp -fdiagnostics-color=always
export CXXFLAGS =  -std=c++11 ${DEBUG} 


SRC = $(wildcard *.cc)
OBJ = $(patsubst %.cc, %.o,${SRC}) 
EXEC =  fw_ray.x fw_mod3d.x test.x 
all: getopt_sub ${OBJ} ${EXEC}



#install:
#	cp ${EXEC} ${BINDIR}

getopt_sub:
	make -C getopt

%.d: %.cc
	@set -e; rm -f $@; \
	${CXX} -MM -std=c++11  $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

sinclude $(SRC:.cc=.d)

${EXEC}:%.x:%.o
	${CXX} ${DEBUG} ${LINK} -o $@ $<  global_tomo.o  -lm 

clean:
	make -C getopt clean
	-rm ${OBJ} ${EXEC} *.d -f

cleanobj:
	-rm ${OBJ} *.d -f
