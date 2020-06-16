export CPLUS_INCLUDE_PATH = /usr/include/python3.5m/
export DEBUG = -O3 -fdiagnostics-color=always
export H5FlAGS = -I/usr/include/hdf5/serial -D_LARGEFILE64_SOURCE -D_LARGEFILE_SOURCE -Wdate-time 
export CXXFLAGS =  -std=c++11 ${DEBUG} ${H5FlAGS}
export CXX = g++

SRC = $(wildcard *.cc)
OBJ = $(patsubst %.cc, %.o,${SRC}) 
EXEC_test = 
EXEC =      fw_ray.x fw_generate_mod3d.x fw_traveltime_sensitivity.x  fw_decipher_raypath.x \
${EXEC_test}



all: getopt_sub ${OBJ} ${EXEC} 



#install:
#	cp ${EXEC} ${BINDIR}

getopt_sub:
	make -C getopt

%.d: %.cc
	@set -e; rm -f $@; \
	${CXX} -MM -std=c++11 ${CXXFLAGS}  $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

sinclude $(SRC:.cc=.d)

${EXEC}:%.x:%.o
	h5c++ ${DEBUG} ${LINK} -o $@ $< earthmodel.o  -lm

clean:
	make -C getopt clean
	-rm ${OBJ} ${EXEC} *.d -f

cleanobj:
	-rm ${OBJ} *.d -f
