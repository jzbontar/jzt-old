PREFIX=/home/jure/build/torch7
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lcublas -lluaT -lTHC -lTH

OBJ = jzt.o SpatialLogSoftMax.o

%.o : %.cu
	nvcc -arch sm_35 --compiler-options '-fPIC' -c $(CFLAGS) $<

libjzt.so: ${OBJ}
	nvcc -arch sm_35 -o libjzt.so --shared ${OBJ} $(LDFLAGS)

install: libjzt.so
	ln -s /home/jure/torch7/jzt/libjzt.so /home/jure/build/torch7/lib/lua/5.1/

clean:
	rm -f *.o *.so
