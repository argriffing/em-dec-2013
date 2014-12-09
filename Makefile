CC = gcc
CYTHON_CFLAGS = -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing

all: fastem.so

fastem.c fastem.html: fastem.pyx
	cython -a fastem.pyx

fastem.so: fastem.c
	$(CC) $(CYTHON_CFLAGS) -I/usr/include/python2.7 -o fastem.so fastem.c

clean:
	rm fastem.so fastem.c fastem.html
	rm -f *.pyc

