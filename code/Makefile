CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -pedantic -fPIC
LDFLAGS = -lm -lpthread -lSDL2

SRC = cg19.c qb16.c spark19.c spark_bridge.c
OBJ = $(SRC:.c=.o)

all: libspark_core.a libspark_core.so speed_test continual_test bridge_continual_test

libspark_core.a: $(OBJ)
	ar rcs $@ $(OBJ)

libspark_core.so: $(OBJ)
	$(CC) $(CFLAGS) -shared $(OBJ) -o $@ $(LDFLAGS)

speed_test: speed_test.c libspark_core.a
	$(CC) $(CFLAGS) speed_test.c -L. -lspark_core -o speed_test $(LDFLAGS)

continual_test: continual_test.c libspark_core.a
	$(CC) $(CFLAGS) continual_test.c -L. -lspark_core -o continual_test $(LDFLAGS)

bridge_continual_test: bridge_continual_test.c libspark_core.a
	$(CC) $(CFLAGS) bridge_continual_test.c libspark_core.a -o bridge_continual_test $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) libspark_core.a libspark_core.so bench_memory speed_test continual_test bridge_continual_test

.PHONY: all clean
