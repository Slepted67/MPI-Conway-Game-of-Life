CC = mpicc
CFLAGS = -Wall -O2
TARGET = gol_mpi
OBJS = gol_mpi.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f $(TARGET) $(OBJS)
