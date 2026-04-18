NVCC ?= nvcc
TARGET ?= pimontecarlo
SRC := src/main.cu
CUDAFLAGS ?= -O3 -std=c++17
LDFLAGS ?= -lcurand

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).exe
