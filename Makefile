TARGET = Demo

SOURCES = Demo.cpp NUFFT3D.cpp
HEADERS =
OBJECTS = $(SOURCES:.cpp=.o)

CXX = icpc
FLAGS = -std=c++11 -O3 -xHost -Wall -fopenmp -fp-model precise -DTIMING
LIB = -fopenmp -L/home/pac70/fftwd/usr/local/lib -lfftw3 -lfftw3_threads
INC = -I/home/pac70/fftwd/usr/local/include

################################################################################
################################################################################
################################################################################

%.o : %.cpp
	$(CXX) -c $(SOURCES) $(INC) $(FLAGS)

$(TARGET) : $(OBJECTS)
	$(CXX) -o $(TARGET) $(OBJECTS) $(LIB)

.PHONY: clean
clean:	
	rm -rf $(OBJECTS) $(TARGET)
