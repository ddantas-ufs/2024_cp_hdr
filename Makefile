CC = g++

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

DEP = src/input.cpp src/keypoint.cpp src/cp_hdr.cpp src/harris.cpp src/dog.cpp

harrisdemo:
	$(CC) -o bin/harrisdetector test/harrisdemo.cpp $(DEP) $(LIBS)

dogdemo:
	$(CC) -o bin/dogdetector test/dogdemo.cpp $(DEP) $(LIBS)