CC = g++

CV_LIB = `pkg-config opencv --cflags --libs`
CPP_FLAGS = -fPIC -g
LD_FLAGS = -shared

BIN_DIR = bin/
TEST_DIR = test/
SRC_DIR = src/
INC_DIR = include/
LIB_DIR = lib/
IMG_DIR = img/

NAME_LIB = cphdr
SRC_FILES = $(shell echo $(SRC_DIR)*.cpp)
TARGET_LIB = $(LIB_DIR)$(NAME_LIB).so
MAKE_LIB = $(CC) $(CPP_FLAGS) $(LD_FLAGS) -o $(TARGET_LIB) $(SRC_FILES)
COPY_INC = cp $(SRC_DIR)$(NAME_LIB).h $(INC_DIR)

GET_URL = wget -r -np -R "index.html*" -e robots=off
DTSET_URL = http://www.fit.vutbr.cz/~ipribyl/FPinHDR/dataset_JVCI/
DTSET_DIR = www.fit.vutbr.cz/~ipribyl/FPinHDR/dataset_JVCI/

DEMO_DOG = demodog
DEMO_HARRIS = demoharris
DEMO_SURF = demosurf
DEMO_IMG = $(IMG_DIR)$(DTSET_DIR)2D/distance/100/100.LDR.jpg
DEMO_ROI = $(IMG_DIR)$(DTSET_DIR)2D/distance/100/

demoharris:
	$(CC) -o $(BIN_DIR)$(DEMO_HARRIS) $(TEST_DIR)$(DEMO_HARRIS).cpp $(SRC_FILES) $(CV_LIB)

demodog:
	$(CC) -o $(BIN_DIR)$(DEMO_DOG) $(TEST_DIR)$(DEMO_DOG).cpp $(SRC_FILES) $(CV_LIB)

demosurf:
	$(CC) -o $(BIN_DIR)$(DEMO_SURF) $(TEST_DIR)$(DEMO_SURF).cpp $(SRC_FILES) $(CV_LIB)

run_demoharris:
	./$(BIN_DIR)$(DEMO_HARRIS) $(IMG_DEMO) $(DEMO_ROI)

run_demodog:
	./$(BIN_DIR)$(DEMO_DOG) $(IMG_DEMO) $(DEMO_ROI)

run_demosurf:
	./$(BIN_DIR)$(DEMO_SURF) $(IMG_DEMO) $(DEMO_ROI)

libcphdr:
	$(MAKE_LIB)
	$(COPY_INC)

img_dtset:
	$(GET_URL) $(DTSET_URL) -P $(IMG_DIR)