CC = g++

CV_LIB = `pkg-config opencv --cflags --libs`
CPP_FLAGS = -fPIC -g
LD_FLAGS = -shared

BIN_DIR = bin
TEST_DIR = test
SRC_DIR = src
INC_DIR = include
INC_SUBDIR = include/detectors
LIB_DIR = lib
IMG_DIR = img
OUT_DIR = out
INSTALL_DIR = ${HOME}

NAME_LIB = cphdr
SRC_FILES = $(shell echo $(SRC_DIR)/*.cpp)
TARGET_LIB = $(LIB_DIR)/$(NAME_LIB).so
MAKE_LIB = $(CC) $(CPP_FLAGS) $(LD_FLAGS) -o $(TARGET_LIB) $(SRC_FILES)

GET_URL = wget -r -np -R "index.html*" -e robots=off
PRIBYL_DIR = www.fit.vutbr.cz/~ipribyl/FPinHDR/dataset_JVCI
RANA_ZIP = webpages.l2s.centralesupelec.fr/perso/giuseppe.valenzise/sw/HDR%20Scenes.zip

DEMO_DOG = demodog
DEMO_HARRIS = demoharris
DEMO_SURF = demosurf
DEMO_HDR_IMG = $(IMG_DIR)/$(PRIBYL_DIR)/2D/distance/100/100.LDR.jpg
DEMO_LDR_IMG = $(IMG_DIR)/$(PRIBYL_DIR)/2D/distance/100/100.LDR.jpg
DEMO_ROI = $(IMG_DIR)/$(PRIBYL_DIR)/2D/distance/100/

install:
	mkdir -p $(INSTALL_DIR)/$(NAME_LIB)
	cp -ar $(INC_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/
	cp -ar $(LIB_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/

libcphdr:
	$(MAKE_LIB)

demoharris:
	$(CC) -o $(BIN_DIR)/$(DEMO_HARRIS) $(TEST_DIR)/$(DEMO_HARRIS).cpp $(SRC_FILES) $(CV_LIB)

demodog:
	$(CC) -o $(BIN_DIR)/$(DEMO_DOG) $(TEST_DIR)/$(DEMO_DOG).cpp $(SRC_FILES) $(CV_LIB)

demosurf:
	$(CC) -o $(BIN_DIR)/$(DEMO_SURF) $(TEST_DIR)/$(DEMO_SURF).cpp $(SRC_FILES) $(CV_LIB)

run_demoharris:
	./$(BIN_DIR)/$(DEMO_HARRIS) $(DEMO_LDR_IMG) $(DEMO_ROI) $(OUT_DIR)

run_demodog:
	./$(BIN_DIR)/$(DEMO_DOG) $(DEMO_LDR_IMG) $(DEMO_ROI) $(OUT_DIR)

run_demosurf:
	./$(BIN_DIR)/$(DEMO_SURF) $(DEMO_LDR_IMG) $(DEMO_ROI) $(OUT_DIR)

pribyl_dtset:
	$(GET_URL) http://$(PRIBYL_DIR)/ -P $(IMG_DIR)/

rana_dtset:
	$(GET_URL) http://$(RANA_ZIP) -P $(IMG_DIR)/
	unzip $(IMG_DIR)/$(RANA_ZIP)
	rm $(IMG_DIR)/$(RANA_ZIP)