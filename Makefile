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
DEMO_DOG_HDR = demodog_hdr
DEMO_HARRIS = demoharris
DEMO_HARRIS_HDR = demoharris_hdr
DEMO_SURF = demosurf
DEMO_SURF_HDR = demosurf_hdr
DEMO_HDR_IMG = $(IMG_DIR)/lena.png
DEMO_LDR_IMG = $(IMG_DIR)/lena.png

install:
  mkdir -p $(INSTALL_DIR)/$(NAME_LIB)
  cp -ar $(INC_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/
  cp -ar $(LIB_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/

libcphdr:
  $(MAKE_LIB)

demoharris:
  $(CC) -o $(BIN_DIR)/$(DEMO_HARRIS) $(TEST_DIR)/$(DEMO_HARRIS).cpp $(SRC_FILES) $(CV_LIB)

demoharris_hdr:
  $(CC) -o $(BIN_DIR)/$(DEMO_HARRIS_HDR) $(TEST_DIR)/$(DEMO_HARRIS_HDR).cpp $(SRC_FILES) $(CV_LIB)

demodog:
  $(CC) -o $(BIN_DIR)/$(DEMO_DOG) $(TEST_DIR)/$(DEMO_DOG).cpp $(SRC_FILES) $(CV_LIB)

demodog_hdr:
  $(CC) -o $(BIN_DIR)/$(DEMO_DOG_HDR) $(TEST_DIR)/$(DEMO_DOG_HDR).cpp $(SRC_FILES) $(CV_LIB)

demosurf:
  $(CC) -o $(BIN_DIR)/$(DEMO_SURF) $(TEST_DIR)/$(DEMO_SURF).cpp $(SRC_FILES) $(CV_LIB)

defaultdog:
  $(CC) -o $(BIN_DIR)/$(DEF_DOG) $(TEST_DIR)/$(DEF_DOG).cpp $(SRC_FILES) $(CV_LIB)

run_demoharris: demoharris
  ./$(BIN_DIR)/$(DEMO_HARRIS) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demoharris_hdr: demoharris_hdr
  ./$(BIN_DIR)/$(DEMO_HARRIS_HDR) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demodog: demodog
  ./$(BIN_DIR)/$(DEMO_DOG) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demodog_hdr: demodog_hdr
  ./$(BIN_DIR)/$(DEMO_DOG_HDR) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demosurf: demosurf
  ./$(BIN_DIR)/$(DEMO_SURF) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_defaultdog: defaultdog
  ./$(BIN_DIR)/$(DEMO_DOG) $(DEMO_LDR_IMG) $(OUT_DIR)/
	
pribyl_dtset:
  $(GET_URL) http://$(PRIBYL_DIR)/ -P $(IMG_DIR)/

rana_dtset:
  $(GET_URL) http://$(RANA_ZIP) -P $(IMG_DIR)/
  unzip $(IMG_DIR)/$(RANA_ZIP)
  rm $(IMG_DIR)/$(RANA_ZIP)
