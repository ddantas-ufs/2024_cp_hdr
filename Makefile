CC = g++

CV_LIB = `pkg-config opencv --cflags --libs` -g
CPP_FLAGS = -fPIC -g
LD_FLAGS = -shared

BIN_DIR = bin
OLD_DIR = old
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

DEMO_IMG_LOWE = lena.pgm
#DEMO_IMG = lena.png
DEMO_IMG = 00.jpg
DEMO_HDR = 00.hdr

DEMO_IMG1_MATCH = 00.jpg
DEMO_IMG2_MATCH = 04.jpg

DEMO_HDR1_MATCH = 00.hdr
DEMO_HDR2_MATCH = 04.hdr

DEF_DOG = defaultdog
DEMO_DOG = demodog
DEMO_DOG_HDR = demodog_hdr
DEMO_HARRIS = demoharris
DEMO_HARRIS_HDR = demoharris_hdr
DEMO_SURF = demosurf
DEMO_SURF_HDR = demosurf_hdr
DEMO_HDR_IMG = $(IMG_DIR)/$(DEMO_HDR)
DEMO_LDR_IMG = $(IMG_DIR)/$(DEMO_IMG)
DEMO_LDR_IMG_LOWE = $(IMG_DIR)/$(DEMO_IMG_LOWE)

DEMO_LDR_IMG1_MATCH = $(IMG_DIR)/$(DEMO_IMG1_MATCH)
DEMO_LDR_IMG2_MATCH = $(IMG_DIR)/$(DEMO_IMG2_MATCH)

DEMO_HDR_IMG1_MATCH = $(IMG_DIR)/$(DEMO_HDR1_MATCH)
DEMO_HDR_IMG2_MATCH = $(IMG_DIR)/$(DEMO_HDR2_MATCH)

LOWE_SIFT = $(OLD_DIR)/sift_lowe_implementation/sift
SIFT_DESCRIPTOR = demosift
SIFT_OPENCV = demosift_opencv

MATCHING_OPENCV = matching_opencv

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

descriptor_sift:
	$(CC) -o $(BIN_DIR)/$(SIFT_DESCRIPTOR) $(TEST_DIR)/$(SIFT_DESCRIPTOR).cpp $(SRC_FILES) $(CV_LIB)

demosift_opencv:
	$(CC) -o $(BIN_DIR)/$(SIFT_OPENCV) $(TEST_DIR)/$(SIFT_OPENCV).cpp $(SRC_FILES) $(CV_LIB)

matching_opencv:
	$(CC) -o $(BIN_DIR)/$(MATCHING_OPENCV) $(TEST_DIR)/$(MATCHING_OPENCV).cpp $(SRC_FILES) $(CV_LIB)

run_demoharris: demoharris
	./$(BIN_DIR)/$(DEMO_HARRIS) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demoharris_hdr: demoharris_hdr
	./$(BIN_DIR)/$(DEMO_HARRIS_HDR) $(DEMO_HDR_IMG) $(OUT_DIR)/

run_demodog: demodog
	./$(BIN_DIR)/$(DEMO_DOG) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_demodog_hdr: demodog_hdr
	./$(BIN_DIR)/$(DEMO_DOG_HDR) $(DEMO_HDR_IMG) $(OUT_DIR)/

run_demosurf: demosurf
	./$(BIN_DIR)/$(DEMO_SURF) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_defaultdog: defaultdog
	./$(BIN_DIR)/$(DEF_DOG) $(DEMO_LDR_IMG) $(OUT_DIR)/

run_descriptor_sift: descriptor_sift
	./$(BIN_DIR)/$(SIFT_DESCRIPTOR) $(DEMO_LDR_IMG) $(OUT_DIR)/
	./$(BIN_DIR)/$(SIFT_DESCRIPTOR) $(DEMO_HDR_IMG) $(OUT_DIR)/

run_demosift_opencv: demosift_opencv
	./$(LOWE_SIFT) <./$(DEMO_LDR_IMG_LOWE) >./$(OUT_DIR)/$(DEMO_IMG_LOWE).key
	./$(BIN_DIR)/$(SIFT_OPENCV) $(DEMO_LDR_IMG) $(OUT_DIR)/ $(OUT_DIR)/$(DEMO_IMG_LOWE).key

run_matching_opencv: matching_opencv
	./$(BIN_DIR)/$(MATCHING_OPENCV) $(DEMO_LDR_IMG1_MATCH) $(DEMO_LDR_IMG2_MATCH) $(OUT_DIR)/
	./$(BIN_DIR)/$(MATCHING_OPENCV) $(DEMO_HDR_IMG1_MATCH) $(DEMO_HDR_IMG2_MATCH) $(OUT_DIR)/

pribyl_dtset:
	$(GET_URL) http://$(PRIBYL_DIR)/ -P $(IMG_DIR)/

rana_dtset:
	$(GET_URL) http://$(RANA_ZIP) -P $(IMG_DIR)/
	unzip $(IMG_DIR)/$(RANA_ZIP)
	rm $(IMG_DIR)/$(RANA_ZIP)
