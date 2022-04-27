CC = g++
PY = python3

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
OUT_DIR_SEG = out/results/segmentation
INSTALL_DIR = ${HOME}

NAME_LIB = cphdr
SRC_FILES = $(shell echo $(SRC_DIR)/*.cpp)
TARGET_LIB = $(LIB_DIR)/$(NAME_LIB).so
MAKE_LIB = $(CC) $(CPP_FLAGS) $(LD_FLAGS) -o $(TARGET_LIB) $(SRC_FILES)

GET_URL = wget -r -np -R "index.html*" -e robots=off
PRIBYL_DIR = www.fit.vutbr.cz/~ipribyl/FPinHDR/dataset_JVCI
RANA_ZIP = webpages.l2s.centralesupelec.fr/perso/giuseppe.valenzise/sw/HDR%20Scenes.zip

LUMINANCE_MAP_GENERATOR = python/src/generate_luminance_map.py

DEMO_IMG_LOWE = lena.pgm
#DEMO_IMG = lena.png
DEMO_IMG = 00.jpg
DEMO_HDR = 00.hdr

DEMO_IMG1_MATCH = 00.LDR.jpg
DEMO_IMG2_MATCH = 04.LDR.jpg

DEMO_HDR1_MATCH = 00.hdr
DEMO_HDR2_MATCH = 04.hdr

DEMO_ROI_IMG1_TXT = ROI.00.txt
DEMO_ROI_IMG2_TXT = ROI.04.txt
DEMO_ROI_IMG1_PNG = ROI.00.png
DEMO_ROI_IMG2_PNG = ROI.04.png

DEMO_HOMOGRAPHY_IMG1_IMG2 = H.00_04.txt
DEMO_HOMOGRAPHY_IMG2_IMG1 = H.04_00.txt

DEMO_CAMERA_MATRIX_FILE = K.txt

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

DEMO_ROI_IMG1 = $(IMG_DIR)/$(DEMO_ROI_IMG1_PNG)
DEMO_ROI_IMG2 = $(IMG_DIR)/$(DEMO_ROI_IMG2_PNG)

DEMO_LDR_IMG1_MATCH = $(IMG_DIR)/$(DEMO_IMG1_MATCH)
DEMO_LDR_IMG2_MATCH = $(IMG_DIR)/$(DEMO_IMG2_MATCH)

DEMO_HDR_IMG1_MATCH = $(IMG_DIR)/$(DEMO_HDR1_MATCH)
DEMO_HDR_IMG2_MATCH = $(IMG_DIR)/$(DEMO_HDR2_MATCH)

DEMO_HOMOGRAPHIC_MATRIX = $(IMG_DIR)/$(DEMO_HOMOGRAPHY_IMG1_IMG2)

DEMO_CAMERA_MATRIX = $(IMG_DIR)/$(DEMO_CAMERA_MATRIX_FILE)

LOWE_SIFT = $(OLD_DIR)/sift_lowe_implementation/sift
SIFT_DESCRIPTOR = demosift
SIFT_OPENCV = demosift_opencv

MATCHING_OPENCV = matching_opencv
MATCHING_CPHDR = matching_cphdr

COMPARE_ALGORITHMS = compare_algorithms
SINTETIC_TEST = sintetic_test

DEMO_HOMOGRAPHY = demo_homography

install:
	mkdir -p $(INSTALL_DIR)/$(NAME_LIB)
	cp -ar $(INC_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/
	cp -ar $(LIB_DIR)/ $(INSTALL_DIR)/$(NAME_LIB)/

libcphdr:
	$(MAKE_LIB)

demohomography:
	$(CC) -o $(BIN_DIR)/$(DEMO_HOMOGRAPHY) $(TEST_DIR)/$(DEMO_HOMOGRAPHY).cpp $(SRC_FILES) $(CV_LIB)

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

matching_cphdr:
	$(CC) -o $(BIN_DIR)/$(MATCHING_CPHDR) $(TEST_DIR)/$(MATCHING_CPHDR).cpp $(SRC_FILES) $(CV_LIB)

compare_algorithms:
	$(CC) -o $(BIN_DIR)/$(COMPARE_ALGORITHMS) $(TEST_DIR)/$(COMPARE_ALGORITHMS).cpp $(SRC_FILES) $(CV_LIB)

sintetic_test:
	$(CC) -o $(BIN_DIR)/$(SINTETIC_TEST) $(TEST_DIR)/$(SINTETIC_TEST).cpp $(SRC_FILES) $(CV_LIB)

run_generate_luminance:
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-0/scene-0.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-1/scene-1.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-2/scene-2.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-3/scene-3.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-4/scene-4.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-5/scene-5.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3
	$(PY) $(LUMINANCE_MAP_GENERATOR) /home/artur/builded_apps/dataset/lightRoom/scene-6/scene-6.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR_SEG)/ 3

run_demohomography: demohomography
	./$(BIN_DIR)/$(DEMO_HOMOGRAPHY) $(DEMO_LDR_IMG1_MATCH) $(DEMO_LDR_IMG2_MATCH) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/

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
	./$(BIN_DIR)/$(MATCHING_OPENCV) $(DEMO_LDR_IMG1_MATCH) $(DEMO_LDR_IMG2_MATCH) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/
#	./$(BIN_DIR)/$(MATCHING_OPENCV) $(DEMO_HDR_IMG1_MATCH) $(DEMO_HDR_IMG2_MATCH) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/

run_matching_cphdr: matching_cphdr
	./$(BIN_DIR)/$(MATCHING_CPHDR) $(DEMO_LDR_IMG1_MATCH) $(DEMO_LDR_IMG2_MATCH) $(OUT_DIR)/
	./$(BIN_DIR)/$(MATCHING_CPHDR) $(DEMO_HDR_IMG1_MATCH) $(DEMO_HDR_IMG2_MATCH) $(OUT_DIR)/

run_compare_algorithms: compare_algorithms
#	./$(BIN_DIR)/$(COMPARE_ALGORITHMS) $(DEMO_LDR_IMG1_MATCH) $(DEMO_ROI_IMG1) $(DEMO_LDR_IMG2_MATCH) $(DEMO_ROI_IMG2) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/
	./$(BIN_DIR)/$(COMPARE_ALGORITHMS) /home/artur/builded_apps/dataset/lightRoom/scene-0/scene-0.hdr /home/artur/builded_apps/dataset/lightRoom/scene-1/scene-1.hdr /home/artur/builded_apps/dataset/lightRoom/ROI_lr.png $(OUT_DIR)/
#	./$(BIN_DIR)/$(COMPARE_ALGORITHMS) $(DEMO_HDR_IMG1_MATCH) $(DEMO_ROI_IMG1) $(DEMO_HDR_IMG2_MATCH) $(DEMO_ROI_IMG2) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/
#	./$(BIN_DIR)/$(COMPARE_ALGORITHMS) $(DEMO_LDR_IMG1_MATCH) $(DEMO_LDR_IMG2_MATCH) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/
#	./$(BIN_DIR)/$(COMPARE_ALGORITHMS) $(DEMO_HDR_IMG1_MATCH) $(DEMO_HDR_IMG2_MATCH) $(DEMO_HOMOGRAPHIC_MATRIX) $(OUT_DIR)/

run_sintetic_test: sintetic_test
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step2.jpg img/teste_sintetico/escada_branco_step2.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step2.hdr img/teste_sintetico/escada_branco_step2.hdr $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step2.jpg  img/teste_sintetico/escada_preto_step2.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step2.hdr  img/teste_sintetico/escada_preto_step2.hdr $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step10.jpg img/teste_sintetico/escada_branco_step10.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step10.hdr img/teste_sintetico/escada_branco_step10.hdr $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step10.jpg  img/teste_sintetico/escada_preto_step10.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step10.hdr  img/teste_sintetico/escada_preto_step10.hdr $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step20.jpg img/teste_sintetico/escada_branco_step20.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_branco_step20.hdr img/teste_sintetico/escada_branco_step20.hdr $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step20.jpg  img/teste_sintetico/escada_preto_step20.jpg $(OUT_DIR)/
	./$(BIN_DIR)/$(SINTETIC_TEST) img/teste_sintetico/escada_preto_step20.hdr  img/teste_sintetico/escada_preto_step20.hdr $(OUT_DIR)/

pribyl_dtset:
	$(GET_URL) http://$(PRIBYL_DIR)/ -P $(IMG_DIR)/

rana_dtset:
	$(GET_URL) http://$(RANA_ZIP) -P $(IMG_DIR)/
	unzip $(IMG_DIR)/$(RANA_ZIP)
	rm $(IMG_DIR)/$(RANA_ZIP)
