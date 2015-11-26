############### 
# Directories # 
############### 
# Binary Output Directory 
BIN 			=	./bin
 
# Documentation Files/Directory 
DOC 			=	./doc
 
# Source Directory 
SRC 			=	./src
 
######### 
# Other # 
######### 
 
# Source Files 
SOURCES			=	$(SRC)/main.c $(SRC)/iamge.c $(SRC)/gpu.cu
 
# Zip File Name 
ZIP 			=	project 
 
# Other Files to add to zip 
ZIP_FILES 		=	./Makefile ./README.md 
 

# Example input/output
INPUT 			= 	$(DOC)/lena.ppm
OUTPUT 			= 	$(DOC)/out.ppm


################ 
# Make Modules # 
################ 
 
all: _bin_directory_ build

_bin_directory_:
	@rm -r -v $(BIN)
	@mkdir -v $(BIN)

build:
	nvcc $(SOURCES) -o $(BIN)/prog

run:
	$(BIN)/prog $(INPUT) $(OUTPUT)

zip:
	@zip -r $(ZIP) $(SRC) $(DOC) $(ZIP_FILES)

clear:
	@touch $(ZIP).zip
	@rm -r $(ZIP).zip
	@touch $(BIN) $(OBJ)
	@rm -r $(BIN) $(OBJ) -v
	@touch .dummy~
	@find ./ -name *~ | xargs rm -v
