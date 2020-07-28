MKDIR_P = mkdir -p
DATA_DIR = data/raw

.PHONY: clean data

directories:
	${MKDIR_P} ${DATA_DIR}

data/raw/ml-100k.zip: directories
	wget -O $@ http://files.grouplens.org/datasets/movielens/ml-100k.zip

data/ml-100k: data/raw/ml-100k.zip
	unzip -j  $< -d $@

## Make Dataset
data: data/ml-100k

