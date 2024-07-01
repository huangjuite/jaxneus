
DATA_FOLDER="public_data"
mkdir $DATA_FOLDER 
cd $DATA_FOLDER
wget https://www.dropbox.com/scl/fo/um3wj3ctiuoottbfmqmgb/ABZRltszDvWHJ824UL6DHw0?rlkey=3vjok0aivnoiaf8z5j6w05k92&e=1&dl=0
wait
mv ABZRltszDvWHJ824UL6DHw0?rlkey=3vjok0aivnoiaf8z5j6w05k92 public_data.zip
unzip public_data.zip
rm public_data.zip
unzip *.zip
rm *.zip
cd ..