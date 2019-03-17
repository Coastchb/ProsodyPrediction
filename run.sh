# get segmented words and POS
python src/seg_pos_for_file.py -t data/all_data_20190316.txt data/train.data data/test.data

[ ! -d models ] && mkdir models
[ ! -d log ] && mkdir log

# train prosody model
bin/crf_learn CN_template data/train.data models/prosody.model

# test prosody model
bin/crf_test -m models/prosody.model data/test.data > log/test.log