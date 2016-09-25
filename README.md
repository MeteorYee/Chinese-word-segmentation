# Chinese-word-segmentation
For short-text words from Weibo (Weibo is analogous to Twitter)

Special Annoucement:

I have to attribute to an example the implementation of this whole system 
on CSDN, and the example is implemented by python and based on Perceptron.
I take some reference to that system's architecture. Here are the link:

http://download.csdn.net/detail/SummerRain2008/468453


PRF_Score.py:
calculate the PRF value


Pcpt_Train.py:
the fundamental segmentation

Pcpt_exdictTrain.py:
add extra dictionary

the EXTRA DICTIONARY is stored in raw/extra_dict

Pcpt_posTrain.py:
add POS tagging


my_train_seg.txt:
a little example of training data

my_train_segpos.txt:
a little example of training data for POS tagging

use：
get into the right path in which these files are stored
use terminal

Training:

python XXX.py train infile model_name
eg. python Pcpt_Train.py train my_train_seg.txt test_model

Segmentation:

python XXX.py seg infile model_name result_file
eg. python Pcpt_Train.py seg test.txt avgmodel test_result.txt
    python Pcpt_exdictTrain.py seg test.txt exdict_model ex_result

POS tagging：

ONLY for Pcpt_posTrain.py
eg. python Pcpt_posTrain.py pos_seg test.txt posmodel pos_result

note:
note: avgmodel, exdict_model and posmodel are all the names of my trained
models, you can use yours.

PRF_Score.py

python PRF_Score.py testFile goldFile
eg. python PRF_Score.py yourResultFile raw/gold_result.txt (for seg)
    python PRF_Score.py yourResultFile raw/gold_pos_result.txt (for seg and pos)
