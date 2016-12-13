# Chinese-word-segmentation
For short-text words from Weibo (Weibo is analogous to Twitter)

## Evaluation
### PRF for fundamental system
![](https://github.com/MeteorYee/Chinese-word-segmentation/blob/master/images/f1.jpg)
### PRF for extra dict
![](https://github.com/MeteorYee/Chinese-word-segmentation/blob/master/images/f2.jpg)
### PRF for POS tagging
![](https://github.com/MeteorYee/Chinese-word-segmentation/blob/master/images/f3.jpg)

## Special Annoucement:

I have to attribute to an example on CSDN the implementation of this 
whole system.I took some reference to that system's architecture and
got a lot of help.
Here are the link:

http://download.csdn.net/detail/SummerRain2008/468453

## File Introduction
### PRF_Score.py:
> calculate the PRF value<br>

### Pcpt_Train.py:
> the fundamental segmentation

### Pcpt_exdictTrain.py:
> add extra dictionary<br>
> the EXTRA DICTIONARY is stored in raw/extra_dict

### Pcpt_posTrain.py:
> add POS tagging

### my_train_seg.txt:
> a little example of training data

### my_train_segpos.txt:
> a little example of training data for POS tagging

## Usage：
> get into the right path in which these files are stored
use terminal

### Training:

* python XXX.py train infile model_name<br>
eg. python Pcpt_Train.py train my_train_seg.txt test_model

### Segmentation:

* python XXX.py seg infile model_name result_file<br>
eg. python Pcpt_Train.py seg test.txt avgmodel test_result.txt<br>
    python Pcpt_exdictTrain.py seg test.txt exdict_model ex_result

### POS tagging：

_**ONLY**_ for Pcpt_posTrain.py<br>
eg. python Pcpt_posTrain.py pos_seg test.txt posmodel pos_result

_**NOTE**_:
avgmodel, exdict_model and posmodel are all the names of my trained
models, you may use yours.

## PRF_Score.py:

* python PRF_Score.py testFile goldFile<br>
eg. python PRF_Score.py yourResultFile raw/gold_result.txt (for seg)<br>
    python PRF_Score.py yourResultFile raw/gold_pos_result.txt (for seg and pos)

## Contact me:
* meteor_yee@163.com
* meteoryee0924@gmail.com
