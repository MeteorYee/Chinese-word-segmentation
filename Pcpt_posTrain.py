#-*- coding: utf-8 -*-
#coding=utf-8

# Perceptron word segment for Chinese sentences
# Author:
# Meteor Yee

import os
import time
import random
import cPickle
import sys

class Perceptron:
    # initialize
    def __init__(self, segment, train, model_name):
        # Generate our symbol sets which provide us more features of the instance.
        # Not just extract the sentences with unigram template.
        # n-gram
        self.__char_type = {} # our symbol sets

        # punctuation, alphabet, date and Chinese numbers
        data_path = "PTData"
        for ind, name in enumerate(["punc", "alph", "date", "num"]):
            fn = data_path + "/" + name
            if os.path.isfile(fn):
                for line in file(fn, "rU"):
                    self.__char_type[line.strip().decode("gbk")] = ind
            else:
                print "can't open", fn
                exit()

        # Add our POS set
        self.__POS_Set = []
        inpt = open('raw/POS_Set', 'rU')
        for line in inpt:
            self.__POS_Set.append(line.strip())
        
        self.__train_insts = None           # all instances for training.
        self.__feats_weight = None          # ["b", "m", "e", "s"][all the features] --> weight.
        # It changes in each iteration, in other words, the feats_wight's scale gets lager with
        # the iteration.

        self.__words_num = None             # total words num in all the instances.
        self.__insts_num = None             # namley the sentences' num.
        self.__cur_ite_ID = None            # current iteration index.
        self.__cur_inst_ID = None           # current index_th instance.
        self.__real_inst_ID = None          # the accurate index in training instances after randimizing.
        self.__last_update = None           # ["b".."s"][feature] --> [last_update_ite_ID, last_update_inst_ID]
        self.__feats_weight_sum = None      # sum of ["b".."s"][feature] from begin to end.

        if segment and train or not segment and not train:
            print "there is only a True and False in segment and train"
            exit()
        elif train:
            self.Train = self.__Train
            # Change the private methon into public. Same as the below.
        else:
            self.__LoadModel(model_name)
            self.Segment = self.__Segment
            self.posSegment = self.__pos_Segment

    # Perceptron training
    def __Train(self, train_file_name, max_train_num, max_ite_num):
        if not self.__loadTrainFile(train_file_name, max_train_num):
            return False

        starttime = time.clock()

        self.__feats_weight = {}
        self.__last_update = {}
        self.__feats_weight_sum = {}

        for self.__cur_ite_ID in xrange(max_ite_num):
            if self.__Iterate():
                break

        self.__SaveModel()
        endtime = time.clock()
        print "Total iteration times is %d seconds" %(endtime - starttime)

        return True

    def __GenerateFeats(self, inst):
        inst_feat = []
        L = len(inst)

        for ind, [c, tag, t] in enumerate(inst):
            inst_feat.append([])

            if t == -1:
                continue

            # Cn
            for n in xrange(-2, 3):
                inst_feat[-1].append("C%d==%s" %(n, inst[ind + n][0]))
            # CnCn+1
            for n in xrange(-2, 2):
                inst_feat[-1].append("C%dC%d==%s%s" %(n, n + 1, inst[ind + n][0], inst[ind + n + 1][0]))
            # C-1C1 character
            inst_feat[-1].append("C-1C1==%s%s" %(inst[ind - 1][0], inst[ind + 1][0]))
            # Pu(C0) puc
            inst_feat[-1].append("Pu(%s)==%d" %(c, int(t == 0)))
            # T(C-2)T(C-1)T(C0)T(C1)T(C2) tag
            inst_feat[-1].append("T-2...2=%d%d%d%d%d" %(inst[ind - 2][2], inst[ind - 1][2], inst[ind][2], inst[ind + 1][2], inst[ind + 2][2]))
            
        return inst_feat

    # Load the training file
    def __loadTrainFile(self, train_file_name, max_train_num):
        if not os.path.isfile(train_file_name):
            print "can't open", train_file_name
            return False
        
        self.__train_insts = []
        self.__words_num = 0

        for ind, line in enumerate(file(train_file_name, "rU")):
            if max_train_num > 0 and ind >= max_train_num:
                break

            self.__train_insts.append(self.__PreProcess(line.strip()))

            self.__words_num += len(self.__train_insts[-1]) - 4

        self.__insts_num = len(self.__train_insts)

        print "number of total insts is", self.__insts_num
        # The number of sentences.
        print "number of total characters is", self.__words_num
        print
        
        return True

    # Transform the raw into the form we need.
    def __PreProcess(self, sent):
        inst = []

        for i in xrange(2):
            # Just a boundary
            inst.append(["<s>", "s", -1])
        for word in sent.decode("utf-8").split():
            rt = word.rpartition("/")
            # Divide the raw into 3 parts: [c, tag, t].

            t = self.__char_type.get(rt[0], 4)
            inst.append([rt[0], rt[2], t])
        for i in xrange(2):
            inst.append(["<s>", "s", -1])

        return inst

    # Just the segment result.
    def __Segment(self, src):
        inst = []
        for i in xrange(2):
            inst.append(["<s>", "s", -1])
        for c in src.decode("utf-8"):
            inst.append([c, "", self.__char_type.get(c, 4)])
        for i in xrange(2):
            inst.append(["<s>", "s", -1])
        
        feats = self.__GenerateFeats(inst)
        tags = self.__DPSegment(inst, feats)      

        rst = []
        for i in xrange(2, len(tags) -2):
            if tags[i][0] in ["S", "B"]:
                rst.append(inst[i][0])
            else:
                rst[-1] += inst[i][0]
                
        return " ".join(rst).encode("utf-8")

    # Result with POS.
    def __pos_Segment(self, src):
        inst = []
        for i in xrange(2):
            inst.append(["<s>", "s", -1])
        for c in src.decode("utf-8"):
            inst.append([c, "", self.__char_type.get(c, 4)])
        for i in xrange(2):
            inst.append(["<s>", "s", -1])
        
        feats = self.__GenerateFeats(inst)
        tags = self.__DPSegment(inst, feats)      

        pos_rst = []
        word = ''
        pos = ''
        for i in xrange(2, len(tags) -2):
            if tags[i][0] == 'B':
                word += inst[i][0]
                pos = '/' + tags[i][2:]
            elif tags[i][0] == 'S':
                word = inst[i][0]
                pos = '/' + tags[i][2:]
                pos_rst.append(word + pos + ' ')
                word = ''
                pos = ''
            elif tags[i][0] == 'M':
                word += inst[i][0]
            else:
                word += inst[i][0]
                pos_rst.append(word + pos + ' ')
                word = ''
                pos = ''
                
        return "".join(pos_rst).encode("utf-8")

    # Load the model.
    def __LoadModel(self, model_name):
        model = "PTData/" + model_name
        print "loading", model, "..."

        self.__feats_weight = {}

        if os.path.isfile(model):
            start = time.clock()
            self.__feats_weight = cPickle.load(file(model, "rb"))
            end = time.clock()
            print "It takes %d seconds" %(end - start)
        else:
            print "can't open", model

    # Save the model.
    def __SaveModel(self, model_name):
        # the last time to sum all the features.
        norm = float(self.__cur_ite_ID + 1) * self.__insts_num # Just a parameter

        for feat in self.__feats_weight_sum:
            last_ite_ID = self.__last_update[feat][0]
            last_inst_ID = self.__last_update[feat][1]

            # About 'c', please see the method __Update() below.
            c = (self.__cur_ite_ID - last_ite_ID) * self.__insts_num + self.__cur_inst_ID - last_inst_ID

            self.__feats_weight_sum[feat] += self.__feats_weight[feat] * c
            self.__feats_weight_sum[feat] = self.__feats_weight_sum[feat] / norm

        model = "PTData/" + model_name
        cPickle.dump(self.__feats_weight_sum, file(model, "wb"))
        self.__train_insts = None

    def __Iterate(self):
        start = time.clock()
        print "%d th iteration" %self.__cur_ite_ID

        train_list = random.sample(xrange(self.__insts_num), self.__insts_num)
        error_sents_num = 0
        error_words_num = 0
        
        for self.__cur_inst_ID, self.__real_inst_ID in enumerate(train_list):
            num = self.__TrainInstance()
            error_sents_num += 1 if num > 0 else 0
            error_words_num += num

        st = 1 - float(error_sents_num) / self.__insts_num
        wt = 1 - float(error_words_num) / self.__words_num

        end = time.clock()
        print "sents accuracy = %f%%, words accuracy = %f%%, it takes %d seconds" %(st * 100, wt * 100, end - start)
        print

        return error_sents_num == 0 and error_words_num == 0

    def __TrainInstance(self):
        cur_inst = self.__train_insts[self.__real_inst_ID]

        feats = self.__GenerateFeats(cur_inst)
        
        seg = self.__DPSegment(cur_inst, feats)

        return self.__Correct(seg, feats)

    def __DPSegment(self, inst, feats):        
        num = len(inst)

        # get all position's score.
        value = [{} for i in xrange(num)]
        for i in xrange(2, num - 2):
            for t in ["B", "M", "E", "S"]:
                for pos in self.__POS_Set:
                    tag = t + '-' + pos
                    value[i][tag] = self.__GetScore(i, tag, feats)

        # Viterbi DP
        tags = [None for i in xrange(num)]
        path = [{} for i in xrange(num)]

        # Initialize
        for i in xrange(2, num - 2):
            for pos in self.__POS_Set:
                path[i]["B-" + pos] = "B-" + pos
                path[i]["M-" + pos] = "M-" + pos
                path[i]["E-" + pos] = "E-" + pos
                path[i]["S-" + pos] = "S-" + pos

        for t in ["B", "M", "E", "S"]:
            # If t in ["B", "S"], the pos can be random.
            if t in ["B", "S"]:
                for cur_pos in self.__POS_Set:
                    maxv = float("-inf") # negative infinite
                    bestp = ''
                    for pre_pos in self.__POS_Set:
                        if value[2]['S-' + pre_pos] > maxv:
                            maxv = value[2]['S-' + pre_pos]
                            bestp = path[2]['S-' + pre_pos]
                    value[3][t + '-' + cur_pos] = maxv + value[3][t + '-' + cur_pos]
                    path[3][t + '-' + cur_pos] = bestp + ' ' + path[3][t + '-' + cur_pos]
            # If t in ["M", "E"], the pos must be same as the previous.
            else:
                for cur_pos in self.__POS_Set:
                    value[3][t + '-' + cur_pos] = value[2]['B-' + cur_pos] + value[3][t + '-' + cur_pos]
                    path[3][t + '-' + cur_pos] = path[2]['B-' + cur_pos] + ' ' + path[3][t + '-' + cur_pos]

        # find optimal path.
        for i in xrange(4, num - 2):
            for t in ["B", "M", "E", "S"]:
                # If t in ["B", "S"], the pos can be random.
                if t in ["B", "S"]:
                    for cur_pos in self.__POS_Set:
                        maxv = float("-inf") # negative infinite
                        bestp = ''
                        for pre_pos in self.__POS_Set:
                            # The previous is S
                            if value[i - 1]['S-' + pre_pos] > maxv:
                                maxv = value[i - 1]['S-' + pre_pos]
                                bestp = path[i - 1]['S-' + pre_pos]
                            # The previous is E
                            if value[i - 1]['E-' + pre_pos] > maxv:
                                maxv = value[i - 1]['E-' + pre_pos]
                                bestp = path[i - 1]['E-' + pre_pos]
                        value[i][t + '-' + cur_pos] = maxv + value[i][t + '-' + cur_pos]
                        path[i][t + '-' + cur_pos] = bestp + ' ' + path[i][t + '-' + cur_pos]
                # If t in ["M", "E"], the pos must be same as the previous.
                else:
                    for cur_pos in self.__POS_Set:
                        # The previous is B
                        if value[i - 1]['B-' + cur_pos] > value[i - 1]['M-' + cur_pos]:
                            value[i][t + '-' + cur_pos] = value[i - 1]['B-' + cur_pos] + value[i][t + '-' + cur_pos]
                            path[i][t + '-' + cur_pos] = path[i - 1]['B-' + cur_pos] + ' ' + path[i][t + '-' + cur_pos]
                        # The previous is M
                        else:
                            value[i][t + '-' + cur_pos] = value[i - 1]['M-' + cur_pos] + value[i][t + '-' + cur_pos]
                            path[i][t + '-' + cur_pos] = path[i - 1]['M-' + cur_pos] + ' ' + path[i][t + '-' + cur_pos]
        
        mypath = ''
        maxscore = float("-inf")
        for f_pos in self.__POS_Set:
            # end with E
            if value[num - 3]['E-' + f_pos] > maxscore:
                maxscore = value[num - 3]["E-" + f_pos]
                mypath = path[num - 3]["E-" + f_pos].split(' ')
            # end with S
            if value[num - 3]['S-' + f_pos] > maxscore:
                maxscore = value[num - 3]["S-" + f_pos]
                mypath = path[num - 3]["S-" + f_pos].split(' ')

        for i in xrange(len(mypath)):
            tags[i + 2] = mypath[i]
            
        return tags

    def __GetScore(self, pos, t, feats):
        pos_feats = feats[pos]
        score = 0.0
        for feat in pos_feats:
            score += self.__feats_weight.get(feat + "=>" + t, 0)
            
        return score  

    def __Correct(self, tags, feats):
        updates = {}
        cur_inst = self.__train_insts[self.__real_inst_ID]
        error_words_num = 0
        for i in xrange(2, len(cur_inst) - 2):
            if tags[i] == cur_inst[i][1]:
                continue
            error_words_num += 1
            pos_feats = feats[i]
            target = cur_inst[i][1]
            mine = tags[i]
            for feat in pos_feats:
                updates[feat + "=>" + target] = updates.get(feat + "=>" + target, 0.0) + 1
                updates[feat + "=>" + mine] = updates.get(feat + "=>" + mine, 0.0) - 1

        self.__Update(updates)
        
        return error_words_num;

    # update the features weight.
    def __Update(self, updates):
        for feat in updates:
            pair = self.__last_update.get(feat, [0, 0])
            last_ite_ID = pair[0]
            last_inst_ID = pair[1]
            
            c = (self.__cur_ite_ID - last_ite_ID) * self.__insts_num + self.__cur_inst_ID - last_inst_ID
            # About 'c':
            # 'c' here means a weight parameter, which is used to describe the degree of learning influence.
            # For example,
            # if perceptron corrects a wrong segment in its last few iterations, the revision of the feature
            # weight is so much bigger than before.

            self.__feats_weight_sum[feat] = self.__feats_weight_sum.get(feat, 0) + c * self.__feats_weight.get(feat, 0)
            
            self.__feats_weight[feat] = self.__feats_weight.get(feat, 0) + updates[feat]
            self.__last_update[feat] = [self.__cur_ite_ID, self.__cur_inst_ID]


if __name__ == "__main__":
    function = sys.argv[1]
    infile = sys.argv[2]
    model = sys.argv[3]

    if function == 'train':
        train = Perceptron(train = True, segment = False, model_name = model)
        train.Train(train_file_name = infile, model_name = model, max_train_num = 10010, max_ite_num = 20)
        del train

    elif function == 'seg':
        result_file = sys.argv[4]
        seg = Perceptron(train = False, segment = True, model_name = model)

        inpt = open(infile, 'rU')
        output = open(result_file, 'wb')

        for line in inpt:
            output.write(seg.Segment(line))

        print "Finished!"
        del seg

    elif function == 'pos_seg':
        result_file = sys.argv[4]
        seg = Perceptron(train = False, segment = True, model_name = model)

        inpt = open(infile, 'rU')
        output = open(result_file, 'wb')

        for line in inpt:
            output.write(seg.posSegment(line.strip()) + "\n")

        print "Finished!"
        del seg
        
    else:
        print 'unknown command'
        print function