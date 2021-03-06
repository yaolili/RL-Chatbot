# coding=utf-8

from __future__ import print_function
import cPickle as pickle
import config
import random


class Data_Reader:
    def __init__(self, training_data_path, cur_train_index=0, load_list=False):
        self.training_data = pickle.load(open(training_data_path, 'rb'))
        print("Read %s done!" % training_data_path)
        self.data_size = len(self.training_data)
        if load_list:
            self.shuffle_list = pickle.load(open(config.index_list_file, 'rb'))
        else:
            self.shuffle_list = self.shuffle_index()
        self.train_index = cur_train_index

    def get_batch_num(self, batch_size):
        return self.data_size // batch_size

    def shuffle_index(self):  # disabled
        shuffle_index_list = range(self.data_size)
        pickle.dump(shuffle_index_list, open(config.index_list_file, 'wb'), True)
        return shuffle_index_list

    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size:
            batch_index = self.shuffle_list[self.train_index:self.data_size]
            self.shuffle_list = self.shuffle_index()
            remain_size = batch_size - (self.data_size - self.train_index)
            batch_index += self.shuffle_list[:remain_size]
            self.train_index = remain_size
        else:
            batch_index = self.shuffle_list[self.train_index:self.train_index + batch_size]
            self.train_index += batch_size

        return batch_index

    def generate_training_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]  # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]  # batch_size of conv_b

        return batch_X, batch_Y

    def generate_training_batch_with_former(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]  # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]  # batch_size of conv_b
        former = [self.training_data[i][2] for i in batch_index]  # batch_size of former
        kw = [self.training_data[i][3] for i in batch_index]  # batch_size of kw

        return batch_X, batch_Y, former, kw

    def generate_testing_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]  # batch_size of conv_a

        return batch_X


if __name__ == "__main__":
    dr = Data_Reader('data2/train_origin.txt.kw.pkl')
    print("Load done!")
    batch_X, batch_Y = dr.generate_training_batch(50)
    for i in range(50):
        print(batch_X)
        for each in batch_X:
            print(each.decode("utf-8"))
        print("*****")
        exit()
