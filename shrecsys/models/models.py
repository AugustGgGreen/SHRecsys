# -*- coding:utf-8 -*-
class Model(object):
    def __init__(self, model, epoch=1, lr=1, batch_size=1):
        self.model = model
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size

    def fit(self, input, output):
        model = self.model
        batch_size = self.batch_size
        epoch = self.epoch
        lr = self.lr
        model.build(lr)
        model.fit(input, output, epoch, batch_size)

    def load(self):
        model = self.model
        model.load()

    def predict(self, input):
        return self.model.predict(input)
