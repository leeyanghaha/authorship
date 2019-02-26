import time

class Time:
    def __init__(self):
        self.name_dict  = dict()


    def initialize(self,name=None):
        self.time = time.time()
        if time:
            self.name_dict.update({name:self.time})


    def consumed(self, name=None):
        end = time.time()
        if name:
            print( name + ' time consumed %.2fs' % (end - self.name_dict[name]))
            del self.name_dict[name]
        else:
            print('%.2fs consumed from initialize' % (end - self.time))