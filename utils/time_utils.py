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
            print('{} time consumed {}s'.format(name, str(end - self.name_dict[name])))
            del self.name_dict[name]
        else:
            print('{}s consumed from initialize'.format(end - self.time))