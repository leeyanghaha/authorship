import math
import multiprocessing as mp
class Multiprocess:
    def __init__(self, process_num=10):
        self.process_num = process_num

    def split_data(self, data_array):
        batch_size = math.ceil(len(data_array) / self.process_num)
        batch_data = [data_array[i*batch_size:(i+1)*batch_size] for i in range(self.process_num)]
        return batch_data

    def multi_process(self,func, split=True, arg_list=None, kwarg_list=None):
        batch_data = arg_list
        if split:
            batch_data = self.split_data(arg_list)
        pool = mp.Pool(self.process_num)
        res_holder = []
        for i in range(self.process_num):
            res_getter = pool.apply_async(func=func, args=batch_data[i] if arg_list else [],
                                   kwds=kwarg_list[i] if kwarg_list else {})
            res_holder.append(res_getter)
        pool.close()
        pool.join()
        results = [rh.get() for rh in res_holder]
        return results

