import time
import subprocess
import datetime


def unpack_sub_iters(master_list, iter_type=(list, set)):
    ret = []
    for element in master_list:
        if isinstance(element, iter_type):
            ret.extend(unpack_sub_iters(element, iter_type))
        else:
            ret.append(element)
    return ret


def remove_keys(dictionary, keys):
    for key in keys:
        del dictionary[key]
    return dictionary


def list2dict(list_of_dicts, key, keep_key=False):
    ret = {}
    for element in list_of_dicts:
        ret[element[key]] = element
        if not keep_key:
            del element[key]
    return ret


def read(path, cast=str):
    with open(path) as f:
        return [cast(line.strip()) for line in f.readlines() if line.strip()]


def read_pairs(path, separator='\t', cast=(str, str)):
    with open(path) as f:
        return [(cast[0](k), cast[1](v)) for line in f.readlines() for k, v in [line.rstrip().split(separator)]]


def add_arithmetic_methods(base_cls=None):
    def decorate(cls):
        def make_func(func_name):
            def func(self, *args, **kwargs):
                super_method = getattr(base_cls or super(cls, self), func_name)
                ret = cls(super_method(*args, **kwargs))
                ret.__dict__ = self.__dict__
                return ret

            func.__name__ = func_name
            func.__qualname__ = '{}.{}'.format(cls.__qualname__, func_name)
            func.__module__ = cls.__module__
            return func

        for func_name in ['add', 'sub', 'mul', 'matmul', 'truediv', 'floordiv',
                          'mod', 'divmod', 'pow', 'lshift', 'rshift', 'and',
                          'xor', 'or', 'radd', 'rsub', 'rmul', 'rmatmul',
                          'rtruediv', 'rfloordiv', 'rmod', 'rdivmod', 'rpow',
                          'rlshift', 'rrshift', 'rand', 'rxor', 'ror', 'iadd',
                          'isub', 'imul', 'imatmul', 'itruediv', 'ifloordiv',
                          'imod', 'ipow', 'ilshift', 'irshift', 'iand', 'ixor',
                          'ior', 'neg', 'pos', 'abs', 'invert']:
            func_name = '__{}__'.format(func_name)
            func = make_func(func_name)
            setattr(cls, func_name, func)
        return cls

    return decorate


@add_arithmetic_methods()
class TimeStamp(float):
    def __init__(self, time, description=None):
        float.__init__(time)
        if description is not None:
            self.description = description

    def __new__(self, time, description=None):
        return float.__new__(self, time)

    def __str__(self):
        attribute_dict = self.__dict__.copy()
        attribute_dict['time'] = float(self)
        return attribute_dict.__str__()

    def __repr__(self):
        return self.__str__()


class Timer():
    def __init__(self, start=False):
        self.times = []
        if start:
            self.start()

    def start(self):
        self.step()

    def step(self, description=None):
        self.times.append(TimeStamp(time.time(), description))

    def __str__(self):
        return [self.times[i] - self.times[i - 1] for i in range(1, len(self.times))].__str__()


def get_gpu_memory_usage():
    queries = ['total', 'used', 'free']
    query_gpu = ','.join(['.'.join(['memory', query]) for query in queries])
    result = subprocess.check_output(
        [
            'nvidia-smi', f'--query-gpu={query_gpu}',
            '--format=csv,nounits,noheader'
        ]).decode('utf-8')
    gpu_memory = [[int(y) for y in x.split(', ')] for x in result.strip().split('\n')]
    gpu_memory_usage = dict(zip(range(len(gpu_memory)), [dict(zip(queries, info)) for info in gpu_memory]))
    return gpu_memory_usage


def get_empty_cuda_devices():
    ret = []
    devices = get_gpu_memory_usage()
    for device in devices:
        if devices[device]['used'] < 13:
            ret.append(device)
    return ret


def current_time():
    return datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%s")