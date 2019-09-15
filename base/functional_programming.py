"""函数式编程

"""
from functools import reduce


def even_filter(nums):
    for num in nums:
        if num % 2 == 0:
            yield num


def multiply_by_three(nums):
    for num in nums:
        yield num * 3


def convert_to_string(nums):
    for num in nums:
        yield 'The Number: %s' % num


nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pipeline = convert_to_string(multiply_by_three(even_filter(nums)))
print("传统方式")
for num in pipeline:
    print(num)


def even_filter(nums):
    return filter(lambda x: x % 2 == 0, nums)


def multiply_by_three(nums):
    return map(lambda x: x * 3, nums)


def convert_to_string(nums):
    return map(lambda x: 'The Number: %s' % x, nums)


nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pipeline = convert_to_string(multiply_by_three(even_filter(nums)))
print("函数式")
for num in pipeline:
    print(num)


def pipeline_func(data, fns):
    return reduce(lambda a, x: x(a), fns, data)


print("pipeline")
pipeline_func(nums, [even_filter, multiply_by_three, convert_to_string])


class Pipe(object):
    def __init__(self, func):
        self.func = func

    def __ror__(self, other):
        def generator():
            for obj in other:
                if obj is not None:
                    yield self.func(obj)

        return generator()


@Pipe
def even_filter(num):
    return num if num % 2 == 0 else None


@Pipe
def multiply_by_three(num):
    return num * 3


@Pipe
def convert_to_string(num):
    return 'The Number: %s' % num


@Pipe
def echo(item):
    print(item)
    return item


def force(sqs):
    for item in sqs: pass


nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print("force函数以及decorator模式")
force(nums | even_filter | multiply_by_three | convert_to_string | echo)
