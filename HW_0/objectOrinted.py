from random import random, choice
from functools import reduce


class MyClass(object):
    def __init__(self):
        print('this is my class')


class Point(object):
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def show(self):
        print("my location is %d, %d" % (self.x, self.y))


def randomB5(num):
    rndNum = random(0, 1000)
    return 0 if rndNum > num else rndNum


def randomB6(lst, num):
    return [choice(lst) for i in range(num)]


def stringToSum(myStr):
    assert type(myStr) is str
    numbers = list(myStr.split(sep='.', maxsplit=1))
    assert len(numbers) == 2
    numbers = [int(i) for i in numbers]
    return reduce(lambda n1, n2: n1 + n2, numbers)


if __name__ == "__main__":
    myClass = MyClass()
    myClass = Point(1, 2)
    myClass.show()
    myClass = Point()
    myClass.show()
    print(randomB6([1, 2, 3, 4, 5, 6, 7, 8], 3))
    print(stringToSum('-123.456'))
    print(stringToSum('1.2'))
