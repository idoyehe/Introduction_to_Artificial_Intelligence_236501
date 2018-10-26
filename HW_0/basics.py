from functools import reduce


def printHelloWorld():
    print('Hello World!')


def boolConvert(flag):
    return 1 if flag else 0


def xPowY(x, y):
    return x ** y


def listSum(num_list):
    return reduce(lambda n1, n2: n1 + n2, num_list)


def perfectNumber(num):
    return reduce(lambda n1, n2: n1 + n2, list(filter(lambda i: num % i == 0, range(1, num)))) == num


if __name__ == "__main__":
    printHelloWorld()
    print(boolConvert(False))
    print(xPowY(2, 10))

    # ex A.4
    x = 10
    y = 20
    print('x is: ', x)
    print('y is: ', y)
    print('\nSwitching...\n')
    (x, y) = (y, x)
    print('x is: ', x)
    print('y is: ', y)
    print('\n')

    # ex A.5
    myList = []
    myList.append("Ido")
    myList.append("Yehezkel")
    print('The List: ', myList)
    myList.reverse()
    print('The reversed List: ', myList)
    print('\n')

    # ex A.6
    print(*range(2, 24))
    print('\n')

    # ex A.7
    num_list = list(range(1, 11))
    print(num_list[3: 10: 2])
    print('\n')

    # ex A.8
    num_list = list(range(1, 11))
    print(listSum(num_list))
    print('\n')

    # ex A.9
    myFile = open('my_file.txt', 'w+')
    myFile.write("I know how to write")
    print('\n')

    # ex A.10
    print('is 6 perfect ?', perfectNumber(6))
    print('is  20 perfect ?', perfectNumber(20))
    print('is  28 perfect ?', perfectNumber(28))
    print('\n')
