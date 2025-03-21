def EvenNums(num):
    if num == 0:
        return 0
    else:
        print(num)
        return EvenNums(num-2)

##EvenNums(10)

def OddNums(num):
    if num == 1:
        return 1
    else:
        print(num)
        return OddNums(num - 2)

OddNums(6) 