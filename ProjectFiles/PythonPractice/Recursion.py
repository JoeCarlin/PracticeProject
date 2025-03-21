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

#OddNums(7)

def Factorial(num):
    if num == 1:
        return 1
    else:
        return num * Factorial(num - 1)

#print(Factorial(6))

def SumList(lst):
    if len(lst) == 1:
        return lst[0]
    else :
        return lst[0] + SumList(lst[1:])

#print(SumList([1,2,3,4,5,6,7,8,9,10]))

def maxList(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return max(lst[0], maxList(lst[1:]))

#print(maxList([1,2,3,4,5,6,7,8,9,10]))

def minList(lst):
    if len(lst) == 1:
        return lst[0]
    else:
        return min(lst[0], minList(lst[1:]))

#print(minList([1,2,3,4,5,6,7,8,9,10]))

def reverseList(lst):
    if len(lst) == 1:
        return lst
    else:
        return reverseList(lst[1:]) + [lst[0]]

print(reverseList([1,2,3,4,5,6,7,8,9,10]))