# Python Calculator

operator = input("Enter the operator (+ - * /): ")
num1 = float(input("Enter the first number: "))
num2 = float(input("Enter the second number: "))

if operator == "+":
    print(round(num1 + num2))
elif operator == "-":
    print(round(num1 - num2))
elif operator == "*":
    print(round(num1 * num2))
elif operator == "/":
    print(round(num1 / num2))
else:
    print("Invalid operator")