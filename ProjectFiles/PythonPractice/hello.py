# Variables and Data Types
integer_var = 10
float_var = 3.14
string_var = "Hello, Python!"
boolean_var = True

print("Variables and Data Types:")
print(f"Integer: {integer_var}, Float: {float_var}, String: {string_var}, Boolean: {boolean_var}\n")

# Control Structures
print("Control Structures:")
if integer_var > 5:
    print("Integer is greater than 5")
else:
    print("Integer is 5 or less")

print("For Loop:")
for i in range(3):
    print(f"Iteration {i}")

print("While Loop:")
count = 0
while count < 3:
    print(f"Count: {count}")
    count += 1
print()

# Functions
def greet(name):
    return f"Hello, {name}!"

print("Functions:")
print(greet("Alice"))
print()

# Lists and Dictionaries
my_list = [1, 2, 3]
my_dict = {"name": "Alice", "age": 25}

print("Lists and Dictionaries:")
print(f"List: {my_list}")
my_list.append(4)
print(f"Updated List: {my_list}")

print(f"Dictionary: {my_dict}")
my_dict["city"] = "Wonderland"
print(f"Updated Dictionary: {my_dict}\n")

# File Handling
print("File Handling:")
with open("example.txt", "w") as file:
    file.write("This is a test file.")

with open("example.txt", "r") as file:
    content = file.read()
    print(f"File Content: {content}\n")

# Exception Handling
print("Exception Handling:")
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Caught an exception: {e}\n")

# Classes and Objects
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"My name is {self.name} and I am {self.age} years old."

print("Classes and Objects:")
person = Person("Alice", 25)
print(person.introduce())