# Inheritance examople in python = allows a class to inherit attributes and methods from a parent class
# Inheritance is a powerful feature in object oriented programming

class Animal:
    def __init__(self, name):
        self.name = name
        self.is_alive = True
    
    def eat(self):
        print(f"{self.name} is eating")

    def sleep(self):
        print(f"{self.name} is sleeping")


# the dog class is inheriting from the Animal class
class Dog(Animal):
    def bark(self):
        print("Woof! Woof!")
    
    def dig(self):
        print("Digging...")

class Cat(Animal):
    def meow(self):
        print("Meow! Meow!")
    
    def scratch(self):
        print("Scratching...")

class Mouse(Animal):
    def squeak(self):
        print("Squeak! Squeak!")
    
    def run(self):
        print("Running...")

# creating an instance of the Dog class
dog = Dog("Buddy")

# creating an instance of the cat class
cat = Cat("Whiskers")

# creating an instance of the mouse class
mouse = Mouse("Jerry")

print(dog.name)
print(dog.is_alive)
dog.bark()