# multiple inheritance = a class can inherit from multiple classes
# multilevel inheritance = a class can inherit from a class that inherits from another class

# main parent class
class Animal:
    def __init__(self, name):
        self.name = name
        self.is_alive = True

    def eat(self):
        print(f"{self.name} is eating...")
    def sleep(self):
        print(f"{self.name} is sleeping...")

# Parent class to Preys
class Prey(Animal):
    def flee(self):
        print(f"{self.name} is fleeing...")

# Parent class to Predators
class Predator(Animal):
    def hunt(self):
        print(f"{self.name} is hunting...")

# Child class
class Hawk(Predator):
    pass

# Child class
class Rabbit(Prey):
    pass

# Child class using multiple inheritance
class Fish(Prey, Predator):
    pass

rabbit = Rabbit("Bugz Bunny")
hawk = Hawk("Hawkeye")
fish = Fish("Nemo")

rabbit.flee()
hawk.hunt()
fish.flee()
fish.hunt()