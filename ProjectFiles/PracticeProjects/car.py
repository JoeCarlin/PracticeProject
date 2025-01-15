class Car:
    def __init__(self, make, model, year, color):
        self.make = make
        self.model = model
        self.year = year
        self.color = color
    
    def drive(self):
        print(f"You drive the {self.color} {self.model}.")

    def stop(self):
        print(f"You stop the {self.color} {self.model}.")

    def describe(self):
        print(f"{self.year} {self.make} {self.model} in {self.color}.")


car1 = Car("Toyota", "Corolla", 2015, "Red")
car2 = Car("Honda", "Civic", 2016, "Blue")
car3 = Car("Ford", "Mustang", 2014, "Black")

car3.describe()