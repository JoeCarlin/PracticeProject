# python weight converter

weight = float(input("Enter your weight: "))
unit = input("Kiolograms or Pounds? (L or K): ")

if unit == "K":
    weight = weight * 2.205
    unit = "Lbs"
elif unit == "L":
    weight = weight / 2.205
    unit = "Kgs"
else:
    print("Invalid unit")

print(f"Your weight is {round(weight, 1)} {unit}")