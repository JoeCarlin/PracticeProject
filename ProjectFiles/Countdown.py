import time

my_time = int(input("Enter the time in seconds: "))

for x in range(my_time, 0, -1):
    seconds = x % 60
    minutes = int(x / 60) % 60
    print(f"00:{minutes:02}:{seconds:02}")
    time.sleep(1)


print("Time is up!")