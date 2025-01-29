import math

# Given value
x = math.pi / 4

# True value of sin(pi/4)
true_value = math.sin(x)

# Taylor series approximation truncated to 6 terms
taylor_approx = x - (x**3) / math.factorial(3) + (x**5) / math.factorial(5) - (x**7) / math.factorial(7) + (x**9) / math.factorial(9) - (x**11) / math.factorial(11)

# Relative error
relative_error = (true_value - taylor_approx) / true_value * 100

true_value, taylor_approx, round(relative_error, 4)