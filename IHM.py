# %%
import numpy as np
import matplotlib.pyplot as plt
import math

# %%
def my_fun(x):
    return ((x+5)**2+2)


# %%
xpoints = x = np.linspace(-20.0, 5.0, 1000)
ypoints = my_fun(xpoints)
plt.plot(xpoints,ypoints)

# %%
print(math.dist([-4],[4]))

# %%
steps = 0
def findMinima(a,b,d):
    global steps
    if abs(math.dist([a],[b]))<d:
        print("a»b Distance: ",math.dist([a],[b]))
        steps+=1
        return a,b
    steps+=1 
    x1 = b - (0.618*(math.dist([a],[b])))   
    x2 = a + (0.618*(math.dist([a],[b])))
    print("a»b Distance: ",math.dist([a],[b]))
    if(abs(my_fun(x1)) > abs(my_fun(x2))):
        a = x1
    elif(my_fun(x1) < my_fun(x2)):
        b = x2
    elif(my_fun(x1)==my_fun(x2)):
        a,b = x1,x2
    return findMinima(a,b,d)

# %%
d = 0.001
i = findMinima(-6,6,d)
# print(np.dtype(i))
print(f"Minima Between Range :{list(i)} With Tollerence {d}\n Calculated in {steps} steps.")



