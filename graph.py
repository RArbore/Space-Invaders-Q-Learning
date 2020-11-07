import matplotlib.pyplot as plt
import sys

f = open(sys.argv[1])

lines = f.read().split("\n")

points = []

for l in lines:
    if l == "":
        continue
    points.append(float(l.split(" ")[int(sys.argv[2])]))

plt.plot(points)
plt.show()