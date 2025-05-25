from matplotlib import pyplot as plt

with open("popsize_9.csv") as data:
    x = []
    sheep = []
    wolves = []
    for i, line in enumerate(data):
        line = line.split(',')
        # if line[0] == 'a':
        #     break
        sheep.append(float(line[0]))
        wolves.append(float(line[1]))
        x.append(i)

x_hun = list(range(100))
plt.figure()
plt.plot(x, sheep, label='Sheep')
plt.plot(x, wolves, label='Wolves')
plt.legend()
plt.title('Pop size evolution')
plt.xlabel('Time (*10 ticks)')
plt.ylabel('Amount of individuals')
plt.show()
plt.close()
