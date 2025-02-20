import matplotlib.pyplot as plt

from network.sigmoid import sigmoid

START = -5
END = 5
STEP = 0.1

x_lst = []
y_lst = []

nb = START
while nb <= END:
    x_lst.append(nb)
    y_lst.append(sigmoid(nb))
    nb += STEP

plt.plot(x_lst, y_lst)
plt.show()
