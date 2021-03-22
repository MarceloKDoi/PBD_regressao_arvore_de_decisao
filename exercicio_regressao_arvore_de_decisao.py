import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


dataset = pd.read_csv('winequality-red.csv')
print(dataset)
ind = dataset.iloc[:, 0:-1].values
#ind = dataset.iloc[:, 10:-1].values
dep = dataset.iloc[:, -1].values

decisionTreeRegressor = DecisionTreeRegressor(random_state=0)
decisionTreeRegressor.fit(ind, dep)

# grafico 1

print(ind)
plt.scatter(ind[:, 0], decisionTreeRegressor.predict(ind), color="red")
plt.xlabel('Alcohol')
plt.ylabel('Quality')
plt.title('Alcohol vs Quality')
plt.show()

# grafico 2

fig = plt.figure()
subplot = fig.add_subplot(111, projection='3d')
subplot.scatter(ind[:, -2], ind[:, -5],
                decisionTreeRegressor.predict(ind), color="red")
plt.show()
