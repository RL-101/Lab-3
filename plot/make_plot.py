import pandas as pd
import matplotlib.pyplot as plt


rewards = pd.read_csv("./rewards.csv") 
rewards = rewards.to_numpy()
plt.plot([i+1 for i in range(0, 105)], rewards[::2])
plt.xlabel('Episode no.')
plt.ylabel('Time spent to reach the flag')
plt.show()