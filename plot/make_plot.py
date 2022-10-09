import pandas as pd
import matplotlib.pyplot as plt

rewards = pd.read_csv("/content/drive/Shareddrives/Reinforcement_Learning/rewards.csv") 
rewards = rewards.to_numpy()

plt.plot([i for i in range(len(rewards))], rewards)
plt.xlabel('Episode no.')
plt.ylabel('Reward of DQN')
plt.show()
print()