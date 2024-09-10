# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with slippery terrain.

## PROBLEM STATEMENT
We are assigned with the task of creating an RL agent to solve the "Bandit Slippery Walk" problem.
The environment consists of Seven states representing discrete positions the agent can occupy.
The agent must learn to navigate this environment while dealing with the challenge of slippery terrain.
Slippery terrain introduces stochasticity in the agent's actions, making it difficult to predict the outcomes of its actions accurately.

## POLICY EVALUATION FUNCTION
### FORMULA:
![image](https://github.com/user-attachments/assets/8701d70f-bd9e-445c-8726-3080a501664f)
### PROGRAM:

```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
```

```
import warnings ; warnings.filterwarnings('ignore')

import gym, gym_walk
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123)
```
```
def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```
```
def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
```
```
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
```
```
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)
```
```
env = gym.make('SlipperyWalkFive-v0')
P = env.env.P
init_state = env.reset()
goal_state = 6
LEFT, RIGHT = range(2)
```
```
P
```
```
init_state
```
```
state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))
```
```
# First Policy
pi_1 = lambda s: {
    0:LEFT, 1:LEFT, 2:LEFT, 3:LEFT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]
print_policy(pi_1, P, action_symbols=('<', '>'), n_cols=7)
```
```
# Find the probability of success and the mean return of the first policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_1, goal_state=goal_state)*100,
    mean_return(env, pi_1)))
```
```
# Create your own policy

pi_2 = lambda s: {
    0:RIGHT, 1:RIGHT, 2:RIGHT, 3:RIGHT, 4:LEFT, 5:LEFT, 6:LEFT
}[s]

print_policy(pi_2, P, action_symbols=('<', '>'), n_cols=7)
```
```
## Find the probability of success and the mean return of you your policy
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
      probability_success(env, pi_2, goal_state=goal_state)*100,
      mean_return(env, pi_2)))
```
```
# Calculate the success probability and mean return for both policies
success_prob_pi_1 = probability_success(env, pi_1, goal_state=goal_state)
mean_return_pi_1 = mean_return(env, pi_1)

success_prob_pi_2 = probability_success(env, pi_2, goal_state=goal_state)
mean_return_pi_2 = mean_return(env, pi_2)
```
```
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    # Write your code here to evaluate the given policy
    while True:
      V = np.zeros(len(P))
      for s in range(len(P)):
        for prob, next_state, reward, done in P[s][pi(s)]:
          V[s] += prob * (reward + gamma *  prev_V[next_state] * (not done))
      if np.max(np.abs(prev_V - V)) < theta:
        break
      prev_V = V.copy()
    return V
```
```
# Code to evaluate the first policy
V1 = policy_evaluation(pi_1, P)
print_state_value_function(V1, P, n_cols=7, prec=5)
```
```
# Code to evaluate the second policy
# Write your code here
V2 = policy_evaluation(pi_2, P)
print_state_value_function(V2, P, n_cols=7, prec=5)
```
```
# Comparing the two policies

# Compare the two policies based on the value function using the above equation and find the best policy

V1

print_state_value_function(V1, P, n_cols=7, prec=5)

V2

print_state_value_function(V2, P, n_cols=7, prec=5)

V1>=V2

if(np.sum(V1>=V2)==7):
  print("The first policy has the better policy")
elif(np.sum(V2>=V1)==7):
  print("The second policy has the better policy")
else:
  print("Both policies have their merits.")
```

## OUTPUT:
### Policy - 1:
![Screenshot 2024-09-10 153253](https://github.com/user-attachments/assets/87fa6f20-c6ab-4b0b-aa0a-8ac67a8d2544)

![Screenshot 2024-09-10 153309](https://github.com/user-attachments/assets/297a5ba2-d889-4b3f-abd3-17d7b7d01809)

![Screenshot 2024-09-10 153715](https://github.com/user-attachments/assets/81a8a4c4-4f40-42a2-a04b-6b6db201eab2)
### Policy - 2:
![Screenshot 2024-09-10 153328](https://github.com/user-attachments/assets/20600d3e-607c-4e46-bd0b-cbc276bb91bf)

![Screenshot 2024-09-10 153338](https://github.com/user-attachments/assets/df23415a-29f5-4f23-aa51-5af0db9252be)

![Screenshot 2024-09-10 153747](https://github.com/user-attachments/assets/3f178093-0045-441c-9f8b-df65d83577ea)
### Comparison:
![Screenshot 2024-09-10 153811](https://github.com/user-attachments/assets/e4d15422-c365-4adb-8e18-43aae06c21cc)

## RESULT:
Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and execcuted successfully.
