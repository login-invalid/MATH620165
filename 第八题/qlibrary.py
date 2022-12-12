import numpy as np
import random

GAMMA = 0.9 # discount rate
ALPHA = 0.5 # learning rate
EPSILON = 0.05
WALL = -100
GOAL = 5
WHITE = -1

'''
Create the reward matrix based off of the provided grid
'''
def get_reward_matrix(grid, total):
    a,b = np.shape(grid)
    reward_matrix = np.array(np.zeros([total, 4]))  # Initialize reward matrix
    # [left, right, above ,bellow]
    # 0 represent you can't do that!
    for row in range(a):
        for column in range(b):

            # Check for 0s, -1s, and 1s to the left of each cell
            if column > 0:
                if grid[row, column - 1] == 0:
                    reward_matrix[row * b + column, 0] = WHITE
                elif grid[row, column - 1] == -1:
                    reward_matrix[row * b + column, 0] = WALL
                elif grid[row, column - 1] == 1:
                    reward_matrix[row * b + column, 0] = GOAL

            # Check for 0s, -1s, and 1s to the right of each cell
            if column < b - 1:
                if grid[row, column + 1] == 0:
                    reward_matrix[row * b + column, 1] = WHITE
                elif grid[row, column + 1] == -1:
                    reward_matrix[row * b + column, 1] = WALL
                elif grid[row, column + 1] == 1:
                    reward_matrix[row * b + column, 1] = GOAL

            # Check for 0s, -1s, and 1s above each cell
            if row > 0:
                if grid[row - 1, column] == 0:
                    reward_matrix[row * b + column, 2] = WHITE
                elif grid[row - 1, column] == -1:
                    reward_matrix[row * b + column, 2] = WALL
                elif grid[row - 1, column] == 1:
                    reward_matrix[row * b + column, 2] = GOAL

            # Check for 0s, -1s, and 1s below each cell
            if row < a - 1:
                if grid[row + 1, column] == 0:
                    reward_matrix[row * b + column, 3] = WHITE
                elif grid[row + 1, column] == -1:
                    reward_matrix[row * b + column, 3] = WALL
                elif grid[row + 1, column] == 1:
                    reward_matrix[row * b + column, 3] = GOAL

    return reward_matrix


'''
Generate the Q-Matrix based off of the provided grid.
'''
def get_q_matrix(grid, total):
    q_matrix = np.array(np.zeros([total, 4]))
    return q_matrix


'''
Retrieve a list of valid actions based off of the current state and reward matrix.
'''
def available_actions(state, reward_matrix):
    current_state_row = reward_matrix[state, :]  # Gather the correct row from the matrix based on state
    av_act = []  # Initialize the list of available actions
    for i in range(len(current_state_row)):
        if current_state_row[i] != 0:
            av_act.append(i)  # Check for a valid movement value. This is either 1 or -1 but not 0.
    return av_act


'''
Return a random action based off of a given state.
'''
def sample_next_action(action_range):
    next_action = random.choice(action_range)  # Randomly choose action
    return next_action


'''
Updates the q value within the q matrix via the q-learning algorithm after an action is taken.
'''
def update_q(state, action, q_matrix, reward_matrix, new_state):
    action_points = []  # Initialize list of points for each action
    for i in available_actions(new_state, reward_matrix):
        action_points.append(q_matrix[new_state, i])
    new_q = q_matrix[state, action] + (ALPHA * (reward_matrix[state, action] +  # Q-Learning
                  GAMMA * (max(action_points)) - q_matrix[state, action]))      # Algorithm
    q_matrix[state, action] = new_q  # Change q-value within the q-matrix


'''
Changes the state within the q_matrix depending on the aciton taken. This calls the update_q function to
update the corresponding q value as well.
'''
def update_state(state, action, b, q_matrix, reward_matrix):
    new_state = 0  # Initialize the new_state value
    if action == 0: # left
        new_state = state - 1
    elif action == 1: # right
        new_state = state + 1
    elif action == 2: # above
        new_state = state - b
    elif action == 3: # bellow
        new_state = state + b

    update_q(state, action, q_matrix, reward_matrix, new_state)  # Calls to update Q after selecting the new state
    return new_state


'''
As opposed to the sample_next_action function, makes an educated decision based on the best reward provided by a
list of possible actions. returns the decision so the update state function must be called manually.
'''
def educated_next_action(current_state, q_matrix, reward_matrix):
    state_actions = available_actions(current_state, reward_matrix)  # Creates list of actions for current state
    state_actions_2 = [ i for i in [0,1,2,3] if reward_matrix[current_state,i] != WALL] # not get to the wall
    if (np.random.uniform() < EPSILON):
        #print(state_actions_2)
        decision = random.choice(state_actions_2)
    else:
        tmp = q_matrix[current_state][state_actions].tolist()
        decision = tmp.index(max(tmp))
    return decision