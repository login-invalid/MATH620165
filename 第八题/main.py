import numpy as np
import random
import pygame as pg
import qlibrary
from PIL import Image

'''
read in the maze table!
'''
BLOCK_SIZE = 8
EPOSIDE = 100
WARMUP = 200
PATH = "/Users/vielyi/Desktop/课程/神经网络导论/神经网络考试/qlearning-pathfinding/maze.jpg"
FLAG = False

im = Image.open(PATH)
pixels = im.load()
width, height = im.size
offset = int(BLOCK_SIZE / 5)

maze = [
  [
    int(pixels[x + offset, y + offset][0] < 122)*(-1)
    for x in range(0, width, BLOCK_SIZE)
  ]
  for y in range(0, height, BLOCK_SIZE)
]

# Set width and height of UI window
windowwidth = 78*15
windowheight = 60*15

# Initialize PyGame
pg.init()

# Create PyGame window
win = pg.display.set_mode((windowwidth, windowheight))
pg.display.set_caption("RL for Maze")

# Initialize loop for customizing maze grid
setup = True

# Set dimensions of cells and their margins
width = 13
margin = 2

# Initialize the maze grid
grid = np.array(maze)
a,b = np.shape(grid)

'''
Update the game window.
'''
def update():
    global grid
    global win
    for row in range(len(grid)):
        for column in range(len(grid[row])):

            # Draw empty cells
            if grid[row, column] == 0:
                pg.draw.rect(win, pg.Color("White"), ((width + margin) * column, (width + margin) * row, width, width))
            # Draw cells occupied by the end goal
            if grid[row, column] == 1:
                pg.draw.rect(win, pg.Color("Green"), ((width + margin) * column, (width + margin) * row, width, width))
            # Draw cells occupied by walls
            if grid[row, column] == -1:
                pg.draw.rect(win, pg.Color("BLACK"), ((width + margin) * column, (width + margin) * row, width, width))
            # Draw cells occupied by the path finder
            if grid[row, column] == 2:
                pg.draw.rect(win, pg.Color("Blue"), ((width + margin) * column, (width + margin) * row, width, width))
            # Draw cells occupied by the starter
            if grid[row, column] == 4:
                pg.draw.rect(win, pg.Color("RED"), ((width + margin) * column, (width + margin) * row, width, width))
    pg.display.update()

# Initialize game window by calling update function
update()

'''
Loop for customizing game grid
'''
while setup:

    for event in pg.event.get():
        if event.type == pg.QUIT:
            setup = False
        # Take user input for drawing game environment
        elif event.type == pg.MOUSEBUTTONDOWN:
            pos = pg.mouse.get_pos()
            column = pos[0] // (width + margin)
            row = pos[1] // (width + margin)
            if event.button == 1: # 按下左键设置起点
                grid[row, column] = 4
            if event.button == 3: # 按下右键设置终点
                grid[row, column] = 1
            if event.button == 4: 
                grid[row, column] = 0 # 鼠标上滑设置路径
            if event.button == 5: 
                grid[row, column] = -1 # 鼠标下滑设置墙
            update()

# Initialize the total variable using qlibrary
total = a * b
FINAL_STEPS = np.zeros(total).tolist() # very large

# Initialize the q_matrix using qlibrary
q_matrix = qlibrary.get_q_matrix(grid, total)

# Initialize the reward matrix using qlibrary
reward_matrix = qlibrary.get_reward_matrix(grid, total)

'''
Initial 'exploration' training, 200 iterations
'''
for i in range(WARMUP):
    state = random.choice(range(0, total, 1)) # Picks random state
    action_range = qlibrary.available_actions(state, reward_matrix) # Evaluate all possible next actions
    action = qlibrary.sample_next_action(action_range) # Execute random action
    qlibrary.update_state(state, action, a, q_matrix, reward_matrix) # Update the state and q-value of Q(s, a)

# Initialize list of obstacles for 'game over' check

# Add obstacles to obstacle list
obstacles = [b * row + column for row in range(a) for column in range(len(grid[row])) 
            if grid[row, column] == -1]

# Locate user-placed start point
start = int(len(grid[row]) * np.where(grid == 4)[0] + np.where(grid == 4)[1])
# Locate user-placed end goal
goal = int(len(grid[row]) * np.where(grid == 1)[0] + np.where(grid == 1)[1])

print(f"Goal index is {goal} and start index is {start}.")
'''
'Exploitation' training, 1000 iterations
'''
for index in range(EPOSIDE):
    # Start
    current_state = start
    # Initialize list of steps taken
    steps = [current_state]
    # Loop that resets Solomon after he meets his end goal
    while current_state != goal:
        if current_state in obstacles:
            #print("I hit an obstacle!")
            break

        # Make an educated decision based on maximum immediate reward
        decision = qlibrary.educated_next_action(current_state, q_matrix, reward_matrix)

        # Updates the state based on the educated decision
        current_state = qlibrary.update_state(current_state, decision, b, q_matrix, reward_matrix)

        steps.append(current_state)

    # Color the cells to illustrate path that Solomon took to his end point
    for i in steps:
        grid[int(i/b), i % b] = 2
        update()

    # If any other significant cells were covered up, return them to their original color
    for i in steps:
        if i != goal and i not in obstacles:
            grid[int(i/b), i % b] = 0
        if i in obstacles:
            grid[int(i/b), i % b] = -1
        if i == goal:
            grid[int(i/b), i % b] = 1
    update()
    
    if current_state == goal:
        if not FLAG:
            print("find the path!")
            FLAG = True
        else:
            if len(FINAL_STEPS) > len(steps):
                FINAL_STEPS = steps

# Initialize the final loop which displays the frame showing the final path
wait = True

# Display the steps taken via blue cells
for i in steps:
    grid[int(i / b), i % b] = 2
    if i == goal:
        grid[int(i / b), i % b] = 1
update()

# Print the coordinates of the final path taken
if FLAG:
    print(FINAL_STEPS)
else:
    print("Failed to find the path!")

# Frozen frame loop
while wait:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            wait = False