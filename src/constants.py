import os
import pygame
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai/src')))

# Directory for user study logs
USER_STUDY_LOG = os.path.join(os.getcwd(), 'user_study/log')

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 1000000

# Directions
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action, PlayerState, ObjectState

n, s = Direction.NORTH, Direction.SOUTH
e, w = Direction.EAST, Direction.WEST
stay, interact = Action.STAY, Action.INTERACT
P, Obj = PlayerState, ObjectState
DISPLAY = False
MAX_STEPS = 20000
USER_STUDY_LOG = os.path.join(os.getcwd(), 'user_study/log')
PROMPT_LIMIT =10
TIMER, t = pygame.USEREVENT+1, 400

# light shade of the button 
COLOR_LIGHT = (170,170,170) 
# dark shade of the button 
COLOR_DARK = (100,100,100) 
COLOR_OVERCOOKED = (155, 101, 0)
WHITE = (255,255,255)

PAUSE_BUTTON_HEIGHT = 50
PAUSE_BUTTON_WIDTH = 100
INPUT_BOX_HEIGHT = 50
INPUT_BOX_WIDTH = 800
CONVERSATION_BOX_WIDTH = 500