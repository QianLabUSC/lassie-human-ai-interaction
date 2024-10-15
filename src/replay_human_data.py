"""
log agent 1 prompt and action
"""
import pickle
import argparse
import os
import json
import copy

# Create the argument parser
parser = argparse.ArgumentParser(description="Load a pickle file by name")
parser.add_argument("--trajectory_folder", type=str, default="test",help="The name of the trajectory folder")
from llm.llm_api import GPT
from llm.utils import read_from_file, write_to_file
from llm.llm_agent import LLMModel, ManagerReactiveModel, managerReasoning
from mdp.llm_overcooked_mdp import LLMOvercookedGridworld
from overcooked_ai_py.utils import read_layout_dict
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from planning.planners import LLMMediumLevelActionManager
from llm.llm_agent import Agent
import pygame
import pygame_gui
from moviepy.editor import ImageSequenceClip
import shutil
from constants import (COLOR_OVERCOOKED, CONVERSATION_BOX_WIDTH,COLOR_DARK,COLOR_LIGHT, TIMER, WHITE,COLOR_RED,COLOR_BLUE, t)
from overcooked_ai_py.utils import read_layout_dict, load_from_json
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action


# Parse the command line arguments
args = parser.parse_args()


PICKLE_LOG_DIR = "user_study"

folder_name = args.trajectory_folder
traj_file_path = os.path.join(PICKLE_LOG_DIR, f"{folder_name}.pkl")
# traj_file_path = os.path.join(PICKLE_LOG_DIR, folder_name,"trajectory.pkl")
# config_file_path = os.path.join(PICKLE_LOG_DIR, folder_name,"config.pkl") no need. 

if not os.path.exists(traj_file_path):
    raise FileNotFoundError(f"Error: File {traj_file_path} does not exist.")

traj_file_path = "user_study/test.pkl"
with open(traj_file_path, "rb") as file:
    env_list = pickle.load(file)


layoutname = env_list["config"]
traj = env_list["traj"]
print(layoutname)
"""
preprocess traj data to get intended position. since RL,BC,human data do not have intended position. only planner has intended position.
"""
def add_intended_pos(traj):
    agent_1_intended_pos = None
    agent_2_intended_pos = None
    
    for state in reversed( traj):
        # print(state)
        if state["agent1_action"] == "interact":#assume agent interact position is their targeted position. label all previous state
            agent_1_pos = state["player1"]["position"]
            agent_1_ori = state["player1"]["orientation"]
            agent_1_intended_pos = (agent_1_pos[0] + agent_1_ori[0], agent_1_pos[1] + agent_1_ori[1]) #intended pos is the one agent is facing
        
        elif state["agent2_action"] == "interact":
            agent_2_pos = state["player2"]["position"]
            agent_2_ori = state["player2"]["orientation"]
            agent_2_intended_pos = (agent_2_pos[0] + agent_2_ori[0], agent_2_pos[1] + agent_2_ori[1]) 
        state["agent1_intended_pos"] = agent_1_intended_pos
        state["agent2_intended_pos"] = agent_2_intended_pos

    # # Filter out any state where intended_pos is still an empty string
    # filtered_traj = [
    #     state for state in traj
    #     if state.get("agent1_intended_pos") != "" and state.get("agent2_intended_pos") != ""
    # ]
    return traj

"""helper function to get agent greedy path based on agent curr state and intended pos. """
def get_greedy_actions(agent, agent_intended_pos, state):
    agent_state=state.players[agent.agent_index].to_dict()

    #based on intended postion, get greedy path/action 
    motion_planner = agent.mlam.joint_motion_planner.motion_planner
    #get agent1 greedy action 
    invalid_pos = world_mdp.terrain_pos_dict[" "] #empty space is not valid
    greedy_decisions = agent.mlam._get_ml_actions_for_positions(
        [agent_intended_pos]
    )

    start_pos = (agent_state["position"], agent_state["orientation"])
    best_cost = float("inf")
    best_plan= []

    for greedy_decision in greedy_decisions:
        try:
            plan, _, cost = motion_planner.get_plan(start_pos, greedy_decision)
        except:
            print(greedy_decision, " is not reachable")
            cost = float("inf")
            plan = []
        if cost < best_cost:
            best_cost = cost
            best_plan = plan


    return best_plan

traj = add_intended_pos(traj)
agent1_action_history = []
agent2_action_history = []
agent1_state_history = []  
agent2_state_history = []  

for state in traj:
    # Extract the action history for each agent
    agent1_action_history.append(state.get("agent1_action"))
    agent2_action_history.append(state.get("agent2_action"))
    player1 = state.get("player1", {})
    player2 = state.get("player2", {})
    # Append the entire player object for both agents
    agent1_state_history.append(player1)
    agent2_state_history.append(player2)
print(agent1_action_history)
print(len(agent2_action_history))
print(agent1_state_history[0])
print(len(agent2_state_history))
print(traj[0])
print(len(traj))
"""
Visualize trajectory, use keyboard to fast/backward, label state with follow_greedy(0/1) and correct decision(1-6)
"""

# create overcooked environments
world_mdp = LLMOvercookedGridworld.from_layout_name(
    layout_name=layoutname
)  

FULL_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": world_mdp.terrain_pos_dict["X"],
    "counter_drop": world_mdp.terrain_pos_dict["X"],
    "counter_pickup": world_mdp.terrain_pos_dict["X"],
    "same_motion_goals": False,
}
mlam = LLMMediumLevelActionManager(world_mdp, FULL_PARAMS)
world_mdp.old_dynamics = True #Enable automatic cooking pot
base_env = OvercookedEnv.from_mdp(world_mdp)

class SimulationAgent(Agent):
    """a preset agent that has a designated set of actions to take."""
    def __init__(
        self,
        mlam,
        agent_name,
        action_hist,
        state_hist,
    ):
        # args
        self.mlam = mlam
        self.agent_name = agent_name
        self.action_hist = action_hist 
        self.state_hist = state_hist
        # self.curr_frame = 0
        self.agent_name = agent_name


    def action(self, state):
        # self.curr_frame +=1
        return self.action_hist[self.curr_frame], {}
    # def set_agent_frame(self,curr_frame):
        # self.curr_frame = curr_frame
class Simulation:
    """
    Simulate the trajectory, use right/left key to navigate. 
    save and modify agent prompts at run time. 
    """
    def __init__(self, env, mdp, agent1,agent2,traj):
        self.env = env
        self.mdp = mdp
        self.agent1 = agent1
        self.agent2 = agent2    
        self.traj = traj
        self.time_step = 0
       
        self.screen_width = self.env.mdp.width * 30  + 70
        self.screen_height = self.env.mdp.height * 30 + 230
        self.score = 0
        self.debug = False
        # self.pause_button_clicked = False
        self._running = True
        self.curr_frame =0 #frame number
        self.state_hist ={} # store state history for rewinding. map current frame number to state
        self.follow_greedy = 1
        self.agent1_prompts = {}
        self.agent2_prompts = {}
        self.action2index = {(-1, 0): 2, (1, 0): 1, (0, -1): 3, (0, 1): 4, "interact": 5, (0, 0): 6}
        self.index2action = {2:(-1, 0), 1: (1, 0),  3:(0, -1), 4:(0, 1), 5:"interact",  6:(0, 0)}
        self.action2string = {(-1, 0): "move left", (1, 0): "move right", (0, -1)
                               : "move up", (0, 1): "move down", "interact": "interact", (0, 0): "stay"}

        self.arrow = pygame.image.load('./data/graphics/arrow.png')
        self.arrow = pygame.transform.scale(self.arrow, (15, 30))
        self.stay = pygame.image.load('./data/graphics/stay.png')
        self.stay = pygame.transform.scale(self.stay, (10, 20))
        self.target = pygame.image.load('./data/graphics/target.png')
        self.target = pygame.transform.scale(self.target, (20, 20))
        self.interact = pygame.image.load('./data/graphics/interact.png')
        self.interact = pygame.transform.scale(self.interact, (20, 20))
    def on_init(self):
        pygame.init()
        # pygame.display.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height)
        )

        self.screen.fill(COLOR_OVERCOOKED)
        # Initialize agents
        for agent in [agent1,agent2]:
            agent.set_mdp(self.env.mdp)
        # self.agent1.name = "agent1"
        # self.agent2.name = "agent2"

        # 1 second (1000 milisecond) timer
        pygame.time.set_timer(TIMER, t)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ds = load_from_json(
            os.path.join(script_dir, "data", "config", "kitchen_config.json")
        )
        test_dict = copy.deepcopy(ds)

        self.state_visualizer = StateVisualizer(**test_dict["config"])

        # initialize the UI manager
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        pygame.font.init()


    def next_frame(self):
        print(".................................")
   
       
        prev_state = self.env.state
        self.state_hist[self.curr_frame] = prev_state

        self.curr_frame +=1 
        print(f"NEXT FRAME!({self.curr_frame})")
        joint_action = (self.agent1.action_hist[self.curr_frame],self.agent2.action_hist[self.curr_frame])
        self.env.mdp.get_state_transition(prev_state, joint_action)
        self.env.step(
            joint_action, joint_agent_action_info=[{"1"}, {"2"}]
        )



    def prev_frame(self):
        print(".................................")
        if self.curr_frame>=0:
            self.curr_frame -=1 
            self.env.state = self.state_hist[self.curr_frame]
        
        print(f"PREV FRAME!({self.curr_frame})")
        
  



    # render the game state
    def on_render(self):
        kitchen = self.state_visualizer.render_state(
            self.env.state,
            self.env.mdp.terrain_mtx,
            hud_data=self.state_visualizer.default_hud_data(
                self.env.state,
                # time_left = (int(self.max_time/2) - self.time_step),
                # time_left= round(max(int(self.max_time/2) - (self.paused_at - self.start_time),0)) if self._paused else round(max(self.max_time/2 - (time() - self.start_time), 0)),
                score=self.score,
            ),
        )

        grid_x_num = len(self.env.mdp.terrain_mtx[0])
        grid_y_num = len(self.env.mdp.terrain_mtx)
        self.screen.blit(kitchen, (0, 0))
        
        rect = pygame.Rect(0, 140, grid_x_num*30+30, grid_y_num*30+40)
        mdp_surface = self.screen.subsurface(rect)

        #get current greedy action and current action 
        next_action = self.agent1.action_hist[self.curr_frame+1]
        agent1_intend_pos = self.traj[self.curr_frame]["agent1_intended_pos"]
        agent1_state=self.env.state.players[0].to_dict()
        invalid_pos = self.mdp.terrain_pos_dict[" "] #empty space is not valid

        if agent1_intend_pos not in invalid_pos: 
            greedy_plan = get_greedy_actions(self.agent1,agent1_intend_pos,self.env.state)
            #only render if we have greedy action
            self.render_action(
                mdp_surface, self.action2index[greedy_plan[0]], agent1_state,COLOR_RED)
        self.render_action(mdp_surface,self.action2index[next_action], agent1_state, COLOR_BLUE)
        
        self.render_target(mdp_surface, [agent1_intend_pos[0], agent1_intend_pos[1]] * 30)
        # self.screen.blit(mdp_surface,rect.topleft)
        pygame.image.save(mdp_surface, "my.png")

        self.manager.draw_ui(self.screen)
        pygame.display.update()

        pygame.display.flip()

        # add horizontal grid number
        for i in range(grid_x_num):
            # font white color, fira_code, size 20
            font = pygame.font.SysFont("fira_code", 22)
            self.text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(
                self.text_surface, (i * 30 + 10, self.env.mdp.height * 30 + 140 + 10)
            )
        # add vertical grid number
        for i in range(grid_y_num):
            font = pygame.font.SysFont("fira_code", 22)
            self.text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(
                self.text_surface, (self.env.mdp.width * 30 + 10, i * 30 + 140 + 10)
            )

    # record the game playthrough, save the log as pickle
    def on_cleanup(self):
        pygame.quit()

    def on_loop(self):
        time_delta = self.clock.tick(60) / 6000.0
        self.manager.update(time_delta)

        # if state changed: other agent holding/pot ingredient number,score.
        cur_state = self.env.state
        self.old_game_state = cur_state
    

    # Event handler
    def on_event(self, event):
        done = False
        self.manager.process_events(event)
        if event.type == pygame.KEYDOWN:

            pressed_key = event.dict['key']

            if pressed_key == pygame.K_RIGHT:
                self.next_frame()
                self.follow_greedy = 1
            elif pressed_key == pygame.K_LEFT:
                if self.curr_frame >=0:
                    self.prev_frame()
                    self.follow_greedy = 1
                else:
                    print("already at the start!!!")
            elif pressed_key == pygame.K_SPACE:
                self.follow_greedy = 0
            

        elif event.type == pygame.QUIT or done:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

    def _terminal(self):
        return self.env.mdp.is_terminal(self.env.state)

    def play(self):
        if self.on_init() == False:
            self._running = False

        import time
        while self._running and not self._terminal(): 
            for event in pygame.event.get():
                self.on_event(event)

            # time.sleep(0.2)
            
            self.on_loop()
            self.on_render()
            
        
        self.on_cleanup()

    def render_action(self, pygame_surface, action_index, agent,action_color = None):
        
        def colorize(image, newColor):
            image = image.copy()
            image.fill((0, 0, 0, 255), None, pygame.BLEND_RGBA_MULT)
            image.fill(newColor[0:3] + (0,), None, pygame.BLEND_RGBA_ADD)
            return image
        arrow = colorize(self.arrow,action_color)
        stay = colorize(self.stay,action_color)
        # target = colorize(self.target,action_color)
        interaction = colorize(self.interact,action_color)


        center_x = (agent["position"]*30)[0] * 30 + 15
        center_y = (agent["position"]*30)[1] * 30 + 15
        orientation = agent["orientation"]
        # Rotate and position the arrow based on the direction
        if action_index == 1:  # right
            # No rotation needed, assuming arrow points right by default
            arrow_rotated = pygame.transform.rotate(arrow, -90)
            center_x += 30
        elif action_index == 2:  # left
            arrow_rotated = pygame.transform.rotate(
                arrow, 90)  # Rotate 180 degrees to point right
            center_x -= 30
        elif action_index == 3:  # up
            # Rotate 90 degrees counter-clockwise to point up
            arrow_rotated = pygame.transform.rotate(arrow, 0)
            center_y -= 30
        elif action_index == 4:  # down
            # Rotate 90 degrees clockwise to point down
            arrow_rotated = pygame.transform.rotate(arrow, 180)
            center_y += 30
        elif action_index == 6:  # stay
            arrow_rotated = stay
            center_x = center_x
        elif action_index == 5:  # interact
            arrow_rotated = interaction
            center_x += 30 * orientation[0]
            center_y += 30 * orientation[1]

        rect = arrow_rotated.get_rect(center=(center_x, center_y))
        pygame_surface.blit(arrow_rotated, rect.topleft)
    def render_target(self, pygame_surface, pos):
        center_x = pos[0] * 30 + 15
        center_y = pos[1] * 30 + 15
        target = self.target
        rect = target.get_rect(center=(center_x, center_y))
        pygame_surface.blit(target, rect.topleft)

agent1 = SimulationAgent(mlam,"agent1",agent1_action_history,agent1_state_history)
agent2 = SimulationAgent(mlam,"agent2",agent2_action_history,agent2_state_history)


agent1.set_agent_index(0)
agent2.set_agent_index(1)

simulation = Simulation(env=base_env,mdp=world_mdp,agent1=agent1,agent2=agent2,traj=traj)
simulation.play()



#save generated prompts as pickle
from datetime import datetime
save_path_agent1 = f"reactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_agent1.json"
save_path_agent2 = f"reactive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_agent2.json"
agent1_prompts = list(simulation.agent1_prompts.values())
# agent2_prompts = list(simulation.agent2_prompts.values())
with open(os.path.join("trajectory_data", "processed", save_path_agent1), "w") as json_file:
    json.dump(agent1_prompts, json_file, indent=4)
    print(f"Agent 1 prompts saved to: {save_path_agent1}")
# with open(os.path.join("trajectory_data", "processed", save_path_agent2), "w") as json_file:
#     json.dump(agent2_prompts, json_file, indent=4)
#     print(f"Agent 2 prompts saved to: {save_path_agent2}")