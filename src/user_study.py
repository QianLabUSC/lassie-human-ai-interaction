import pygame
import pygame.scrap
from pygame.locals import *
import copy
from overcooked_ai_py.utils import generate_temporary_file_path, load_from_json
from overcooked_ai_py.static import TESTING_DATA_DIR
import os
import cv2
import pickle
import shutil
import json
import time as tm
import argparse
# from llm.llm_agent import llm_heurstics_model
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState # , OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import  Direction, Action, PlayerState, ObjectState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, OvercookedEnvPettingZoo
import overcooked_ai_py.agents.agent as agent
# import overcooked_ai_py.planning.planners as planners
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.utils import load_dict_from_file
from time import time
from llm.llm_api import generate_function
from planning.planners import LLMMediumLevelActionManager
from mdp.llm_overcooked_mdp import LLMOvercookedGridworld
import pygame_gui
import datetime
from overcooked_ai_py.utils import (
    read_layout_dict,
)



# Maximum allowable game time (in seconds)
MAX_GAME_TIME = 1000000

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

class OvercookedPygame():
    """     
    Class to run the game in Pygame.
    Args:
        - game_time: Number of seconds the game should last
        - agent1: Agent object for the first player (human player)
        - agent2: Agent object for the second player (llm agent)
        - logger: Logger object to log the game states
        - agent_level: Level of the agent (default, passive_mentee, active_supervisor)
    """
    def __init__(
            self,
            env,
            agent1,
            agent2,
            logger,
            gameTime = 30,
            user_feedback = "do anything",
            agent_level = "default"
    ):
        self._running = True
        self._paused = False
        self._updating = False
        self.logger = logger
        self.env = env
        self.score = 0 
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.max_players = 2
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent_level_ = agent_level
        self.start_time = time()
        self.init_time = time()
        # game window size + 50 border
        self.screen_width = INPUT_BOX_WIDTH
        # 140 for hud and 50 for the grid number
        self.screen_height = self.env.mdp.height * 30 + 140 + 50 +INPUT_BOX_HEIGHT
        self.human_symbol = ('<font color=#E784A2 size=4.5>'
                                    '<b>Human: </b></font>')
        self.agent_symbol = ('<font color=#4CD656 size=4.5>'
                                    '<b>Agent: </b></font>')
        self._response_recording = ('<b> ---Chat--- </b> <br> ')
        self.conversation_history = []
        self.human_feedback = user_feedback
        self.last_active_checkpoint = self.max_time
        self.check_interval = self.max_time/2
        file_path = 'llm/task/default_code.txt'
        # Open the file and read its contents
        with open(file_path, 'r', encoding='utf-8') as file:
            self.create_functions = file.read()
        self.env.mdp.human_event_list = []
        self.human_event_list = []

        self.last_image_time = 0  # Track the last time an image was saved
        self.image_interval = 100  # Interval in milliseconds (0.1 second)

    # helper function used to append the response to the text box
    def _append_response(self, response, name):
        symbol = ""
        if(name == "human"):
            symbol = self.human_symbol
        else:
            symbol = self.agent_symbol
        self._response_recording = self._response_recording + name + "says : " + response + "\n"
        self.conversation_history.append([name, response])
        self.text_box.append_html_text(symbol + response + ('<br>'))

    # Initialize the game
    def on_init(self):
        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height))#, pygame.RESIZABLE)
    
        self.screen.fill(COLOR_OVERCOOKED)
        # Initialize agents
        self.agent1.set_mdp(self.env.mdp)
        self.agent2.set_mdp(self.env.mdp)
        
        # 1 second (1000 milisecond) timer
        pygame.time.set_timer(TIMER, t)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ds = load_from_json(os.path.join(script_dir,
            "data", "config", "kitchen_config.json"))
        test_dict = copy.deepcopy(ds)
        print(test_dict["config"])
        self.state_visualizer = StateVisualizer(
            **test_dict["config"])
        self._running = True
        self.logger.env = self.env
        
        # initialize the UI manager
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock() # used by UImanager
        # initialize UI elements
        self.pause_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((INPUT_BOX_WIDTH-PAUSE_BUTTON_WIDTH ,0), (PAUSE_BUTTON_WIDTH, PAUSE_BUTTON_HEIGHT)),
                                             text='PAUSE',
                                             manager=self.manager)

        self.text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((0, self.screen_height -INPUT_BOX_HEIGHT), (INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)),
                                                 manager=self.manager,
                                                 placeholder_text='Enter Chat here...')
        self.text_box = pygame_gui.elements.UITextBox(  html_text=self._response_recording,
                                                        relative_rect=pygame.Rect((INPUT_BOX_WIDTH-CONVERSATION_BOX_WIDTH ,PAUSE_BUTTON_HEIGHT + 10), (CONVERSATION_BOX_WIDTH, self.screen_height - INPUT_BOX_HEIGHT - PAUSE_BUTTON_HEIGHT - 20)),
                                                        manager=self.manager,
                                                        object_id='abc')
        self.status_label = pygame_gui.elements.UITextBox(relative_rect=pygame.Rect((INPUT_BOX_WIDTH-4*PAUSE_BUTTON_WIDTH ,0), (3*PAUSE_BUTTON_WIDTH, PAUSE_BUTTON_HEIGHT)),
                                                          html_text="Agent Running",
                                                          manager=self.manager,
                                                            object_id='status')                     
        self.text_entry.disable() # text entry is disabled by default

        ## add grid number
        pygame.font.init()
        
    # Event handler
    def on_event(self, event):
        done = False
        player_1_action = Action.STAY
        player_2_action = Action.STAY
        self.manager.process_events(event)

        if event.type == TIMER:
            agent2_action, _ =self.agent2.action(self.env.state, self.create_functions)
            self._human_step_env(Action.STAY, agent2_action)
            self.env.mdp.step_environment_effects(self.env.state)
            state = self._get_state()
            # print(self.last_active_checkpoint)
            # print( state['time_left'])
            if self.agent_level_ == "active_supervisor":
                if self.last_active_checkpoint - state['time_left'] >= self.check_interval:
                    
                    self.last_active_checkpoint = state['time_left'] 
                
                    # pause the timer , aka agent action
                    pygame.time.set_timer(TIMER, 0)
                    self.text_entry.disable()
                    self.pause_button.disable()

                    self._paused = True

                    # pause the game timer
                    self.paused_at = time()
                    self.pause_button.set_text("RESUME")
                    self.status_label.set_text("Agent suggesting...")
                    self.manager.update(0.001)
                    self.manager.draw_ui(self.screen)
                    pygame.display.update()

                    print("game is paused")
                    
                    # needs to get agent feedback here 
                    # only keep the odd index of human event list since it is redundant
                    temp = self.human_event_list
                    self.human_event_list = self.human_event_list[1::2]
                    suggestions = self.ask_for_suggestions(self.human_event_list)
                    self.agent_supervison = "here are my suggestions: "  + suggestions
                    self.status_label.set_text("Agent paused...")
                    self.text_entry.enable()
                    #change human event list back to original, to avoid any potential bugs
                    self.human_event_list = temp
                    self._append_response(self.agent_supervison, "agent")

        #click button to pause the game and get user feedback using text box
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if hasattr(event, 'ui_element'):
                if event.ui_element == self.pause_button:
                    if self.agent_level_ == "default":
                        self.text_box.set_text("The default agent refuses any communication! You can do whatever you want to do to achieve highest efficiency!! ")
                    else:
                        if self._paused == False:
                            # pause the timer , aka agent action
                            pygame.time.set_timer(TIMER, 0)
                            self.text_entry.enable()
                            self.pause_button.set_text("RESUME")
                            self.status_label.set_text("Agent paused")
                            print("game is paused")
                            
                            self.pause_button.disable()

                            self._paused = True
                            self.human_feedback = self.text_entry.get_text()
                            # pause the game timer
                            self.paused_at = time()
                        # if the game is already paused, resume the game, and update the agent with LLM
                        else:
                            # start timer
                            self.text_entry.disable()
                            self.pause_button.set_text("PAUSE")
                            self.status_label.set_text("Agent updating...")
                            self.manager.update(0.001)
                            self.manager.draw_ui(self.screen)
                            pygame.display.update()
                            # get the code and analysis from the llm.
                            self.create_functions= self._llm_process_input(self.conversation_history, if_generate_code = True)
                            
                            self.status_label.set_text("Agent running")
                            self._response_recording = ('<b> ---Chat--- </b> <br> ')
                            self.text_box.set_text(self._response_recording)
                            # Record the time at which the game was paused
                            #stop the game timer 
                            # Calculate the paused duration and adjust start_time accordingly
                            paused_duration = time() - self.paused_at
                            self.start_time += paused_duration

                            self._paused = False
                            pygame.time.set_timer(TIMER, t)

        # Event handler for text entry, trigger when user press enter. 
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
            print("Entered text:", event.text)
            self._append_response(event.text, 'human')
            self.status_label.set_text("Agent responding...")
            
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            pygame.display.update()


            # call the llm to get the analysis based on the conversation history, if_generate_code = False, since we generating analysis
            analysis = self._llm_process_input(self.conversation_history, if_generate_code=False)
            self.human_feedback = self.text_entry.get_text()
            self._append_response(analysis, 'agent')
            self.status_label.set_text("Press resume to update")
            self.text_entry.set_text("")
            self.pause_button.enable()
            # print("***************************code  ***************************")
            # print(self.create_functions)
            # print("***************************analysis ***************************")
            # print(analysis)
            # print("*****************************************************")
        # Event handler for key press  
        if event.type == pygame.KEYDOWN:
            pressed_key = event.dict['key']

            if pressed_key == pygame.K_UP:
                player_1_action = Direction.NORTH
            elif pressed_key == pygame.K_RIGHT:
                player_1_action = Direction.EAST
            elif pressed_key == pygame.K_DOWN:
                player_1_action = Direction.SOUTH
            elif pressed_key == pygame.K_LEFT:
                player_1_action = Direction.WEST
            elif pressed_key == pygame.K_SPACE:
                player_1_action = Action.INTERACT
            if player_1_action in Action.ALL_ACTIONS and player_2_action in Action.ALL_ACTIONS:
                if not self._paused:
                    #update the game state
                    done = self._human_step_env(player_1_action, player_2_action)
        if event.type == pygame.QUIT or done:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

    # update game timer, log image frame.
    def on_loop(self):
        time_now_in_millisecond = round((time() - self.init_time) *1000)
        if not self._paused:
            self.logger.env = self.env
            time_now_in_milisecond_start_time = round(time() * 1000 - self.start_time *1000)
            self.env.state.timestep = float('%.1f'%(time_now_in_milisecond_start_time/1000))
        time_delta = self.clock.tick(60)/6000.0
        self.manager.update(time_delta)
        # # HACK TODO: be sure to comment out step_environment_effects in overcooked_mdp. so that time is not updated when action applied 
        
        #for 0.1s save an image
        if time_now_in_millisecond - self.last_image_time >= self.image_interval:
            if self.logger.video_record:
                frame_name = self.logger.img_name(time_now_in_millisecond/ 1000)
                pygame.image.save(self.screen, frame_name)
                self.last_image_time = time_now_in_millisecond  # Update the last saved time


    # render the game state
    def on_render(self):
        def _customized_hud_data(state, **kwargs):
            result = {
                "time left": self.max_time - round(time() - self.start_time),
                "all_orders": [r.to_dict() for r in state.all_orders],
                "bonus_orders": [r.to_dict() for r in state.bonus_orders],
                "score": self.score,
            }
            result.update(copy.deepcopy(kwargs))
            return result
        grid_x_num = len(self.env.mdp.terrain_mtx[0])
        grid_y_num = len(self.env.mdp.terrain_mtx)
        kitchen = self.state_visualizer.render_state(
            self.env.state, self.env.mdp.terrain_mtx, hud_data=_customized_hud_data(self.env.state)
        )
        self.screen.blit(kitchen, (0, 0))

        self.manager.draw_ui(self.screen)
        pygame.display.update()

        ## logginng
        self.logger.env = self.env

        pygame.display.flip()

        #add horizontal grid number
        for i in range(grid_x_num):
            # font white color, fira_code, size 20
            font = pygame.font.SysFont('fira_code', 22)
            self.text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(self.text_surface, (i*30+10, self.env.mdp.height * 30 + 140+10))
        #add vertical grid number
        for i in range(grid_y_num):
            font = pygame.font.SysFont('fira_code', 22)
            self.text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(self.text_surface, (self.env.mdp.width* 30 +10 ,i*30 + 140+10))
    # record the game playthrough, save the log as pickle
    def on_cleanup(self):
        self.logger.save_log_as_pickle()
        if self.logger.video_record:
            self.logger.create_video()
        pygame.quit()


    def on_execute(self):
        if self.on_init() == False:
            self._running = False
        while self._running and not self._time_up():
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

    def _time_up(self):
        if self._paused:
            return False  # Game is paused; don't end the game based on time
        current_time = time()
        elapsed_time = current_time - self.start_time
        return elapsed_time > self.max_time

    # update the game state based on the human and agent action
    def _human_step_env(self, human1_action,agent1_action):
        joint_action = (human1_action, agent1_action)
        prev_state = self.env.state
        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action, joint_agent_action_info =[{"1"},{"2"}])

        self.state = next_state
        curr_reward = sum(info["sparse_r_by_agent"])
        self.score += curr_reward
        # log joint action with time stamp
        log = {"time": self.env.state.timestep, "joint_action": joint_action,"state":self.env.state.to_dict(), "score": self.score, "user_feedback": self.human_feedback}
        self.logger.episode.append(log)
        self.human_event_list = self.env.mdp.human_event_list
        return done
    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict
    # process the input text and generate the analysis and code(optional)
    def _llm_process_input(self, input_text, if_generate_code):
        # call GPT to get modified mlam function  mlam.ml_action(self, state)
        response = generate_function(input_text, if_generate_code,agent2=self.agent2,state=self.env.state)
        return response
    
    # get active agent suggestions
    def ask_for_suggestions(self,human_action_history):
        response = generate_function(self.conversation_history, False, human_action_history)
        return response


class Logger:
    def __init__(self, config, filename, agent1=None, agent2=None, video_record=False):
        self.participant_id = config.participant_id
        self.json_filename = filename+'.json'
        self.filename = filename
        self.video_record = config.record_video
        self.user_feedback = config.user_feedback
        self.agent_level = config.agent_level
        # create log folder
        self.log_folder = os.path.join(
            USER_STUDY_LOG, str(self.participant_id))
        print(self.log_folder)
        self.img_dir = os.path.join(self.log_folder, 'img')

        if not os.path.exists(self.log_folder):
            # shutil.rmtree(self.log_folder)
            os.makedirs(self.log_folder)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        self.img_name = lambda timestep: f"{self.img_dir}/{int(timestep*10):05d}.png"

        # game info
        # self.env = config.base_env
        self.layout_name = config.layout_name
        self.agent1 = agent1
        self.agent2 = agent2
        self.episode = []
    def save_log_as_pickle(self):
        with open(os.path.join(self.log_folder, self.json_filename), 'w') as file:
            json.dump({"layout_name": self.layout_name,
                       "participant_id": self.participant_id,
                       "total_time": self.env.state.timestep,
                       "agent_level": self.agent_level,
                      "episode": self.episode}, file)
        print(f"Pickle log saved to {self.json_filename}")

    """
        Create video from images 
    """
    def create_video(self):
        # os.system("ffmpeg -r 5 -i \"{}%*.png\"  {}{}.mp4".format(self.img_dir +
        #         '/', self.log_folder+'/', self.filename))
        images = [img for img in os.listdir(self.img_dir) if img.endswith(".png")]
        frame = cv2.imread(os.path.join(self.img_dir, images[0]))
        height, width, layers = frame.shape
        video_name = '{}{}.mp4'.format(self.log_folder+'/', self.filename)

        video = cv2.VideoWriter(video_name, 0, 10, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(self.img_dir, image)))
        cv2.destroyAllWindows()
        video.release()
        shutil.rmtree(self.img_dir)



class StudyConfig:
    def __init__(self, args):
        self.participant_id = args.participant_id
        self.layout_name = args.layout
        layout_file_name = self.layout_name + ".layout"
        self.log_file_name = args.log_file_name
        self.record_video = args.record_video
        self.user_feedback = args.user_feedback
        self.agent_level = args.agent_level
        self.total_time = args.total_time
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Copy layout to Overcooked AI code base
        path_from = os.path.join(
            script_dir, "data", "layout", layout_file_name)
        # Construct the absolute path by joining the script directory with the relative path
        path_to = os.path.join(script_dir, "..", "overcooked_ai", "src", 
                               "overcooked_ai_py", "data", "layouts", layout_file_name)

        # path_to = os.path.join( "..","overcooked_ai", "src",
        #                        "overcooked_ai_py", "data", "layouts", layout_file_name)
        shutil.copy(path_from, path_to)

        self.world_mdp = LLMOvercookedGridworld.from_layout_name(
                self.layout_name)

        self.base_env = OvercookedEnv.from_mdp(
            self.world_mdp)
            ### chage LLM description file to include the layout
        base_layout_params = read_layout_dict(self.layout_name)
        print(base_layout_params)
        grid = base_layout_params["grid"]
        order_list = base_layout_params["start_all_orders"]
        #clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        file_path = 'llm/task/des_1_layout_template.txt'
        new_file_path = 'llm/task/des_1_layout.txt'
        default_code_path = 'llm/task/default_code.txt'
        # Open the default_code_path and read its contents
        with open(default_code_path, 'r', encoding='utf-8') as file:
            default_code = file.read()
        # Step 1: Read all lines from the original file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Step 2: Insert new text at line 6
        lines.insert(5, str(grid) +"\n")  # 0-based index; line 4 is at index 3
        lines.insert(9, str(order_list) +"\n")  # 0-based index; line 4 is at index 3
        lines.insert(27, default_code +"\n")  
        # Step 3: Write the modified content to a new file
        with open(new_file_path, 'w') as file:
            file.writelines(lines)



def initialize_config_from_args():
    parser = argparse.ArgumentParser(
        description='Initialize configurations for a human study.')

    ### Args for the game setup ###
    parser.add_argument('--layout', type=str, default='0_trial_option_coordination',
                        help='List of tasks to be performed in the study')
    parser.add_argument('--total_time', type=int, default=MAX_STEPS,
                        help='Total time to given to complete the game')

    # The following game config options are still undergoing construction
    # parser.add_argument('--served_in_order', type=bool, help='Complete the order list in order')
    # parser.add_argument('--single_player', type=bool, help='Single player mode: one human controlled agent collaborating with a modeled greedy agent')


    ### Args for the study ###
    parser.add_argument('--participant_id', type=int,
                        help='ID of participants in the study', default=0)
    parser.add_argument('--log_file_name', type=str,
                        default='', help='Log file name')
    parser.add_argument('--record_video', dest='record_video',
                        action='store_true', help='Record video during replay')
    parser.add_argument('--no-record_video', dest='record_video',
                        action='store_false', help='Do not record video during replay')
    parser.add_argument('--agent_level', type= str, default='default', help='agent levels: default, passive_mentee, active_supervisor')
    parser.add_argument('--user_feedback', type= str, default="do anything", help='user feedback in text')
    args = parser.parse_args()

    if args.log_file_name == '':
        x = datetime.datetime.now()
        datestr = x.strftime("%m-%d-%f")
        print(datestr)
        args.log_file_name = '-'.join([str(args.participant_id), args.layout,args.agent_level, datestr])

    return StudyConfig(args)


if __name__ == "__main__" :
    study_config = initialize_config_from_args()
    print("Study Configuration Initialized:")
    print(f"Participant ID: {study_config.participant_id}")
    print(f"Layout: {study_config.layout_name}")
    print(f"agent level: {study_config.agent_level}")
    print(f"user feedback: {study_config.user_feedback}")
    # Initialize two human agent
    agent1 = agent.Agent()
    # agent2 = agent.FixedPlanAgent([stay, w, w, e, e, n, e, interact, w, n, interact])
    FULL_PARAMS = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": study_config.world_mdp.terrain_pos_dict["X"],
    "counter_drop": study_config.world_mdp.terrain_pos_dict["X"],
    "counter_pickup": study_config.world_mdp.terrain_pos_dict["X"],
    "same_motion_goals": False,
}
    mlam = LLMMediumLevelActionManager(study_config.world_mdp,FULL_PARAMS)
    agent2 = llm_heurstics_model(mlam=mlam)
    agent1.set_agent_index(0)
    agent2.set_agent_index(1)
    

    # Initialize logging
    logger = Logger(study_config, study_config.log_file_name, agent1=agent1, agent2=agent2)

    # Run the game
    start_time = time()

    theApp = OvercookedPygame(study_config.base_env, agent1, agent2,logger, gameTime=study_config.total_time,user_feedback=study_config.user_feedback, agent_level=study_config.agent_level)
    theApp.on_execute()
    print("It took {} seconds for playing the entire level".format(time() - start_time))

