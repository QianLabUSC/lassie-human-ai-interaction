import os
import copy
import pygame
import pygame_gui
import sys
from pygame.locals import *
from time import time, sleep
from config import initialize_config_from_args
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai/src')))
from overcooked_ai_py.utils import load_from_json
from constants import (
    MAX_GAME_TIME, TIMER, t, COLOR_OVERCOOKED, WHITE,
    CONVERSATION_BOX_WIDTH, INPUT_BOX_HEIGHT,INPUT_BOX_WIDTH, PAUSE_BUTTON_HEIGHT, PAUSE_BUTTON_WIDTH
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action
import threading

class OvercookedPygame():
    """     
    Class to run the game in Pygame.
    Args:
        - game_time: Number of seconds the game should last
        - agent1: Agent object for the first player (human player)
        - agent2: Agent object for the second player (llm agent)
        - logger: Logger object to log the game states
        - user_feedback: User feedback to the agent ###dreprecated
        - agent_level: Level of the agent (default, passive_mentee, active_supervisor) ###dreprecated
    """
    def __init__(
            self,
            study_config,
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
        self.env = study_config.base_env
        self.score = 0 
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.max_players = 2
        self.agent1 = agent1
        self.agent2 = agent2
        self.agent_level_ = agent_level
        self.start_time = time()
        self.init_time = time()
        self.prev_timestep = 0
        self.prev_manager_timestep = 0

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

        # file_path = 'llm/task/default_code.txt'
        # # Open the file and read its contents
        # with open(file_path, 'r', encoding='utf-8') as file:
        #     self.create_functions = file.read()
        self.env.mdp.human_event_list = []
        self.human_event_list = []

        self.last_image_time = 0  # Track the last time an image was saved
        self.image_interval = 100  # Interval in milliseconds (0.1 second)
        self.player_1_action = Action.STAY

        self.manager_response_count_to_finish_dish = []
        self.reactive_response_count_to_finish_dish = []
        self.prev_score = 0
        
        # To access user_mode from cmd args 
        study_config = initialize_config_from_args()
        print(f"User Mode at Overcooked: {study_config.user_mode}")     
        self.usermode = study_config.user_mode
        
        # TODO: LATER IT WILL COME FROM LLM ACTION
        # FROM  ACTION from functioin self.agent2.action
        # self.player_2_action,_, self.robotfeedbackflag : json value = self.agent2.action(self.env.state, self.screen)
        # self.robotfeedback ={}
        self.robotfeedback = {
                "constant_feedback":{
                        "value": "", # Placeholder for the actual constant feedback value
                        "is_updated": False # Flag indicating if this feedback has been updated
                    },
                "frequent_feedback":{
                        "value": "", # Placeholder for the actual frequency feedback value
                        "is_updated": False # Flag indicating if this feedback has been updated
                    },
                "hasAgentPaused":False # Used only For mode 3, since in mode 3 At the beginning Agent will pause the game
            }

    # helper function used to append the response to the text box
    def _append_response(self, response, name):
        symbol = ""
        if(name == "human"):
            symbol = self.human_symbol
        else:
            symbol = self.agent_symbol
        self._response_recording = self._response_recording + name + "says : " + response + "\n"
        self.conversation_history.append([name, response])
        self.chat_box.append_html_text(symbol + response + ('<br>'))
    
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
        self.agent1.name = "agent1"
        self.agent2.name = "agent2"

        # 1 second (1000 milisecond) timer
        pygame.time.set_timer(TIMER, t)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(script_dir)
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
        if self.usermode == 1:
            self.text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((0, self.screen_height -INPUT_BOX_HEIGHT), (INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)),
                                                 manager=self.manager,
                                                 placeholder_text='User Interaction is not allowed as You are in Mode 1')
            self.agent2.set_human_preference("please don't consider other human chef, you just want to finish task independently. ")
        else:    
            self.text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((0, self.screen_height -INPUT_BOX_HEIGHT), (INPUT_BOX_WIDTH, INPUT_BOX_HEIGHT)),
                                                 manager=self.manager,
                                                 placeholder_text='Enter Chat here...')
        self.chat_box = pygame_gui.elements.UITextBox(  html_text=self._response_recording,
                                                        relative_rect=pygame.Rect((INPUT_BOX_WIDTH-CONVERSATION_BOX_WIDTH ,PAUSE_BUTTON_HEIGHT + 10), (CONVERSATION_BOX_WIDTH, self.screen_height - INPUT_BOX_HEIGHT - PAUSE_BUTTON_HEIGHT - 20)),
                                                        manager=self.manager,
                                                        object_id='abc')
        self.status_label = pygame_gui.elements.UITextBox(relative_rect=pygame.Rect((INPUT_BOX_WIDTH-5*PAUSE_BUTTON_WIDTH ,0), (4*PAUSE_BUTTON_WIDTH, PAUSE_BUTTON_HEIGHT)),
                                                          html_text="Agent Running",
                                                          manager=self.manager,
                                                            object_id='status')                     
        #self.text_entry.disable() # text entry is disabled by default
        

        
        # Disable/Enable chatbox and pause/resume buttons based on the mode
        if self.usermode == 1:
            self.pause_button.disable()
            self.text_entry.disable()
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            self.on_render()
        elif self.usermode == 2:
            self.pause_button.enable()
            self.text_entry.enable()
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            self.on_render()
        elif self.usermode == 3: 
            
            self._pause_game()
            self.status_label.set_text("Agent pause game and generate suggestions...")
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            self.on_render()
            human_intention, reactive_rules = self.agent2.reactive_query(self.env.state)
            self._append_response(reactive_rules, 'agent')
            self.pause_button.enable()   

        elif self.usermode == 4: 
            
            self._pause_game()
            self.status_label.set_text("Agent pause game and generate suggestions...")
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            self.on_render()
            human_intention, reactive_rules = self.agent2.reactive_query(self.env.state)
            self._append_response(reactive_rules, 'agent')
  
            self.pause_button.enable()
          
            

            
        ## add grid number
        pygame.font.init()
        
    # Event handler
    def on_event(self, event):
        done = False
        player_1_action = Action.STAY
        # Players stay in place if no keypress are detected
        if event.type == TIMER:
            self.env.mdp.step_environment_effects(self.env.state)
            if self.usermode == 3 or self.usermode == 4:
                if self.robotfeedback["hasAgentPaused"] == True:
                    # pause the timer
                    self._pause_game()
                    human_intention, reactive_rules = self.agent2.reactive_interactive_query(self.env.state)
                    self._append_response(reactive_rules, 'agent')
              
                self.pause_button.enable()
                self.text_entry.enable()
                


            if self.usermode == 4:
                if self.robotfeedback["frequent_feedback"]["is_updated"] == True :
                    self._append_response(self.robotfeedback["frequent_feedback"]["value"], 'agent')


        if event.type == pygame.KEYDOWN:
            
            pressed_key = event.dict['key']
            # check if they are human players
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
        
            # check if action is valid
            if player_1_action in Action.ALL_ACTIONS:
                self.player_1_action = player_1_action
                self.agent2.record_human_log(self.env.state.players[0], self.player_1_action)
                # self._agents_step_env(self.player_1_action, Action.STAY)
        if event.type == pygame.QUIT or done:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

        ## comment ouut the UI interaction for now.
        self.manager.process_events(event)

        

        #click button to pause the game
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if hasattr(event, 'ui_element'):
                if event.ui_element == self.pause_button:
                    # if the game is not paused, pause the game
                    if self._paused == False:
                        # pause the timer
                        self._pause_game()
                    else:
                        # start timer
                        self._resume_game()
                        self.manager.update(0.001)
                        self.manager.draw_ui(self.screen)
                        pygame.display.update()
                                                
                        # self._response_recording = ('<b> ---Chat--- </b> <br> ')
                        # self.chat_box.set_text(self._response_recording)
                        

        # Event handler for text entry, trigger when user press enter. 

        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED and self.usermode!=1 : #Dont show the robot response in mode 1
            print("Entered text:", event.text)
            self._append_response(event.text, 'human')
            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            self.on_render()
            self.status_label.set_text("Agent responding...")
            #when user finish typing,  update prompt with user preference
            
            # this event.text from human should go to llm in  mode 2, mode 3, & mode 4, User is allowed to enter text in chatUI)
            if self.usermode !=1:
                self._pause_game()
                self.agent2.set_human_preference(event.text)
                
                human_intention, reactive_pos, response_plan = self.agent2.reactive_interactive_query(self.env.state)
                self._append_response(f"human wants to {human_intention}, I will, {response_plan} and first move to {reactive_pos}", 'agent')
                self.manager.update(0.001)
                self.manager.draw_ui(self.screen)
                self.on_render()
           
            # after agent response, resume the game. 
            self._resume_game()

            self.manager.update(0.001)
            self.manager.draw_ui(self.screen)
            pygame.display.update()
            
    def on_loop(self,action_fps=2):
        self.logger.env = self.env
        time_step = round((time() - self.init_time) * action_fps)
        time_delta = self.clock.tick(60)/6000.0
        self.manager.update(time_delta)
            
        ## change onloop to update game at 10fps, 60 fps, apply joint action, update logger
        ## step environment every 0.01s/10ms,
        # 1 second = 1000ms
        if(time_step > self.prev_timestep and not self._paused):
            self.prev_timestep = time_step
            self.robotfeedback = self.agent2.subtasking(self.env.state)
            print(self.robotfeedback)
            self.player_2_action,_ = self.agent2.action(self.env.state, self.screen)
            # print("Actual step:", self.player_2_action)
            # self.player_2_action = Action.STAY
            
            #TODO: TO USE FOR OTHER MODES

            if self.usermode == 3:
                print('at time paused1', self.robotfeedback["hasAgentPaused"])
                sleep(3) # this sleep for time being there will be no need of this later, 
                self.robotfeedback["hasAgentPaused"] = False
                print('at time paused2', self.robotfeedback["hasAgentPaused"])
         
           
            
            done = self._agents_step_env(self.player_1_action, self.player_2_action)        
            joint_action = (self.player_1_action, self.player_2_action)

            # log user behavior to json
            log = {"state":self.env.state.to_dict(),"joint_action": joint_action, "score": self.score}
            self.logger.episode.append(log)

            # reinitialize action
            self.player_1_action = Action.STAY
            self.player_2_action = Action.STAY
            if self.logger.video_record:
                frame_name = self.logger.img_name(time_step/action_fps)
                pygame.image.save(self.screen, frame_name)
                # 

            if done:
                self._running = False
  
        #if score changes, update the number of
        if self.score != self.prev_score:
            self.manager_response_count_to_finish_dish.append(self.agent2.manager_response_count)
            self.reactive_response_count_to_finish_dish.append(self.agent2.reactive_response_count)
        self.prev_score = self.score

   
    # render the game state
    def on_render(self):
        def _customized_hud_data(state, **kwargs):
            result = {
                "time left":  self.max_time - round(time() - self.start_time) if not self._paused else self.max_time - round(self.paused_at - self.start_time),
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
#        self.agent2.stop_all_threads()
        if self.logger.video_record:
            self.logger.create_video()
        pygame.quit()

    def on_execute(self):
        
        if self.on_init() == False:
            self._running = False
        while self._running and not self._time_up():
            # handle event and keyboard input
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
            self.on_loop()
            
        self.on_cleanup()
    def _time_up(self):
        return time() - self.start_time > self.max_time

    def _pause_game(self):
        self._paused = True
        self.pause_button.set_text("RESUME")
        self.status_label.set_text("Agent paused")
        self.text_entry.enable()
        self.paused_at = time()
       
        print("game is paused!")
    def _resume_game(self):
        self._paused = False
        self.pause_button.set_text("PAUSE")
        self.status_label.set_text("Agent running")
        self.text_entry.disable()
        paused_duration = time() - self.paused_at
        self.start_time += paused_duration
        
        print("game is resumed!")

    def _agents_step_env(self, agent1_action, agent2_action):
        joint_action = (agent1_action, agent2_action)
        prev_state = self.env.state
        self.state, info = self.env.mdp.get_state_transition(
                prev_state, joint_action
                )

        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        next_state, timestep_sparse_reward, done, info = self.env.step(joint_action, joint_agent_action_info =[{"1"},{"2"}])
        return done

    def _get_state(self):
        state_dict = {}
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict



