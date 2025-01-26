import os
import io
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

class OvercookedUI:
    """
    Class to manage UI components of the Overcooked game.
    """
    def __init__(self, screen_width, screen_height):
        self.clock = pygame.time.Clock()  # Used by UImanager

        # UI Elements
        self.pause_button = None
        self.text_entry = None
        self.chat_box = None
        self.status_label = None
        self._response_recording = ('<b> ---Chat--- </b> <br> ')
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.human_symbol = ('<font color=#E784A2 size=4.5>'
                                    '<b>Human: </b></font>')
        self.agent_symbol = ('<font color=#4CD656 size=4.5>'
                                    '<b>Agent: </b></font>')

    def init_ui(self, user_mode, input_box_width, input_box_height, pause_button_width, pause_button_height, conversation_box_width, manager, screen):
        self.manager = manager
        self.screen = screen
        # Initialize UI elements
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((input_box_width - pause_button_width, 0), (pause_button_width, pause_button_height)),
            text='PAUSE',
            manager=self.manager
        )

        placeholder_text = 'User Interaction is not allowed as You are in Mode 1' if user_mode == 1 else 'Enter Chat here...'
        self.text_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((0, self.screen_height - input_box_height), (input_box_width, input_box_height)),
            manager=self.manager,
            placeholder_text=placeholder_text
        )

        self.chat_box = pygame_gui.elements.UITextBox(
            html_text=self._response_recording,
            relative_rect=pygame.Rect(
                (input_box_width - conversation_box_width, pause_button_height + 10),
                (conversation_box_width, self.screen_height - input_box_height - pause_button_height - 20)
            ),
            manager=self.manager,
            object_id='abc'
        )

        self.status_label = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((input_box_width-pause_button_width-300, 0), (2 * pause_button_width, pause_button_height)),
            html_text="Agent Running",
            manager=self.manager,
            object_id='status'
        )

    def append_response(self, response, name):
        # Helper function to append response to chat box
        symbol = self.human_symbol if name == "human" else self.agent_symbol
        self._response_recording += f"{name} says: {response}\n"
        self.chat_box.append_html_text(symbol + response + '<br>')

    def change_status(self, status_text):
        self.status_label.set_text(status_text)
        

    def enable_text_entry(self):
        self.text_entry.enable()

    def disable_text_entry(self):
        self.text_entry.disable()

    def enable_pause_button(self):
        self.pause_button.enable()

    def disable_pause_button(self):
        self.pause_button.disable()

    def set_ui_to_pause(self):
        
        self.pause_button.set_text("RESUME")
        self.status_label.set_text("Agent paused")
        self.text_entry.enable()
        self.manager.update(0.001)
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    
    def set_ui_to_resume(self):
       
        self.pause_button.set_text("PAUSE")
        self.status_label.set_text("Agent running")
        self.text_entry.disable()
        self._response_recording = ('<b> ---Chat--- </b> <br> ')
        self.chat_box.set_text(self._response_recording)
        self.manager.update(0.001)
        self.manager.draw_ui(self.screen)
        pygame.display.update()

    def text_finish_callback(self, text):
        self.append_response(text, 'human')
        print(text)
        self.status_label.set_text("Agent responding...")
        self.manager.update(0.001)
        self.manager.draw_ui(self.screen)
        

    def robot_generate_callback(self, text):
        self.append_response(text, 'robot')
        
        self.status_label.set_text("Agent responding...")
        self.manager.update(0.001)
        self.manager.draw_ui(self.screen)
       



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
            user_feedback = "do anything" 
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
        self.start_time = time()
        self.init_time = time()
        self.prev_timestep = 0
        self.prev_manager_timestep = 0
        
        self.screen_width = INPUT_BOX_WIDTH + 460
        # 140 for hud and 50 for the grid number
        self.screen_height = self.env.mdp.height * 30 + 140 + 50 +INPUT_BOX_HEIGHT
        self.ui = OvercookedUI(self.screen_width, self.screen_height)
        self.player_1_action = None
        self.player_2_action = None
        self.manager_response_count_to_finish_dish = []
        self.reactive_response_count_to_finish_dish = []
        self.prev_score = 0
        
        
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
        
        # Initialize UI
        self.ui.init_ui(
            user_mode=1,  # Example mode
            input_box_width=INPUT_BOX_WIDTH,
            input_box_height=INPUT_BOX_HEIGHT,
            pause_button_width=PAUSE_BUTTON_WIDTH,
            pause_button_height=PAUSE_BUTTON_HEIGHT,
            conversation_box_width=CONVERSATION_BOX_WIDTH,
            manager=self.manager,
            screen=self.screen
        )                  
        
        self.agent2.initilize_Graph(self.env.state)
        
        # graph_surface = pygame.image.load("init_graph.png")
        # graph_surface = pygame.transform.smoothscale(graph_surface, (450, 450))

        # self.screen.blit(graph_surface, (INPUT_BOX_WIDTH ,20))

        
        # Disable/Enable chatbox and pause/resume buttons based on the mode
        if self.agent2.usermode == 1:
            # self._disable_human_interrupt()
            # self._disable_human_input()
            # self._changing_status("Agent running...")
            pass
        elif self.agent2.usermode == 2:
            # self._enable_human_interrupt()
            # self._enable_human_input()
            # self._changing_status("Agent running...")
            pass
        elif self.agent2.usermode == 3: 
            
            self._pause_game()
            # self._changing_status("Agent pause game and generate suggestions...")
            # human_intention, reactive_rules = self.agent2.reactive_query(self.env.state)
            # self._append_response(reactive_rules, 'agent')
            # self._enable_human_interrupt()

        elif self.agent2.usermode == 4: 
            
            self._pause_game()
            # self._changing_status("Agent pause game and generate suggestions...")
            # human_intention, reactive_rules = self.agent2.reactive_query(self.env.state)
            # self._append_response(reactive_rules, 'agent')
            # self._enable_human_interrupt()
            
          
            

            
        ## add grid number
        pygame.font.init()
        
    # Event handler
    def on_event(self, event):
        done = False
        player_1_action = Action.STAY
        # Players stay in place if no keypress are detected
        if event.type == TIMER:
            self.env.mdp.step_environment_effects(self.env.state)
            # if self.usermode == 3 or self.usermode == 4:
            #     if self.robotfeedback["hasAgentPaused"] == True:
            #         # pause the timer
            #         self._pause_game()
            #         human_intention, reactive_rules = self.agent2.reactive_interactive_query(self.env.state)
            #         self._append_response(reactive_rules, 'agent')
              
            #     self._enable_human_interrupt()
            #     self._enable_human_input()
                


            # if self.usermode == 4:
            #     if self.robotfeedback["frequent_feedback"]["is_updated"] == True :
            #         self._append_response(self.robotfeedback["frequent_feedback"]["value"], 'agent')
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
                self.agent2.record_human_log(self.env.state.players[0], self.env.state.to_dict().pop("objects"), self.player_1_action)
               
        if event.type == pygame.QUIT or done:
            # game over when user quits or game goal is reached (all orders are served)
            self._running = False

        # handle human input
        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if hasattr(event, 'ui_element'):
                if event.ui_element == self.ui.pause_button:
                    # if the game is not paused, pause the game
                    if self._paused == False:
                        self._pause_game()
                    else:
                        # start timer
                        self._resume_game()

        # Event handler for text entry, trigger when user press enter. 
        if event.type == pygame_gui.UI_TEXT_ENTRY_FINISHED : #Dont show the robot response in mode 1
            self.ui.text_finish_callback(event.text)
            self._pause_game()
            self.agent2.communicating(self.env, self.ui, event.text)


        self.manager.process_events(event)
    def on_loop(self,action_fps=4):
        while(True):
            self.logger.env = self.env
            time_step = round((time() - self.init_time) * action_fps)
            if not self._running:
                break
            else:
                if(time_step > self.prev_timestep and not (self._time_up() and not self._paused)
                    and (self.player_2_action is None)):
                    self.prev_timestep = time_step
                    self.agent2.subtasking(self.env.state, self.ui)
                    self.player_2_action, _ = self.agent2.action(self.env.state)
                    
                        
                    
                    if self.logger.video_record:
                        frame_name = self.logger.img_name(time_step/action_fps)
                        pygame.image.save(self.screen, frame_name)
                        # 


   
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
        # add a node graph beside the game 
        self.agent2.dm.node_graph.draw_graph_cairo('graph.png')
        node_graph_surface = pygame.image.load("graph.png")
        node_graph_surface = pygame.transform.smoothscale(node_graph_surface, (400, 400))

        self.screen.blit(node_graph_surface, (INPUT_BOX_WIDTH ,20))

        # for i in range(5):
        #     try:
        #         node_graph_surface = pygame.image.load("init_graph.png")
        #         self.screen.blit(node_graph_surface, (INPUT_BOX_WIDTH ,20))
        #         break
        #     except Exception as e:
        #         print(f"Load failed: {e}, retrying...")
        #         sleep(1)
        # pygame.image.load("init_graph.png")
               #add a node graph beside the game 

        

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
        agent_thread = threading.Thread(target=self.on_loop)
        agent_thread.start()
        while self._running and not (self._time_up() and not self._paused):
            # handle event and keyboard input
            for event in pygame.event.get():
                self.on_event(event)
    
            if self.player_2_action:
                
                if self.player_1_action:
                    print("we have new player 2 action")
                    self.joint_action = (self.player_1_action, self.player_2_action)
                else:
                    self.joint_action = (Action.STAY, self.player_2_action)
                self.player_2_action = None

            else:
                
                if self.player_1_action:
                    self.joint_action = (self.player_1_action, Action.STAY)
                else:
                    self.joint_action = (Action.STAY, Action.STAY)
                self.player_1_action = None
                # reinitialize action
            # print(self.join.
            # 0
            #                                                                                                                        
            #     
            #                 
            #      
            #                                                                                    t_action)
            
            done = self._agents_step_env(self.joint_action[0], self.joint_action[1])  
                # log user behavior to json
            log = {"state":self.env.state.to_dict(),"joint_action": self.joint_action, "score": self.score}
            self.logger.episode.append(log)
         
            if done:
                self._running = False

            #if score changes, update the number of
            if self.score != self.prev_score:
                self.manager_response_count_to_finish_dish.append(self.agent2.manager_response_count)
                self.reactive_response_count_to_finish_dish.append(self.agent2.reactive_response_count)
            self.prev_score = self.score

            self.on_render()
            time_delta = self.clock.tick(60)/6000.0
            self.manager.update(time_delta)
        
        agent_thread.join()
        pygame.quit()
            
            
        self.on_cleanup()
    def _time_up(self):
        return time() - self.start_time > self.max_time

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

    def _pause_game(self):
        self._paused = True
        self.ui.set_ui_to_pause()
        self.paused_at = time()
       
        print("game is paused!")


    def _resume_game(self):
        self._paused = False
        self.ui.set_ui_to_resume()
        paused_duration = time() - self.paused_at
        self.start_time += paused_duration
        
        print("game is resumed!")

