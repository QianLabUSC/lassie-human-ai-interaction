import os
import io
import copy
import sys
import threading
from time import time, sleep

import pygame
import pygame_gui
from pygame.locals import *

from config import initialize_config_from_args
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'overcooked_ai/src')))
from overcooked_ai_py.utils import load_from_json
from constants import (
    MAX_GAME_TIME, TIMER, t, COLOR_OVERCOOKED, WHITE,
    CONVERSATION_BOX_WIDTH, INPUT_BOX_HEIGHT, INPUT_BOX_WIDTH, PAUSE_BUTTON_HEIGHT, PAUSE_BUTTON_WIDTH
)
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_mdp import Direction, Action


class OvercookedUI:
    def __init__(self, screen_width, screen_height):
        self.pause_button = None
        self.chat_box = None
        self.status_label = None
        self._response_recording = ('<b> ---Chat--- </b> <br> ')
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.human_symbol = '<font color=#E784A2 size=4.5><b>Human: </b></font>'
        self.agent_symbol = '<font color=#4CD656 size=4.5><b>Agent: </b></font>'

    def init_ui(self, user_mode, input_box_width, input_box_height, pause_button_width, pause_button_height, conversation_box_width, manager, screen, clock):
        self.manager = manager
        self.screen = screen
        self.clock = clock
        self.pause_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((input_box_width - pause_button_width, 0), (pause_button_width, pause_button_height)),
            text='PAUSE',
            manager=self.manager
        )

        self.chat_box = pygame_gui.elements.UITextBox(
            html_text=self._response_recording,
            relative_rect=pygame.Rect(
                (input_box_width - conversation_box_width, pause_button_height + 10),
                (conversation_box_width, self.screen_height - input_box_height - pause_button_height - 20 - 470)
            ),
            manager=self.manager
        )

        self.status_label = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((input_box_width - pause_button_width - 300, 0), (2 * pause_button_width, pause_button_height)),
            html_text="Agent Running",
            manager=self.manager
        )

    def append_response(self, response, name):
        symbol = self.human_symbol if name == "human" else self.agent_symbol
        self._response_recording += f"{name} says: {response}\n"
        self.chat_box.append_html_text(symbol + response.replace('\n', '<br>') + '<br>')
        self.manager.draw_ui(self.screen)
        self.manager.update(self.clock.tick(100) / 1000.0)
        pygame.display.update()

    def robot_generate_callback(self, text):
        self.append_response(text, 'robot')
        
        self.status_label.set_text("Agent responding...")
    def change_status(self, status_text):
        self.status_label.set_text(status_text)

    def process_command_line_input(self, text):
        """
        Processes command-line text input.
        This function mimics the previous behavior of the submit_text_entry
        method that relied on the GUI text entry box.
        """
        self.ui.append_response(text, 'human')
        self.status_label.set_text("Agent responding...")

    def enable_pause_button(self):
        self.pause_button.enable()

    def disable_pause_button(self):
        self.pause_button.disable()

    def set_ui_to_pause(self):
        self.pause_button.set_text("RESUME")
        self.status_label.set_text("Agent paused")
   

    def set_ui_to_resume(self):
        self.pause_button.set_text("PAUSE")
        self.status_label.set_text("Agent running")
        self._response_recording = ('<b> ---Chat--- </b> <br> ')
        self.chat_box.set_text(self._response_recording)
       


class OvercookedPygame:
    """
    Runs the Overcooked game using Pygame.
    """
    def __init__(self, study_config, agent1, agent2, logger, gameTime=30, user_feedback="do anything"):
        self._running = True
        self._paused = False
        self.logger = logger
        self.env = study_config.base_env
        self.score = 0 
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.agent1 = agent1
        self.agent2 = agent2
        self.start_time = time()
        self.init_time = time()
        self.last_suggestion_time = self.start_time
        self.prev_timestep = 0
        
        # Screen dimensions based on constants and environment grid size
        self.screen_width = INPUT_BOX_WIDTH
        self.screen_height = self.env.mdp.height * 30 + 140 + 50 + INPUT_BOX_HEIGHT + 470
        
        self.ui = OvercookedUI(self.screen_width, self.screen_height)
        self.player_1_action = None
        self.player_2_action = None
        self.prev_score = 0
        

    def on_init(self):
        pygame.init()
        pygame.display.init()
        pygame.display.set_caption('Overcook game interface')
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.screen.fill(COLOR_OVERCOOKED)
        
        # Initialize agents with the environment MDP
        self.agent1.set_mdp(self.env.mdp)
        self.agent2.set_mdp(self.env.mdp)
        self.agent1.name = "agent1"
        self.agent2.name = "agent2"

        # Set up a timer event (1-second interval)
        # pygame.time.set_timer(TIMER, t)

        # Load kitchen configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "data", "config", "kitchen_config.json")
        ds = load_from_json(config_path)
        config_copy = copy.deepcopy(ds["config"])
        self.state_visualizer = StateVisualizer(**config_copy)

        self.logger.env = self.env
        
        # Initialize UI manager and UI elements
        self.manager = pygame_gui.UIManager((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.ui.init_ui(
            user_mode=self.agent2.usermode,
            input_box_width=INPUT_BOX_WIDTH,
            input_box_height=INPUT_BOX_HEIGHT,
            pause_button_width=PAUSE_BUTTON_WIDTH,
            pause_button_height=PAUSE_BUTTON_HEIGHT,
            conversation_box_width=CONVERSATION_BOX_WIDTH,
            manager=self.manager,
            screen=self.screen,
            clock=self.clock
        )
        
        self.agent2.initilize_Graph(self.env.state)
        self.env.mdp.step_environment_effects(self.env.state)
       
        self.on_render()
        self.manager.draw_ui(self.screen)
        self.manager.update(self.clock.tick(100) / 1000.0)
        
        # # Set UI enabled state based on user mode
        if self.agent2.usermode == 1:
            self.ui.disable_pause_button()
        else:
            self.ui.enable_pause_button()
           

        return True
    
    

        

                    


    def on_loop(self, action_fps=3):
        while self._running:
            sleep(0.1)
            self.logger.env = self.env
            time_step = round((time() - self.init_time) * action_fps)
            if time_step > self.prev_timestep and not self._time_up() and (not self._paused) and (self.player_2_action is None):
                self.prev_timestep = time_step
                self.agent2.subtasking(self.env.state, self.ui)
                self.player_2_action, _ = self.agent2.action(self.env.state)

    def on_render(self):
        # Prepare HUD data
        def hud_data(state, **kwargs):
            data = {
                "time left": (self.max_time - round(time() - self.start_time)
                              if not self._paused else self.max_time - round(self.paused_at - self.start_time)),
                "all_orders": [r.to_dict() for r in state.all_orders],
                "bonus_orders": [r.to_dict() for r in state.bonus_orders],
                "score": self.score,
            }
            data.update(copy.deepcopy(kwargs))
            return data

        grid_x_num = len(self.env.mdp.terrain_mtx[0])
        grid_y_num = len(self.env.mdp.terrain_mtx)
        kitchen = self.state_visualizer.render_state(
            self.env.state, self.env.mdp.terrain_mtx, hud_data=hud_data(self.env.state)
        )
        self.screen.blit(kitchen, (0, 0))

        # Draw grid numbers for reference
        for i in range(grid_x_num):
            font = pygame.font.SysFont('fira_code', 22)
            text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(text_surface, (i * 30 + 10, self.env.mdp.height * 30 + 140 + 10))
        for i in range(grid_y_num):
            font = pygame.font.SysFont('fira_code', 22)
            text_surface = font.render(str(i), False, WHITE)
            self.screen.blit(text_surface, (self.env.mdp.width * 30 + 10, i * 30 + 140 + 10))

        # Draw node graph beside the game grid
        self.agent2.dm.node_graph.draw_graph_cairo('graph.png')
        node_graph_surface = pygame.image.load('graph.png')
        self.screen.blit(node_graph_surface, (0, self.env.mdp.height * 30 + 140 + 50))
        

    def on_cleanup(self):
        self.logger.save_log_as_pickle()
        if self.logger.video_record:
            self.logger.create_video()
        pygame.quit()

    def on_execute(self):
        if not self.on_init():
            self._running = False

        # Start a separate thread for the agent loop
        agent_thread = threading.Thread(target=self.on_loop)
        agent_thread.start()
        
        while self._running:
            if self._time_up():
                self._running = False


            for event in pygame.event.get():
                self.manager.process_events(event)
               
                # Handle timer event for environment effects and agent suggestions
                if event.type == TIMER:
                    self.env.mdp.step_environment_effects(self.env.state)
                    if self.agent2.usermode in (3, 4):
                        if (time() - self.last_suggestion_time > 30) and (not self._paused):
                            self._pause_game()
                            state = self.env.state
                            current_agent_state = state.players[1].to_dict()
                            other_agent_state = state.players[0].to_dict()
                            world_state = state.to_dict().pop("objects")
                            grid = self.env.mdp.terrain_mtx
                            from llm.utils import read_from_file
                            prompt_layout = read_from_file("llm/layout/HRT/HRT_active_suggestion.txt")
                            prompt = self.agent2.format_active_suggestion_prompt(
                                prompt_layout, world_state, current_agent_state, other_agent_state,
                                grid=grid, order_list=self.agent2.order_list
                            )
                            response = self.agent2.dm.active_query(prompt)
                            agent_text = response.suggestions
                            self.ui.robot_generate_callback(agent_text)
                            self.agent2.dm.robot_message(agent_text)

                # Handle key press events for movement and interaction
                if event.type == pygame.KEYDOWN:
                    key = event.dict.get('key')
                    action = Action.STAY
                    if key == pygame.K_UP:
                        action = Direction.NORTH
                    elif key == pygame.K_RIGHT:
                        action = Direction.EAST
                    elif key == pygame.K_DOWN:
                        action = Direction.SOUTH
                    elif key == pygame.K_LEFT:
                        action = Direction.WEST
                    elif key == pygame.K_SPACE:
                        action = Action.INTERACT

                    if action in Action.ALL_ACTIONS:
                        self.player_1_action = action
                        self.agent2.record_human_log(
                            self.env.state.players[0],
                            self.env.state.to_dict().pop("objects"),
                            self.player_1_action
                        )

                # Quit the game if requested
                if event.type == pygame.QUIT:
                    self._running = False

                # Handle UI events (pause/resume button and text entry)
                if event.type == pygame_gui.UI_BUTTON_PRESSED:
                    if getattr(event, 'ui_element', None) == self.ui.pause_button and self.agent2.usermode != 1:
                        if not self._paused:
                            # print("Paused")
                            self._pause_game()
                        else:
                            self._resume_game()

            
            # handle agent actions
            if not self._paused:
                if self.player_2_action:
                    if self.player_1_action:
                        joint_action = (self.player_1_action, self.player_2_action)
                        done = self._agents_step_env(*joint_action)
                        self.player_1_action, self.player_2_action = None, None
                    else:
                        joint_action = (Action.STAY, self.player_2_action)
                        done = self._agents_step_env(*joint_action)
                        self.player_2_action = None
                elif self.player_1_action:
                    joint_action = (self.player_1_action, Action.STAY)
                    done = self._agents_step_env(*joint_action)
                    self.player_1_action = None
                else:
                    joint_action = (Action.STAY, Action.STAY)
                    done = self._agents_step_env(*joint_action)

                log = {"state": self.env.state.to_dict(), "joint_action": joint_action, "score": self.score}
                self.logger.episode.append(log)
                if done:
                    self._running = False
            
            
            
            self.manager.update(self.clock.tick(100) / 1000.0)
            
            
            self.on_render()
            self.manager.draw_ui(self.screen)
            
            pygame.display.update()
            if self.logger.video_record:
                frame_name = self.logger.img_name(time() - self.init_time)
                pygame.image.save(self.screen, frame_name)
            
        print("Game Over!")
        agent_thread.join()
        pygame.quit()
        self.on_cleanup()
        sys.exit()

    def _time_up(self):
        return time() - self.start_time > self.max_time

    def _agents_step_env(self, agent1_action, agent2_action):
        joint_action = (agent1_action, agent2_action)
        prev_state = self.env.state
        self.state, info = self.env.mdp.get_state_transition(prev_state, joint_action)
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        next_state, timestep_sparse_reward, done, info = self.env.step(
            joint_action, joint_agent_action_info=[{"1"}, {"2"}]
        )
        return done

        
            

    def _pause_game(self):
        self._paused = True
        self.ui.set_ui_to_pause()
        self.agent2.dm.init_dialogue()
        self.paused_at = time()
        # self.ui.text_entry_box.focus()
        
        print("Game is paused!")
        print("\n=== Cordination Dialogue (type 'quit' to end) ===")
        while True:
            user_input = input("USER >> ")
            if user_input.lower() in ["quit", "exit"]:
                print("Ending dialogue.")
                break
            self.ui.append_response(user_input, 'human')
            self.ui.status_label.set_text("Agent responding...")
            self.agent2.dm.receive_message(user_input)
            message_to_human = self.agent2.dm.process_conversation()
            self.ui.robot_generate_callback(message_to_human)

    def _resume_game(self):
        self.last_suggestion_time = time()
        self._paused = False
        self.ui.set_ui_to_resume()
        self.agent2.dm.clear_dialog()
        paused_duration = time() - self.paused_at
        self.start_time += paused_duration
        print("Game is resumed!")
