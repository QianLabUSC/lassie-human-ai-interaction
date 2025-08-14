# HRT-PR-OVERCOOKED 
<p align="center">
  <img src="./assets/fig1.png" width="100%"> 
  <i>HRT-PR interface based on Overcooked AI environment</i>
</p>

## Introduction

**Effective human-robot collaboration requires robots to adapt their roles and levels of support based on human needs, task requirements, and complexity.** We propose a Human-Robot Teaming Framework with Proactive Reactive Feedback (HRT-PR), designed to enhance human-robot interaction through dynamic adjustment of feedback frequency and content.

**Leveraging the strong communication capabilities of Large Language Models (LLMs) as foundation models**, our framework implements a dual-module architecture with a **DAG (Directed Acyclic Graph)** structure: a **Coordinator** that provides high-level, low-frequency strategic guidance, and a **Manager** that delivers subtask-specific, high-frequency instructions. This design enables both passive and active interaction modes, allowing robotic agents to seamlessly transition between supportive and directive roles based on real-time assessment of human needs and task demands.

## Environment

We built our system on top of the original [Overcooked AI](https://github.com/HumanCompatibleAI/overcooked_ai) environment, which provides an ideal testbed for studying human-robot collaboration as it requires coordination, task division, and real-time communication—key elements of effective teaming.

<p align="center">
  <img src="./assets/fig8.png" width="80%">
  <br><i>Different game layouts used in our experiments</i>
</p>

## Agent Modes

Our framework supports three different agent interaction styles:

<p align="center">
  <img src="./assets/fig2.png" width="80%">
  <br><i>Overview of different agent modes and their characteristics</i>
</p>

- **Default**: A greedy agent collaborates with humans without any communication.
- **Passive leader-follower style**: Agent receives and acts upon instructions based on human preferences.
- **Active peer-to-peer style**: Besides receiving and acting upon instructions, it also provides feedback based on human past behavior.

## Demo Video

https://www.dropbox.com/scl/fi/kzhhb9cj1z5dh8ye8muhh/supplenmentary_video.mp4?rlkey=c6btki1u5c0t859etuqwrxxto&st=e2mvawbi&raw=1




*Note: The human player controls the blue hat agent, while the AI agent (red hat) adapts its communication frequency and strategy based on the HRT-PR framework.*

## Survey Design

<p align="center">
  <img src="./assets/fig3.png" width="80%">
  <br><i>Survey design and participant feedback collection methodology</i>
</p>

## Results

<p align="center">
  <img src="./assets/fig7.png" width="80%">
  <br><i>Experimental results showing the relationship between task complexity and optimal communication frequency</i>
</p>

<p align="center">
  <img src="./assets/fig5.png" width="80%">
  <br><i>Detailed analysis of performance metrics across different agent modes and task complexities</i>
</p>

Our results reveal a nuanced relationship between task complexity, human capabilities, and optimal robot communication strategies. As task complexity increases relative to human capabilities, human teammates demonstrate a stronger preference for robots that offer frequent, proactive support. However, we identify a critical threshold: when task complexities exceed the LLM's capacity, superactive robotic agents can generate noisy and inaccurate feedback that hinders team performance.

## Principles

Based on our experimental results, we identify four key cases that determine optimal robot communication strategies:

<p align="center">
  <img src="./assets/princinple.png" width="80%">
  <br><i>Four cases of task complexity (Th) vs. agent capability (Cl) and human capability (Ch)</i>
</p>

**Case 1: Low Task Complexity, High Capabilities (Th < Cl, Th < Ch)**
- Task is manageable for both human and agent
- Minimal communication needed
- Passive or default agent mode optimal

**Case 2: High Task Complexity, High Capabilities (Th > Cl, Th < Ch)**
- Task exceeds agent capability but human can handle
- Agent should provide supportive communication
- Passive leader-follower style recommended

**Case 3: Low Task Complexity, Low Capabilities (Th < Cl, Th > Ch)**
- Task exceeds human capability but agent can handle
- Agent should take more active role
- Active peer-to-peer style optimal

**Case 4: High Task Complexity, Low Capabilities (Th > Cl, Th > Ch)**
- Task exceeds both human and agent capabilities
- Communication may become noisy and counterproductive
- Need to balance support without overwhelming

## Installation
Create a Conda environment with Python 3.10 

``` 
conda create -n overcooked_ai python=3.10
conda activate overcooked_ai
```

**Note:** We use the original Overcooked AI as our test environment. You need to specify numpy version 1.26.4 in their setup.py file.

Clone the repo recursively
```
git clone --recursive git@github.com:QianLabUSC/lassie-human-ai-interaction.git
```
clone the overcooked ai environment 
```
cd lassie-human-ai-interaction
git clone --recurse-submodules https://github.com/HumanCompatibleAI/overcooked_ai.git
git submodule init
git submodule update
```

To use overcooked ai environment
```
cd overcooked_ai
pip install -e .
```
Install required dependency
```
cd lassie-human-ai-interaction
pip install -r requirements.txt
```

To use Open AI API, create your own API key and create an .env file located in the overcooked_llm folder contains:
```
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```


## Testing out ChatOverCooked Interface

### LLM Model Selection

Our framework supports multiple LLM models for both the Coordinator and Subtask Manager components. You can specify different models using the following command-line arguments:

**Available Models:**
- `gpt` - OpenAI GPT-4 (default)
- `gpt_mini` - OpenAI GPT-4o Mini
- `llama` - Llama 3.1 via Ollama
- `gemma` - Gemma 2B via Ollama  
- `llava` - Llava 7B via Ollama
- `sglang` - Local Llama 3 8B with SGLang
- `rule` - Rule-based system

**Usage Examples:**
```bash
# Use GPT for both coordinator and subtask manager
python main.py --coordinator_model gpt --subtask_manager_model gpt

# Use Llama for coordinator and GPT for subtask manager
python main.py --coordinator_model llama --subtask_manager_model gpt

# Use different models for different components
python main.py --coordinator_model gpt_mini --subtask_manager_model llama
```

**Basic Usage:**
```bash
cd src/
python main.py
```

You can always change the arguments in the commands to experiment with different model combinations.

the user study log and video will be saved under overcooked_llm/src/user_study

## Game Introduction 
The game is between a LLM-driven agent with a human player. Similiar to Original overcooked ai, we added a chat interface where human will be able to chat with another agent and change agent behavior based on their preferences. An example instruction would be: "you should only make onion soup, I will serve dishes". 
### Agent styles: 
There are three type of agents: active Supervisor, Passive mentee, and default receptive agent. 

- Default: a greedy agent collaborates with humans without any communication.
- Passive leader-follower style: agent receives and acts upon instructions based on human preferences.
- Active peer-to-peer style: beside receive and acts upon instructions. it also provide feedback based on human past behavior.
  
### Interface: 
You will be control blue hat agent. use up/down/left/right key to move and space bar to interact with environment. 
in case of communicate with another agent, click on pause button, then type in your instruction in the text input box. afterwards, agent will be updated once the game is resumed. chat session is shown on the right. Vertical coordinates indicating row(x) numbers and  horizontal coordinates indicating column(y) numbers. 


## Code Structure Overview

`src` contains:

`data/`:

- `config/`: contains Overcooked game initial configuration files
- `layout/`: contains different layout files

`llm/`:

- `task/`: LLM prompts. des_1_layout.txt contains the prompt that is actually being fed to LLM.
- `llm_agent.py`: customized greedy human model. injected with LLM modified code.
- `llm_api.py`: LLM API calls with text/code extraction. with code reflection implementation.
- `utils.py`: LLM prompt tips.
  
`mdp/`:

- `llm_overcooked_mdp.py`: LLM agent related code inheritance. added human action history logging for active agent. 

`user_study/`:

- `log/`: contains folders of logs/video for each participants.
  
`planning/`:

- `planners.py`: customized MediumLevelActionManager for LLM. enable more actions to use.

`main.py`: script for start a chatovercook pygame session.

