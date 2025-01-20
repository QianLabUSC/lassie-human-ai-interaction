import sys
import os
import uuid
import json
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np
import threading
import matplotlib.patches as mpatches
class SubTask:
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    READY_TO_EXECUTE = "ready_to_execute"
    NOT_READY = "not_ready"
    EXECUTING = "executing"
    PUTTING = "putting"
    GETTING = "getting"
    OPERATING = "operating"
    TASK_TYPE_MAP = {
        0: PUTTING,
        1: GETTING,
        2: OPERATING  # Mapping "COOKING" to "OPERATING"
    }

    TASK_STATUS_MAP = {
        0: UNKNOWN,
        1: READY_TO_EXECUTE,
        2: SUCCESS,
        3: FAILURE,
        4: NOT_READY
    }

    def __init__(self, id: int, name: str, target_position: list, task_type: int, task_status: int):
        self.id = id
        self.name = name  # Name of the subtask
        self.target_position = target_position  # Target position for the subtask
        self.task_type = SubTask.TASK_TYPE_MAP.get(task_type, SubTask.UNKNOWN)  # Type of task: PUTTING, GETTING, or OPERATING
        self.parent_subtask = None  # Reference to the parent subtask
        self.next_subtask = None  # Reference to the next subtask
        self.status = SubTask.TASK_STATUS_MAP.get(task_status, SubTask.UNKNOWN)
        self.cost = 1 # initialize cost as 1 
        self.agent_id = None
        self.running_time = None

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "target_position": self.target_position,
            "task_type": self.task_type,
            "parent_subtask": self.parent_subtask.name if self.parent_subtask else None,
            "next_subtask": self.next_subtask.name if self.next_subtask else None,
            "status": self.status,
        }


class Graph:
    def __init__(self, mlam):
        self.vertex = []
        self.edge = []
        self.G = nx.DiGraph()
        self.mlam = mlam

    def add_node(self, node: SubTask):
        if node not in self.vertex:
            self.vertex.append(node)

    def add_edge(self, start_node: SubTask, end_node: SubTask):
        if (start_node, end_node) not in self.edge:
            self.edge.append((start_node, end_node, 1))

    def get_node_by_id(self, node_id: str) -> SubTask:
            """Retrieve a node by its ID."""
            for node in self.vertex:
                if node.id == node_id:
                    return node
            return None  # Return None if no node with the given ID is found


    def assign_task(self, node_id, agent_id):
        node = self.get_node_by_id(node_id)
        node.agent_id = agent_id
        node.running_time = 0
        node.status = SubTask.EXECUTING
        return node
    
    def delete_node(self, node: SubTask):
        self.vertex.remove(node)
        self.edge = [edge for edge in self.edge if edge[0] != node and edge[1] != node]

    def to_json(self) -> dict:
        return {
            "vertex": [node.to_json() for node in self.vertex],
            "edge": [(edge[0].name, edge[1].name) for edge in self.edge],
        }

    def draw_graph(self, path):
        self.G.clear()

        # Add nodes to the graph
        for node in self.vertex:
            # Determine node shape based on task type
            if node.task_type == SubTask.PUTTING:
                node_shape = 's'  # Square for putting
            elif node.task_type == SubTask.GETTING:
                node_shape = 'p'  # Diamond for getting
            elif node.task_type == SubTask.OPERATING:
                node_shape = 'o'  # Circle for operating
            else:
                node_shape = 'o'

            # Determine node contour color based on status
            if node.status == SubTask.SUCCESS:
                edge_color = 'green'
            elif node.status == SubTask.FAILURE:
                edge_color = 'red'
            elif node.status == SubTask.READY_TO_EXECUTE:
                edge_color = 'blue'
            elif node.status == SubTask.NOT_READY:
                edge_color = 'orange'
            else:
                edge_color = 'gray'

            self.G.add_node(node.name, description=node.name, shape=node_shape, edge_color=edge_color)

        # Add edges and edge costs to the graph
        edge_labels = {}
        for edge in self.edge:
            start_node, end_node, edge_cost = edge
            self.G.add_edge(start_node.name, end_node.name)
            edge_labels[(start_node.name, end_node.name)] = f"{edge_cost:.2f}"  # Format the cost as a label

        # Positioning the nodes
        try:
            pos = nx.planar_layout(self.G)  # Ensure edges do not intersect
        except nx.NetworkXException:
            pos = nx.spring_layout(self.G)  # Fallback to spring_layout if the graph is not planar

        plt.figure(figsize=(14, 14))  # Make the figure larger to ensure text fits inside shapes

        # Draw nodes with individual shapes and edge colors
        for node in self.vertex:
            nx.draw_networkx_nodes(
                self.G,
                pos,
                nodelist=[node.name],
                node_size=300,
                node_color=self.G.nodes[node.name]['edge_color'],
                edgecolors=self.G.nodes[node.name]['edge_color'],
                node_shape=self.G.nodes[node.name]['shape']
            )

        # Draw edges
        nx.draw_networkx_edges(
            self.G,
            pos,
            edge_color='black',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20
        )

        # Annotate edges with their costs
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=10, font_color='darkred')

        # Draw node labels
        node_labels = {node.name: node.name for node in self.vertex}
        adjusted_labels = {k: f"\n{v}" for k, v in node_labels.items()}
        label_pos = {node: (x, y - 0.05) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(self.G, label_pos, labels=adjusted_labels, font_size=20, verticalalignment='bottom')

        # Add legend for task types and statuses
        legend_elements = [
            mpatches.Patch(facecolor='green', edgecolor='green', label='Success'),
            mpatches.Patch(facecolor='red', edgecolor='red', label='Failure'),
            mpatches.Patch(facecolor='blue', edgecolor='blue', label='Ready to Execute'),
            mpatches.Patch(facecolor='orange', edgecolor='orange', label='Not Ready'),
            mpatches.Patch(facecolor='gray', edgecolor='gray', label='Unknown'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='Shapes:'),
            mpatches.Patch(facecolor='white', label='Square = Putting', edgecolor='black'),
            mpatches.Patch(facecolor='white', label='Diamond = Getting', edgecolor='black'),
            mpatches.Patch(facecolor='white', label='Circle = Operating', edgecolor='black'),
        ]

        plt.legend(handles=legend_elements, loc='upper left', fontsize='small', frameon=True)
        plt.savefig(path)  # Save as PNG image
        plt.close()


    def write_graph_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=4)


    def calculate_distance_to_pos(self, pos_ori, node):
        distance_to_robot = []
        robot_action = []
        best_cost = float('inf')
        best_plan = []
        for position in node.target_position:
            node_pos_pose = self.mlam._get_ml_actions_for_positions([tuple(position)])
            for start_pose in node_pos_pose:
                # Use the motion planner to compute the plan and cost
                motion_planner = self.mlam.joint_motion_planner.motion_planner
                plan, _, cost = motion_planner.get_plan(pos_ori, start_pose)
                # Update the best cost and plan if a lower cost is found
                if cost < best_cost:
                    best_cost = cost
                    best_plan = plan
            distance_to_robot.append(best_cost)
            robot_action.append(best_plan)
        return distance_to_robot, robot_action
    

    def get_state_in_lauguage(self, agent_state, human_state):
        state_description = []
        task_description = ""
        # Iterate over each subtask and generate a description
        for node in self.vertex: 

            if node.status == SubTask.READY_TO_EXECUTE:
                task_description = f"id: {node.id}; description '{node.name}'; type: '{node.task_type}; status: '{node.status}'; target_position: '{node.target_position}';"


                # if node.agent_id:
                #     if node.agent_id == 0:
                #         task_description += " The Subtask is being executed by the human." 
                #     elif node.agent_id == 1:
                #         task_description += " The Subtask is being executed by the robot."
            

                # Add information about the target position

                distance_to_robot, _ = self.calculate_distance_to_pos((agent_state['position'], agent_state['orientation']), node)
                distance_to_human, _ = self.calculate_distance_to_pos((human_state['position'], human_state['orientation']), node)
                task_description += f"Distance to the robot: {distance_to_robot}; Distance to the human: {distance_to_human};"
                # Add information about the parent subtask, if any
                if node.parent_subtask:
                    task_description += f" Prerequiste task id: '{node.parent_subtask}';"
                else:
                    task_description += " This task has no prerequiste task. \n"


            

                # Append to the state description list
                state_description.append(task_description)


            human_executing, human_executing_id = self.check_executing_by_agent_id(0)
            robot_executing, robot_executing_id = self.check_executing_by_agent_id(1)
            if human_executing:
                task_description += f"The human is executing subtask, {human_executing_id}."
            else:
                task_description += "The human is not executing any subtask."
            if robot_executing:
                task_description += f"The robot is executing a subtask, {robot_executing_id}."
            else:
                task_description += " The robot is not executing any subtask."

        # Return the full description
        return "\n".join(state_description)



    def checking_time_out_fail(self, agent_id, agent_step):
        _, id = self.check_executing_by_agent_id(agent_id)
        subtask_node = self.get_node_by_id(id)

        # if subtask_node.running_time > 2 * np.max(subtask_node.cost):
        #     subtask_node.status = SubTask.FAILURE
        #     subtask_node.agent_id = None
        #     subtask_node.running_time = None
        #     return True
        # else:
        #     if agent_step: 
        #         subtask_node.running_time += 1
        #     return False
        return False
        

    def check_executing_by_agent_id(self, agent_id):
        # check if any subtask is executing by the agent id
        for node in self.vertex:
            if node.agent_id == agent_id and node.status == SubTask.EXECUTING:
                return True, node.id
        return False, None
    
    def update_status_by_agent_id(self, agent_id, if_success):
        existing, id = self.check_executing_by_agent_id(agent_id)
        if existing:
            subtask_node = self.get_node_by_id(id)
            if if_success:
                subtask_node.status = SubTask.SUCCESS
            return True
        else:
            return False

    def generate_graph_from_subtask_objects(self, graph, pos_list):

        # Add nodes to the graph
        pos_list = np.array(pos_list)
        for task in graph.subtasks:
            current_node = SubTask(task.id,
                                    name=f"{task.name}-{task.id}", 
                                    target_position=pos_list[task.target_position_id], 
                                    task_type=task.task_type, 
                                    task_status=task.task_status)
            self.add_node(current_node)
            if task.parent_subtask:
                current_node.parent_subtask = task.parent_subtask
                for parent_id in task.parent_subtask:
                    parent_node = self.get_node_by_id(parent_id)
                    
                    if parent_node:
                        self.add_edge(parent_node, current_node)  # Edge from parent to current subtask
                    else:
                        import warnings
                        warnings.warn(
                            f"Parent node is None for current node '{current_node.name}' (ID: {current_node.id}). No edge added.",
                            RuntimeWarning)
        # compute initial status
        self.update_node_status()
        self.compute_edge_cost()

        self.draw_graph("init_graph.png")

    def update_node_status(self):
        """
        Update the status of each node in the graph based on the following rules:
        - If the node's status is already "SUCCESS", it remains "SUCCESS".
        - If the node's status is "UNKNOWN" and:
            - The node has no parent tasks, it is "READY_TO_EXECUTE".
            - The node has parent tasks:
                - If any parent task is not "SUCCESS", the node is "NOT READY".
                - If all parent tasks are "SUCCESS", the node is "READY_TO_EXECUTE".
        """
        for node in self.vertex:
            # If the node is already "SUCCESS", keep it
            if node.status == SubTask.SUCCESS:
                continue

            # If the node has no parent tasks, it is "READY_TO_EXECUTE"
            if not node.parent_subtask:
                node.status = SubTask.READY_TO_EXECUTE
            else:
                for i in node.parent_subtask:
                    parent = self.get_node_by_id(i)
                    node.status = SubTask.READY_TO_EXECUTE
                    if parent.status != SubTask.SUCCESS:
                        node.status = SubTask.NOT_READY
                        break
        self.draw_graph("init_graph.png")
           

    def compute_edge_cost(self):
        for i, edge in enumerate(self.edge):
            start_node, end_node, cost = edge  # Unpack the start and end nodes

            # Initialize the best cost and plan
            best_cost = float('inf')
            best_plan = []
        
            # Iterate through target locations of the start and end nodes
            for start_position in start_node.target_position:
                for end_position in end_node.target_position:
                    # Convert each location to a tuple
                    start_position_tuple = tuple(start_position)
                    end_position_tuple = tuple(end_position)

                    # Call the function to get poses
                    start_node_pos_pose = self.mlam._get_ml_actions_for_positions([start_position_tuple])
                    end_node_pos_pose = self.mlam._get_ml_actions_for_positions([end_position_tuple])
                    for start_pose in start_node_pos_pose:
                        for end_pose in end_node_pos_pose:
                            # Use the motion planner to compute the plan and cost
                            motion_planner = self.mlam.joint_motion_planner.motion_planner
                            plan, _, cost = motion_planner.get_plan(start_pose, end_pose)

                            # Update the best cost and plan if a lower cost is found
                            if cost < best_cost:
                                best_cost = cost
                        
            self.edge[i] = (start_node, end_node, best_cost)  # Update the edge with the best cost




if __name__ == "__main__":
    # TEST
    graph = Graph()

    subtask1 = SubTask(0, "Get onion", (1, 1), 0, 1)

    subtask2 = SubTask(1, "Get onion", (1, 1), 0, 1)

    subtask3 = SubTask(2, "Get onion", (1, 1), 0, 1)



    subtask4 = SubTask(3, "Put onion", (2, 2), 1, 1)
 
    subtask5 = SubTask(4, "Put onion", (3, 3), 1, 1)
 
    subtask6 = SubTask(5, "Put onion", (3, 3), 1, 1)
   

    subtask7 = SubTask(6, "Cook", (2, 2), 2, 1)

    subtask8 = SubTask(7, "Get soup", (3, 3), 0, 1)
  
    subtask9 = SubTask(8, "Put soup", (3, 3), 1, 1)
  

    graph.add_node(subtask1)
    graph.add_node(subtask2)
    graph.add_node(subtask3)
    graph.add_node(subtask4)
    graph.add_node(subtask5)
    graph.add_node(subtask6)
    graph.add_node(subtask7)
    graph.add_node(subtask8)
    graph.add_node(subtask9)

    graph.add_edge(subtask1, subtask4)
    graph.add_edge(subtask2, subtask5)
    graph.add_edge(subtask3, subtask6)
    graph.add_edge(subtask4, subtask7)
    graph.add_edge(subtask5, subtask7)
    graph.add_edge(subtask6, subtask7)
    graph.add_edge(subtask7, subtask8)
    graph.add_edge(subtask8, subtask9)

    graph.write_graph_to_json("graph.json")
    graph.draw_graph("graph.png")
