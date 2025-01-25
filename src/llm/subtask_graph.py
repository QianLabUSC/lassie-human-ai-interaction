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
from PIL import Image  # For handling the returned image
import io

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
        self.task_type = SubTask.TASK_TYPE_MAP.get(task_type, SubTask.UNKNOWN)
        self.parent_subtasks = []  # List[SubTask] references
        self.next_subtasks = []    # List[SubTask] references
        self.status = SubTask.TASK_STATUS_MAP.get(task_status, SubTask.UNKNOWN)
        self.cost = 1  # initialize cost as 1 
        self.agent_id = None
        self.running_time = None

    def to_json(self) -> dict:
        """
        Return a dict representation. 
        parent_subtasks and next_subtasks are turned into lists of subtask IDs.
        """
        return {
            "id": self.id,
            "name": self.name,
            "target_position": self.target_position,
            "task_type": self.task_type,
            "parent_subtasks": [p.id for p in self.parent_subtasks],
            "next_subtasks": [child.id for child in self.next_subtasks],
            "status": self.status
        }


class Graph:
    def __init__(self, mlam=None):
        """
        If you don't need `mlam` for tests, you can default it to None.
        """
        self.vertex = []  # List of SubTask
        self.edge = []    # list of (start_node, end_node, cost)
        self.G = nx.DiGraph()
        self.mlam = mlam

    def add_node(self, node: SubTask):
        if node not in self.vertex:
            self.vertex.append(node)

    def add_edge(self, start_node: SubTask, end_node: SubTask):
        # If the edge isn't in self.edge, add it
        if (start_node, end_node) not in [(e[0], e[1]) for e in self.edge]:
            # The default cost is 1, updated later by compute_edge_cost()
            self.edge.append((start_node, end_node, 1))
            # Also reflect the adjacency in the subtask references
            if end_node not in start_node.next_subtasks:
                start_node.next_subtasks.append(end_node)

    def get_node_by_id(self, node_id: int) -> SubTask:
        """Retrieve a node by its integer ID."""
        for node in self.vertex:
            if node.id == node_id:
                return node
        return None

    def assign_task(self, node_id, agent_id):
        node = self.get_node_by_id(node_id)
        node.agent_id = agent_id
        node.running_time = 0
        node.status = SubTask.EXECUTING
        return node
    
    def delete_node(self, node: SubTask):
        # Remove from self.vertex
        if node in self.vertex:
            self.vertex.remove(node)
        # Remove edges referencing node
        new_edges = []
        for (start, end, cost) in self.edge:
            if start != node and end != node:
                new_edges.append((start, end, cost))
        self.edge = new_edges
        # Also remove from parent_subtasks / next_subtasks if it appears
        for other in self.vertex:
            if node in other.parent_subtasks:
                other.parent_subtasks.remove(node)
            if node in other.next_subtasks:
                other.next_subtasks.remove(node)

    def to_json(self) -> dict:
        """
        Return a dict with:
          "vertex": [ {...}, {...} ]  # each is node.to_json()
          "edge":   [ (start_node_name, end_node_name), ... ]
        """
        return {
            "vertex": [node.to_json() for node in self.vertex],
            "edge": [(edge[0].name, edge[1].name) for edge in self.edge],
        }

    def write_graph_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=4)

    def draw_graph(self, path = "graph.png"):
        self.G.clear()

        # Add nodes to the graph
        for node in self.vertex:
            # Determine node shape
            if node.task_type == SubTask.PUTTING:
                node_shape = 's'  # Square
            elif node.task_type == SubTask.GETTING:
                node_shape = 'p'  # Diamond
            elif node.task_type == SubTask.OPERATING:
                node_shape = 'o'  # Circle
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

            # Actually add the node to the Nx graph
            self.G.add_node(node.name, description=node.name, shape=node_shape, edge_color=edge_color)

        # Add edges and edge costs
        edge_labels = {}
        for (start_node, end_node, edge_cost) in self.edge:
            self.G.add_edge(start_node.name, end_node.name)
            edge_labels[(start_node.name, end_node.name)] = f"{edge_cost:.2f}"

        # Try a planar layout, fallback to spring if not planar
        try:
            pos = nx.planar_layout(self.G)
        except nx.NetworkXException:
            pos = nx.spring_layout(self.G)

        plt.figure(figsize=(14, 14))

        # Draw each node in Nx
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
        nx.draw_networkx_edges(self.G, pos, edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)
        # Edge labels
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=10, font_color='darkred')
        # Node labels
        node_labels = {node.name: node.name for node in self.vertex}
        adjusted_labels = {k: f"\n{v}" for k, v in node_labels.items()}
        label_pos = {n: (x, y - 0.05) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(self.G, label_pos, labels=adjusted_labels, font_size=20, verticalalignment='bottom')

        # Legend
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
        plt.savefig(path)
        buf = io.BytesIO()
        
        plt.savefig(buf, format='png')
        buf.seek(0)

        plt.close()
        return Image.open(buf)

    def calculate_distance_to_pos(self, pos_ori, node: SubTask):
        """
        Example method using your motion planner to compute cost from pos_ori to each position in node.target_position.
        Returns: (distance_to_robot: List[float], robot_action: List[any])
        """
        distance_to_robot = []
        robot_action = []
        best_cost = float('inf')
        best_plan = []

        for position in node.target_position:
            if not self.mlam:
                # If self.mlam isn't set, just do a dummy cost
                distance_to_robot.append(0)
                robot_action.append([])
                continue

            node_pos_pose = self.mlam._get_ml_actions_for_positions([tuple(position)])
            for start_pose in node_pos_pose:
                motion_planner = self.mlam.joint_motion_planner.motion_planner
                plan, _, cost = motion_planner.get_plan(pos_ori, start_pose)
                if cost < best_cost:
                    best_cost = cost
                    best_plan = plan
            distance_to_robot.append(best_cost)
            robot_action.append(best_plan)
        return distance_to_robot, robot_action

    def get_state_in_lauguage(self, agent_state, human_state):
        """
        Return a textual description of each READY_TO_EXECUTE node, plus
        info about who is executing what subtask.
        """
        lines = []
        for node in self.vertex:
            if node.status == SubTask.READY_TO_EXECUTE:
                desc = (f"id: {node.id}; "
                        f"description: '{node.name}'; "
                        f"type: '{node.task_type}'; "
                        f"status: '{node.status}'; "
                        f"target_position: {node.target_position}; ")
                dist_robot, _ = self.calculate_distance_to_pos((agent_state['position'], agent_state['orientation']), node)
                dist_human, _ = self.calculate_distance_to_pos((human_state['position'], human_state['orientation']), node)
                desc += f"Distance to the robot: {dist_robot}; Distance to the human: {dist_human};"
                if not node.parent_subtasks:
                    desc += " no prerequiste tasks. "
                else:
                    parent_ids = [p.id for p in node.parent_subtasks]
                    desc += f" parent subtask(s) = {parent_ids}; "
                lines.append(desc)

        # Who is executing subtask
        human_executing, human_sub_id = self.check_executing_by_agent_id(0)
        robot_executing, robot_sub_id = self.check_executing_by_agent_id(1)

        if human_executing:
            lines.append(f"The human is executing subtask {human_sub_id}.")
        else:
            lines.append("The human is not executing any subtask.")

        if robot_executing:
            lines.append(f"The robot is executing subtask {robot_sub_id}.")
        else:
            lines.append("The robot is not executing any subtask.")

        return "\n".join(lines)

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
        """
        Return (True, subtask_id) if agent_id is assigned to a subtask, else (False, None)
        """
        for node in self.vertex:
            if node.agent_id == agent_id and node.status == SubTask.EXECUTING:
                return True, node.id
        return False, None

    def update_status_by_agent_id(self, agent_id, if_success):
        """
        If the agent is executing a subtask, mark it success or leave alone.
        """
        exists, sub_id = self.check_executing_by_agent_id(agent_id)
        if exists:
            subtask_node = self.get_node_by_id(sub_id)
            if if_success and subtask_node:
                subtask_node.status = SubTask.SUCCESS
            return True
        return False

    def generate_graph_from_subtask_objects(self, graph, pos_list):
        """
        `graph` is presumably an object with `.subtasks` and each subtask has 
         id, name, target_position_id, task_type, task_status, parent_subtask.
        `pos_list` is a numpy array or list of all possible positions, 
        so we can index into them by subtask's target_position_id.
        """
        pos_list = np.array(pos_list)

        for task in graph.subtasks:
            # Create a new SubTask for each
            current_node = SubTask(
                id=task.id,
                name=f"{task.name}-{task.id}",
                target_position=pos_list[task.target_position_id], 
                task_type=task.task_type,
                task_status=task.task_status
            )
            self.add_node(current_node)

        # Now set up parent references & edges
        for task in graph.subtasks:
            current_node = self.get_node_by_id(task.id)
            if current_node is None:
                continue
            if task.parent_subtask:
                for parent_id in task.parent_subtask:
                    parent_node = self.get_node_by_id(parent_id)
                    if parent_node:
                        # set references
                        current_node.parent_subtasks.append(parent_node)
                        self.add_edge(parent_node, current_node)
                    else:
                        import warnings
                        warnings.warn(
                            f"Parent node ID={parent_id} not found for current node '{current_node.name}' (ID: {current_node.id}).",
                            RuntimeWarning
                        )

        # Once all edges are built, compute initial status & cost
        self.update_node_status()
        self.compute_edge_cost()
        self.draw_graph("init_graph.png")

    def update_node_status(self):
        """
        For each node:
         - If node.status=SUCCESS, keep it.
         - Else if node has no parent_subtasks => READY_TO_EXECUTE
         - Else if any parent is not SUCCESS => NOT_READY
         - Else => READY_TO_EXECUTE
        """
        for node in self.vertex:
            if node.status == SubTask.SUCCESS:
                continue
            if not node.parent_subtasks:
                # no parent => can run immediately
                node.status = SubTask.READY_TO_EXECUTE
            else:
                # check if all parents are SUCCESS
                all_parents_success = all(p.status == SubTask.SUCCESS for p in node.parent_subtasks)
                if all_parents_success:
                    node.status = SubTask.READY_TO_EXECUTE
                else:
                    node.status = SubTask.NOT_READY
        # Optionally re-draw after updating
        self.draw_graph("init_graph.png")

    def compute_edge_cost(self):
        """
        For each edge in self.edge, compute the best motion cost from
        start_node's target_position to end_node's target_position.
        """
        for i, (start_node, end_node, cost) in enumerate(self.edge):
            best_cost = float('inf')
            if not self.mlam:
                # No motion planner => leave cost=1 or some default
                self.edge[i] = (start_node, end_node, 1)
                continue

            # For each position in start_node and end_node, find minimal cost
            for start_pos in start_node.target_position:
                for end_pos in end_node.target_position:
                    start_node_pos_pose = self.mlam._get_ml_actions_for_positions([tuple(start_pos)])
                    end_node_pos_pose = self.mlam._get_ml_actions_for_positions([tuple(end_pos)])
                    for start_pose in start_node_pos_pose:
                        for goal_pose in end_node_pos_pose:
                            motion_planner = self.mlam.joint_motion_planner.motion_planner
                            plan, _, c = motion_planner.get_plan(start_pose, goal_pose)
                            if c < best_cost:
                                best_cost = c
            self.edge[i] = (start_node, end_node, best_cost)
            
    def load_from_json(self, graph_data):

        """
        Load the node graph from a JSON dict of the form:
        {
            "vertex": [
            {
                "id": <int>,
                "name": <str>,
                "target_position": [...],
                "task_type": <str>,
                "parent_subtasks": [...],
                "next_subtasks": [...],
                "status": <str>
            },
            ...
            ],
            "edge": [
            [ <start_node_name>, <end_node_name> ],
            ...
            ]
        }
        """
        # 1) Clear existing data
        self.vertex.clear()
        self.edge.clear()

        # 2) Rebuild SubTask objects from the 'vertex' array
        #    We assume you want to keep the same logic for 'task_type' and 'status'
        #    so we must invert the maps or parse them carefully.
        id_to_node = {}

        for node_info in graph_data["vertex"]:
            st_id = node_info["id"]
            st_name = node_info["name"]
            st_positions = node_info["target_position"]   # e.g. list of [x, y] or list-of-lists
            # We might have a *string* for 'task_type'. Convert it back to your numeric code if needed.
            # Or you can store them directly in the SubTask if the constructor now takes strings:
            st_task_type_str = node_info["task_type"]  # e.g. "putting", "getting", ...
            # Convert string back to int using your reversed map:
            inverse_type_map = {
                "putting": 0,
                "getting": 1,
                "operating": 2,
                "unknown": 99   # or however you want to handle it
            }
            st_task_type_num = inverse_type_map.get(st_task_type_str, 99)

            st_status_str = node_info["status"]  # e.g. "ready_to_execute"
            inverse_status_map = {
                "unknown": 0,
                "ready_to_execute": 1,
                "success": 2,
                "failure": 3,
                "not_ready": 4
            }
            st_status_num = inverse_status_map.get(st_status_str, 0)

            # Create the subtask
            new_task = SubTask(
                id=st_id,
                name=st_name,
                target_position=st_positions,
                task_type=st_task_type_num,
                task_status=st_status_num
            )
            self.add_node(new_task)
            id_to_node[st_id] = new_task

        # 3) Rebuild edges from 'edge' array
        #    Here, each entry is [start_node_name, end_node_name].
        #    So we must find the SubTask objects by name or by ID. 
        #    You stored names in your JSON, so let's do lookups by name:
        name_to_node = {node.name: node for node in self.vertex}
        for (start_name, end_name) in graph_data["edge"]:
            start_node = name_to_node.get(start_name)
            end_node = name_to_node.get(end_name)
            if start_node and end_node:
                self.add_edge(start_node, end_node)
        for node_info in graph_data["vertex"]:
            st_id = node_info["id"]
            parents_ids = node_info["parent_subtasks"]  # list of IDs
            next_ids = node_info["next_subtasks"]       # list of IDs
            current_node = id_to_node[st_id]
            for pid in parents_ids:
                parent_node = id_to_node.get(pid)
                if parent_node and parent_node not in current_node.parent_subtasks:
                    current_node.parent_subtasks.append(parent_node)
            for nid in next_ids:
                child_node = id_to_node.get(nid)
                if child_node and child_node not in current_node.next_subtasks:
                    current_node.next_subtasks.append(child_node)
        # self.compute_edge_cost()
        # self.update_node_status()

# ----------------- Simple Test -----------------
if __name__ == "__main__":
    # Minimal test (without a real mlam)
    graph = Graph(mlam=None)

    subtask1 = SubTask(0, "Get onion", [(1, 1)], 0, 1)
    subtask2 = SubTask(1, "Get onion", [(1, 1)], 0, 1)
    subtask3 = SubTask(2, "Get onion", [(1, 1)], 0, 1)
    subtask4 = SubTask(3, "Put onion", [(2, 2)], 1, 1)
    subtask5 = SubTask(4, "Put onion", [(3, 3)], 1, 1)
    subtask6 = SubTask(5, "Put onion", [(3, 3)], 1, 1)
    subtask7 = SubTask(6, "Cook", [(2, 2)], 2, 1)
    subtask8 = SubTask(7, "Get soup", [(3, 3)], 0, 1)
    subtask9 = SubTask(8, "Put soup", [(3, 3)], 1, 1)

    # Add them to the graph
    for s in [subtask1, subtask2, subtask3, subtask4, subtask5, subtask6, subtask7, subtask8, subtask9]:
        graph.add_node(s)

    # Build edges (like your original code)
    graph.add_edge(subtask1, subtask4)
    graph.add_edge(subtask2, subtask5)
    graph.add_edge(subtask3, subtask6)
    graph.add_edge(subtask4, subtask7)
    graph.add_edge(subtask5, subtask7)
    graph.add_edge(subtask6, subtask7)
    graph.add_edge(subtask7, subtask8)
    graph.add_edge(subtask8, subtask9)

    # For demonstration, let's also keep parent_subtasks consistent
    subtask4.parent_subtasks.append(subtask1)
    subtask5.parent_subtasks.append(subtask2)
    subtask6.parent_subtasks.append(subtask3)
    subtask7.parent_subtasks.extend([subtask4, subtask5, subtask6])
    subtask8.parent_subtasks.append(subtask7)
    subtask9.parent_subtasks.append(subtask8)

    graph.write_graph_to_json("graph.json")
    graph.draw_graph("graph.png")

    print("Graph JSON:", json.dumps(graph.to_json(), indent=2))
    print("Done.")
