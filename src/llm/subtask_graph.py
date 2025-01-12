import sys
import os
import uuid
import json
import networkx as nx
import matplotlib.pyplot as plt
import threading
import matplotlib.patches as mpatches
class SubTask:
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"
    READY_TO_EXECUTE = "ready_to_execute"
    NOT_READY = "not_ready"
    PUTTING = "putting"
    GETTING = "getting"
    OPERATING = "operating"

    def __init__(self, name: str, target_position: tuple, task_type: str):
        self.id = str(uuid.uuid4())
        self.name = name  # Name of the subtask
        self.target_position = target_position  # Target position for the subtask
        self.task_type = task_type  # Type of task: PUTTING, GETTING, or OPERATING
        self.parent_subtask = None  # Reference to the parent subtask
        self.next_subtask = None  # Reference to the next subtask
        self.status = SubTask.UNKNOWN  # Status of the subtask
        self.cost = 1 # initialize cost as 1 

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
    def __init__(self):
        self.vertex = []
        self.edge = []
        self.G = nx.DiGraph()

    def add_node(self, node: SubTask):
        if node not in self.vertex:
            self.vertex.append(node)

    def add_edge(self, start_node: SubTask, end_node: SubTask):
        if (start_node, end_node) not in self.edge:
            self.edge.append((start_node, end_node))

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

        for edge in self.edge:
            self.G.add_edge(edge[0].name, edge[1].name)

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
        nx.draw_networkx_edges(self.G, pos, edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)
        # nx.draw_networkx_edges(self.G, pos, edge_color='black')  # Edges are black
        node_labels = {node.name: node.name for node in self.vertex}
        # Move text to the bottom of the shape
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


if __name__ == "__main__":
    # TEST
    graph = Graph()

    subtask1 = SubTask("Get onion 1", (1, 1), SubTask.PUTTING)
    subtask1.status = SubTask.SUCCESS
    subtask2 = SubTask("Get onion 2", (1, 1), SubTask.PUTTING)
    subtask2.status = SubTask.SUCCESS
    subtask3 = SubTask("Get onion 3", (1, 1), SubTask.PUTTING)
    subtask3.status = SubTask.SUCCESS


    subtask4 = SubTask("Put onion 1", (2, 2), SubTask.GETTING)
    subtask4.status = SubTask.FAILURE
    subtask5 = SubTask("Put onion 2", (3, 3), SubTask.GETTING)
    subtask5.status = SubTask.READY_TO_EXECUTE
    subtask6 = SubTask("Put onion 3", (3, 3), SubTask.GETTING)
    subtask6.status = SubTask.READY_TO_EXECUTE

    subtask7 = SubTask("Cook", (2, 2), SubTask.OPERATING)
    subtask7.status = SubTask.NOT_READY
    subtask8 = SubTask("Get soup", (3, 3), SubTask.GETTING)
    subtask8.status = SubTask.NOT_READY
    subtask9 = SubTask("Put soup", (3, 3), SubTask.PUTTING)
    subtask9.status = SubTask.NOT_READY

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
