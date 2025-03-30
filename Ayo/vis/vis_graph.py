import time
import networkx as nx
import matplotlib.pyplot as plt
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node
from Ayo.dags.node_commons import NodeType, NodeIOSchema, NodeAnnotation, NodeOps
from Ayo.engines.engine_types import EngineType
import os

# visualize the DAG with different colors identifying different node types 

def visualize_dag_with_node_types(dag: DAG, output_path: str = None, show: bool = True, left_to_right: bool = False):
    """
    Make a DAG visualization as a graph, with different colors identifying different node types
    
    Args:
        dag: The DAG object to visualize
        output_path: The path to save the image (optional)
        show: Whether to display the image
        left_to_right: Whether to display the graph from left to right
    """
    # Ensure the topological sort is up to date
    dag._ensure_topo_sort()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # The mapping from node types to colors
    node_type_colors = {
        NodeType.INPUT: '#90EE90',  # More vibrant green
        NodeType.COMPUTE: '#87CEFA',  # More vibrant blue
        NodeType.OUTPUT: '#FA8072',  # More vibrant red
    }

    font_type = "DejaVu Sans"
    
    node_colors = []
    node_types = {} 
    
    for node in dag.nodes:
        G.add_node(node.name)
        node_type = node.node_type
        node_types[node.name] = node_type
        color = node_type_colors.get(node_type, '#D3D3D3')
        node_colors.append(color)
    
    # Add edges
    for node in dag.nodes:
        for child in node.children:
            G.add_edge(node.name, child.name)
    

    if len(dag.nodes) > 20:
        plt.figure(figsize=(25, 18), facecolor='white')
    else:
        plt.figure(figsize=(18, 18), facecolor='white')
    plt.tight_layout()
    

    try:
        if left_to_right:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=LR')
        else:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')

        input_nodes = [n for n, t in node_types.items() if t == NodeType.INPUT]
        output_nodes = [n for n, t in node_types.items() if t == NodeType.OUTPUT]
        
        if input_nodes and (left_to_right or not left_to_right):
            for node_name in G.nodes():
                if node_name not in pos:
                    print(f"Warning: Node '{node_name}' is not in the position dictionary, using default position")
                    pos[node_name] = (0, 0)
        
            if left_to_right:
                min_x = min(x for x, y in pos.values())
                for i, node in enumerate(input_nodes):
                    if node in pos:
                        x, y = pos[node]
                        pos[node] = (min_x - 50, y + i * 50)
            else:
                max_y = max(y for x, y in pos.values())
                for i, node in enumerate(input_nodes):
                    if node in pos:
                        x, y = pos[node]
                        pos[node] = (x + i * 100, max_y + 100)
    except Exception as e:
        print(f"Error using graphviz layout: {e}")
        print("Falling back to spring layout...")
        pos = nx.spring_layout(G)
    
    for node_name in G.nodes():
        if node_name in pos:
            print(f"Node {node_name} position: {pos[node_name]}")
        else:
            print(f"Warning: Node {node_name} is not in the position dictionary")
    
    # Ensure all nodes have positions
    for node_name in G.nodes():
        if node_name not in pos:
            print(f"Assigning default position to missing node '{node_name}'")
            pos[node_name] = (0, 0)
    
    # Set different colors for different types of edges
    edge_colors = []
    for u, v in G.edges():
        if node_types.get(u) == NodeType.INPUT:
            edge_colors.append('#4CAF50')  # The edges from the input node - Emerald green
        elif node_types.get(u) == NodeType.COMPUTE:
            edge_colors.append('#2196F3')  # The edges from the compute node - Sky blue
        elif node_types.get(u) == NodeType.OUTPUT:
            edge_colors.append('#FF5722')  # The edges from the output node - Deep orange
        else:
            edge_colors.append('#9E9E9E')  # Other edges - Gray
    
    
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=3000,
                          edgecolors='#2F4F4F',
                          linewidths=2,
                          alpha=0.9)

    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors,
                          width=1.5,
                          alpha=0.7,
                          arrowsize=20,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.1',
                          node_size=3000)
    
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_family=font_type,
                           font_weight='bold')
    
    plt.title(f"DAG: {dag.name or dag.id}", fontsize=16, fontweight='bold', pad=20)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=15, 
                                 markeredgecolor='#2F4F4F', markeredgewidth=1.5,
                                 label=node_type)
                      for node_type, color in node_type_colors.items()]
    
    legend_elements.append(plt.Line2D([0], [0], color='#FF6347', lw=2, 
                                     label='Input Node Connection'))
    legend_elements.append(plt.Line2D([0], [0], color='#2F4F4F', lw=2, 
                                     label='Normal Connection'))
    
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, 
              framealpha=0.9, edgecolor='#2F4F4F')
    
    for node in dag.nodes:
        if hasattr(node, 'io_schema') and node.io_schema:
            input_names = list(node.io_schema.input_format.keys()) if node.io_schema.input_format else []
            output_names = list(node.io_schema.output_format.keys()) if node.io_schema.output_format else []

            op_type = node.op_type 

            if node.input_shards_mapping:
                input_shards_mapping = node.input_shards_mapping 
            else:
                input_shards_mapping = None
            
            if input_names:
                formatted_inputs = "[ " + ",\n  ".join(input_names) + "]"
            else:
                formatted_inputs = "[]"
                
            if output_names:
                formatted_outputs = "[ " + ",\n  ".join(output_names) + "]"
            else:
                formatted_outputs = "[]"
                
            if input_shards_mapping:
                formatted_mapping = str(input_shards_mapping).replace(", ", ",\n  ").replace("{", "{  ").replace("}", "}")
            else:
                formatted_mapping = "None"
                
            if formatted_mapping == "None":
                label_text = f"Input: {formatted_inputs}\nOutput: {formatted_outputs}\nOpType: {op_type}"
            else:
                label_text = f"Input: {formatted_inputs}\nOutput: {formatted_outputs}\nInput Shards: {formatted_mapping}\nOpType: {op_type}"
            
            if node.name in pos:
                x, y = pos[node.name]
                offset = -20 if left_to_right else -28
                plt.annotate(label_text, 
                            xy=(x, y),
                            xytext=(x, y + offset),
                            bbox=dict(boxstyle="round,pad=0.5", fc="#F8F8FF", ec="#2F4F4F", alpha=0.9),
                            ha='center', 
                            va='top',
                            fontsize=8,
                            fontfamily=font_type)
    
    plt.axis('off')
    plt.grid(False)
    plt.tight_layout(pad=2.0)
    
    # Save the image with high DPI
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    
    return G


def visualize_dag_with_compute_nodes_in_line(dag: DAG, output_path: str = None, show: bool = True, horizontal: bool = False):
    """
    Make a DAG visualization as a graph with all compute nodes aligned in a straight line
    
    Args:
        dag: The DAG object to visualize
        output_path: The path to save the image (optional)
        show: Whether to display the image
        horizontal: If True, compute nodes are aligned horizontally, otherwise vertically
    """
    # Ensure the topological sort is up to date
    dag._ensure_topo_sort()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # The mapping from node types to colors
    node_type_colors = {
        NodeType.INPUT: '#90EE90',  # More vibrant green
        NodeType.COMPUTE: '#87CEFA',  # More vibrant blue
        NodeType.OUTPUT: '#FA8072',  # More vibrant red
    }

    font_type = "DejaVu Sans"
    
    node_colors = []
    node_types = {} 
    
    # Add nodes to the graph
    for node in dag.nodes:
        G.add_node(node.name)
        node_type = node.node_type
        node_types[node.name] = node_type
        color = node_type_colors.get(node_type, '#D3D3D3')
        node_colors.append(color)
    
    # Add edges
    for node in dag.nodes:
        for child in node.children:
            G.add_edge(node.name, child.name)
    
    # Set figure size
    if len(dag.nodes) > 20:
        plt.figure(figsize=(25, 18), facecolor='white')
    else:
        plt.figure(figsize=(18, 18), facecolor='white')
    plt.tight_layout()
    
    # Separate nodes by type
    input_nodes = [n for n, t in node_types.items() if t == NodeType.INPUT]
    compute_nodes = [n for n, t in node_types.items() if t == NodeType.COMPUTE]
    output_nodes = [n for n, t in node_types.items() if t == NodeType.OUTPUT]
    
    # Create custom positions
    pos = {}
    
    # Position compute nodes in a straight line
    if horizontal:
        # Horizontal line for compute nodes
        compute_line_y = 0
        compute_spacing = 200
        for i, node in enumerate(compute_nodes):
            pos[node] = (i * compute_spacing, compute_line_y)
        
        # Position input nodes above compute nodes
        input_spacing = compute_spacing * len(compute_nodes) / (len(input_nodes) + 1) if input_nodes else 0
        for i, node in enumerate(input_nodes):
            pos[node] = ((i + 1) * input_spacing, compute_line_y + 150)
        
        # Position output nodes below compute nodes
        output_spacing = compute_spacing * len(compute_nodes) / (len(output_nodes) + 1) if output_nodes else 0
        for i, node in enumerate(output_nodes):
            pos[node] = ((i + 1) * output_spacing, compute_line_y - 150)
    else:
        compute_line_x = 0
        compute_spacing = 150
        for i, node in enumerate(compute_nodes):
            pos[node] = (compute_line_x, -i * compute_spacing)
        
        input_spacing = 200
        if len(input_nodes) > 1:
            input_width = input_spacing * (len(input_nodes) - 1)
            start_x = compute_line_x - input_width / 2
            for i, node in enumerate(input_nodes):
                pos[node] = (start_x + i * input_spacing, 150)
        else:

            for i, node in enumerate(input_nodes):
                pos[node] = (compute_line_x, 150)
        
        output_spacing = 200
        compute_bottom = -((len(compute_nodes) - 1) * compute_spacing) if compute_nodes else 0
        for i, node in enumerate(output_nodes):
            pos[node] = (compute_line_x + 150, compute_bottom - 150)
    
    # Set different colors for different types of edges
    edge_colors = []
    for u, v in G.edges():
        if node_types.get(u) == NodeType.INPUT:
            edge_colors.append('#4CAF50')  # The edges from the input node - Emerald green
        elif node_types.get(u) == NodeType.COMPUTE:
            edge_colors.append('#2196F3')  # The edges from the compute node - Sky blue
        elif node_types.get(u) == NodeType.OUTPUT:
            edge_colors.append('#FF5722')  # The edges from the output node - Deep orange
        else:
            edge_colors.append('#9E9E9E')  # Other edges - Gray
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=node_colors,
                          node_size=3000,
                          edgecolors='#2F4F4F',
                          linewidths=2,
                          alpha=0.9)

    # Draw edges with curved connections
    nx.draw_networkx_edges(G, pos, 
                          edge_color=edge_colors,
                          width=1.5,
                          alpha=0.7,
                          arrowsize=20,
                          arrowstyle='-|>',
                          connectionstyle='arc3,rad=0.2',
                          node_size=3000)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, 
                           font_size=10,
                           font_family=font_type,
                           font_weight='bold')
    
    # Add title
    plt.title(f"DAG: {dag.name or dag.id}", fontsize=16, fontweight='bold', pad=20)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=15, 
                                 markeredgecolor='#2F4F4F', markeredgewidth=1.5,
                                 label=node_type)
                      for node_type, color in node_type_colors.items()]
    
    legend_elements.append(plt.Line2D([0], [0], color='#4CAF50', lw=2, 
                                     label='Input Node Connection'))
    legend_elements.append(plt.Line2D([0], [0], color='#2196F3', lw=2, 
                                     label='Compute Node Connection'))
    legend_elements.append(plt.Line2D([0], [0], color='#FF5722', lw=2, 
                                     label='Output Node Connection'))
    
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, 
              framealpha=0.9, edgecolor='#2F4F4F')
    
    # Add node information annotations
    for node in dag.nodes:
        if hasattr(node, 'io_schema') and node.io_schema:
            input_names = list(node.io_schema.input_format.keys()) if node.io_schema.input_format else []
            output_names = list(node.io_schema.output_format.keys()) if node.io_schema.output_format else []

            op_type = node.op_type 

            if node.input_shards_mapping:
                input_shards_mapping = node.input_shards_mapping 
            else:
                input_shards_mapping = None
            
            if input_names:
                formatted_inputs = "[ " + ",\n  ".join(input_names) + "]"
            else:
                formatted_inputs = "[]"
                
            if output_names:
                formatted_outputs = "[ " + ",\n  ".join(output_names) + "]"
            else:
                formatted_outputs = "[]"
                
            if input_shards_mapping:
                formatted_mapping = str(input_shards_mapping).replace(", ", ",\n  ").replace("{", "{  ").replace("}", "}")
            else:
                formatted_mapping = "None"
                
            if formatted_mapping == "None":
                label_text = f"Input: {formatted_inputs}\nOutput: {formatted_outputs}\nOpType: {op_type}"
            else:
                label_text = f"Input: {formatted_inputs}\nOutput: {formatted_outputs}\nInput Shards: {formatted_mapping}\nOpType: {op_type}"
            
            if node.name in pos:
                x, y = pos[node.name]
                offset_y = -40 if horizontal else 0
                offset_x = 0 if horizontal else 40
                plt.annotate(label_text, 
                            xy=(x, y),
                            xytext=(x + offset_x, y + offset_y),
                            bbox=dict(boxstyle="round,pad=0.5", fc="#F8F8FF", ec="#2F4F4F", alpha=0.9),
                            ha='center', 
                            va='top',
                            fontsize=8,
                            fontfamily=font_type)
    
    plt.axis('off')
    plt.grid(False)
    plt.tight_layout(pad=2.0)
    
    # Save the image with high DPI
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return G


if __name__ == "__main__":
    from Ayo.dags.dag import DAG
    from Ayo.dags.node import Node
    from Ayo.dags.node_commons import NodeType, NodeIOSchema, NodeAnnotation, NodeOps
    from Ayo.engines.engine_types import EngineType
    from typing import Any, List
    from Ayo.opt_pass.stage_decomposition import StageDecompositionPass

    dag = DAG()

    dag.set_query_inputs({"query": "What is the capital of France?", 
                          "passages": ["Paris is the capital of France.", 
                                      "France is a country in Europe.", 
                                      "China is a country in Asia.", 
                                      "Asia is a continent in the world.", 
                                      "Europe is a continent in the world.", 
                                      "America is a continent in the world."]})
    

    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        op_type=NodeOps.EMBEDDING,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str]},
            output_format={"embeddings_passages": List[Any]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    Ingestion_node = Node(
        name="Ingestion",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        op_type=NodeOps.VECTORDB_INGESTION,
        io_schema=NodeIOSchema( 
            input_format={"embeddings_passages": List[Any]},
            output_format={"ingested": bool}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        op_type=NodeOps.VECTORDB_SEARCHING,
        io_schema=NodeIOSchema(
            input_format={"query": List[Any], "ingested": bool, "k": int},
            output_format={"search_results": List[str]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
        }
    )

    embedding_node >> Ingestion_node >> search_node 

    dag.register_nodes(embedding_node, Ingestion_node, search_node)

    print(dag.get_full_dag_nodes_info())

    begin=time.time()

    dag.optimize([StageDecompositionPass()])

    end=time.time()

    print(f"Optimize time: {end-begin} seconds")

    print(dag.get_full_dag_nodes_info())



    visualize_dag_with_node_types(dag, "test_dag_node_types.png")
