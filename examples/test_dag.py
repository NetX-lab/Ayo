import pytest
from Ayo.dags.node import Node, NodeType, NodeAnnotation, NodeIOSchema
from Ayo.dags.dag import DAG
from Ayo.engines.engine_types import EngineType
def test_simple_linear_dag():
    """Test simple linear DAG"""
    # Create node IO schema
    embed_schema = NodeIOSchema(
        input_format={"texts": str},
        output_format={"embeddings": list}
    )
    
    rerank_schema = NodeIOSchema(
        input_format={"embeddings": list},
        output_format={"scores": list}
    )
    
    # Create compute node
    node1 = Node(
        name="Embedding1",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=embed_schema,
        anno=NodeAnnotation.BATCHABLE,
        in_kwargs={"texts": None},
        out_kwargs={"embeddings": None}
    )
    
    node2 = Node(
        name="Rerank1",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        io_schema=rerank_schema,
        anno=NodeAnnotation.BATCHABLE,
        in_kwargs={"embeddings": None},
        out_kwargs={"scores": None}
    )
    
    # Create DAG
    dag = DAG(dag_id="test_linear")
    
    # Set query inputs
    dag.set_query_inputs({
        "texts": ["This is a test text"]
    })
    
    # Build dependencies
    node1 >> node2
    
    # Register nodes
    dag.register_nodes(node1, node2)
    
    # Validate topological sort
    sorted_nodes = dag.topological_sort()
    print(sorted_nodes)
    assert len(sorted_nodes) == 4  # input_node + 2个计算节点 + output_node
    assert sorted_nodes[0].node_type == NodeType.INPUT
    assert sorted_nodes[1] == node1
    assert sorted_nodes[2] == node2
    assert sorted_nodes[3].node_type == NodeType.OUTPUT

def test_diamond_dag():
    """Test diamond DAG"""
    # Create node IO schema
    embed_schema = NodeIOSchema(
        input_format={"texts": str},
        output_format={"embeddings": list}
    )
    
    process_schema = NodeIOSchema(
        input_format={"embeddings": list},
        output_format={"processed": list}
    )
    
    merge_schema = NodeIOSchema(
        input_format={"processed1": list, "processed2": list},
        output_format={"final": list}
    )
    
    # Create compute node
    node1 = Node(
        name="Embedding1",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=embed_schema,
        in_kwargs={"texts": None},
        out_kwargs={"embeddings": None}
    )
    
    node2 = Node(
        name="Process1",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.DUMMY,
        io_schema=process_schema,
        in_kwargs={"embeddings": None},
        out_kwargs={"processed1": None}
    )
    
    node3 = Node(
        name="Process2",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.DUMMY,
        io_schema=process_schema,
        in_kwargs={"embeddings": None},
        out_kwargs={"processed2": None}
    )
    
    node4 = Node(
        name="Merge",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.AGGREGATOR,
        io_schema=merge_schema,
        in_kwargs={"processed1": None, "processed2": None},
        out_kwargs={"final": None}
    )
    
    # Create DAG
    dag = DAG(dag_id="test_diamond")
    
    # Set query inputs
    dag.set_query_inputs({
        "texts": ["This is a test text"]
    })
    
    # Build dependencies
    node1 >> node2
    node1 >> node3
    node2 >> node4
    node3 >> node4
    
    # Register nodes
    dag.register_nodes(node1, node2, node3, node4)
    
    # Validate topological sort
    sorted_nodes = dag.topological_sort()
    print(sorted_nodes)
    assert len(sorted_nodes) == 6  # input_node + 4个计算节点 + output_node
    assert sorted_nodes[0].node_type == NodeType.INPUT
    assert sorted_nodes[1] == node1
    assert set(sorted_nodes[2:4]) == {node2, node3}
    assert sorted_nodes[4] == node4
    assert sorted_nodes[5].node_type == NodeType.OUTPUT

def test_cyclic_dag():
    """Test cyclic DAG (should raise an exception)"""
    # Create node IO schema
    process_schema = NodeIOSchema(
        input_format={"input": str},
        output_format={"output": str}
    )
    
    # Create compute node
    node1 = Node(
        name="Node1",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.DUMMY,
        io_schema=process_schema,
        in_kwargs={"input": None},
        out_kwargs={"output": None}
    )
    
    node2 = Node(
        name="Node2",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.DUMMY,
        io_schema=process_schema,
        in_kwargs={"input": None},
        out_kwargs={"output": None}
    )
    
    # Create DAG
    dag = DAG(dag_id="test_cyclic")
    
    # Set query inputs
    dag.set_query_inputs({
        "input": "test"
    })
    
    # Build dependencies
    node1 >> node2
    node2 >> node1  
    
    # Register nodes
    dag.register_nodes(node1, node2)
    
    # Validate if an exception is raised
    with pytest.raises(ValueError, match="Cycle detected in DAG"):
        dag.topological_sort()

def test_dag_with_query_inputs():
    """Test DAG with query inputs"""
    # Create node IO schema
    embed_schema = NodeIOSchema(
        input_format={"texts": str},
        output_format={"embeddings": list}
    )
    
    # Create compute node
    compute_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=embed_schema,
        in_kwargs={"texts": None},
        out_kwargs={"embeddings": None}
    )
    
    # Create DAG
    dag = DAG(dag_id="test_with_inputs")
    
    # Set query inputs
    dag.set_query_inputs({
        "texts": ["This is a test text"]
    })
    
    # Register nodes
    dag.register_nodes(compute_node)
    
    # Validate if the input node is created correctly
    assert len(dag.input_nodes) == 1
    print(dag.input_nodes)
    print(list(dag.input_nodes.values())[0].input_kwargs)
    assert "texts" in list(dag.input_nodes.values())[0].input_kwargs
    
    # Validate topological sort
    sorted_nodes = dag.topological_sort()
    assert len(sorted_nodes) == 3  # input_node + compute_node + output_node
    assert sorted_nodes[0].node_type == NodeType.INPUT
    assert sorted_nodes[1] == compute_node
    assert sorted_nodes[2].node_type == NodeType.OUTPUT

if __name__ == "__main__":
    pytest.main([__file__])
