from enum import Enum
import ray
from typing import List, Dict, Optional, Any, Tuple, Union
from pydantic import Field
from dataclasses import dataclass
from Ayo.engines.payload_transformers import TRANSFORMER_REGISTRY, DefaultTransformer
from Ayo.engines.engine_types import EngineType, ENGINE_REGISTRY
from Ayo.dags.node_commons import NodeType, NodeOps, NodeAnnotation, NodeStatus, NodeIOSchema
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class Node:
    """Node in workflow template representing a component/primitive"""
    
    def __init__(self,
                 name: str, 
                 node_type: NodeType,
                 engine_type: Optional[str] = None,
                 op_type: Optional[NodeOps] = None,         
                 io_schema: Optional[NodeIOSchema] = None,
                 anno: NodeAnnotation = NodeAnnotation.NONE,
                 config: Optional[Dict] = None,
                 **kwargs):
        """Initialize a node
        
        Args:
            name: Name of the node
            node_type: Type of the node
            engine_type: Type of engine for this node
            io_schema: Input/output schema for the node
            anno: Node annotation for optimization hints
            config: Node specific configuration
            **kwargs: Additional keyword arguments
        """
        self.name = name
        self.node_type = node_type
        self.engine_type = engine_type
        self.op_type = op_type
        self.io_schema = io_schema
        self.anno = anno
        self.config = config or {}
        
        # Validate engine type if provided
        if engine_type and not EngineType.validate(engine_type):
            raise ValueError(f"Unsupported engine type: {engine_type}")
        
        # 初始化基本属性
        self.input_values = {}
        self.input_kwargs = {} # the input parameters for the node, {k:v}, k is the name of the input field, v is the value of the input field, default is None 

        self.output_names = []
        self.parents: List[Node] = []
        self.children: List[Node] = []
        self.status = NodeStatus.PENDING
        self.error_message = None
        self.result_ref: Optional[ray.ObjectRef] = None
        self.input_key_from_parents: Dict[str, str] = {}
        self.input_key_to_parent: Dict[str, str] = {}

        
        # 针对不同类型节点的特殊处理
        if node_type == NodeType.INPUT:
            self._init_input_node(kwargs)
            self.engine_type = kwargs.get("engine_type", EngineType.INPUT)
        elif node_type == NodeType.COMPUTE:
            self._init_compute_node(kwargs)
            assert self.engine_type is not None, "Compute node requires engine_type"

        elif node_type == NodeType.OUTPUT:
            self._init_output_node(kwargs)
            self.engine_type = kwargs.get("engine_type", EngineType.OUTPUT)

        self.depth = 0


        # for opt  
        self.decomposed: bool = False
        self.input_shards_mapping: Dict[str, List[Union[slice, tuple]]] = {}  
        """
        The shards mapping for the input fields, the key is the name of the input field, the value is the list of shards
        - key: the name of the input field
        - value: the list of shards, support two formats:
          1. slice object: for direct slicing operation (e.g. slice(0, 100))
          2. (start, end) tuple: will be converted to slice object
        Example: {"text": [slice(0, 512), (513, 1024)]}
        """


        # be used to record the output shape info of the node
        # {
        #     "passages":
        #     {
        #         "type": List[str],
        #         "shape": (50, 300)
        #     }
        # }
        self.output_shape_info = {}

        self.init_output_shape_info()

    def init_output_shape_info(self) -> None:
        """Initialize the output shape info of the node"""
        for output_name, output_type in self.io_schema.output_format.items():
            self.output_shape_info[output_name] = {
                "type": output_type,
                "shape": None
            }

    def _init_input_node(self, kwargs: Dict) -> None:
        """Initialize input node specific attributes"""
        self.input_values = kwargs.get("input_values", {}) 
        self.input_kwargs = {
            k: None for k, v in self.io_schema.input_format.items()
        }
        self.output_names = list(self.io_schema.output_format.keys())

    def _init_compute_node(self, kwargs: Dict) -> None:
        """Initialize compute node specific attributes"""
        if not self.io_schema:
            raise ValueError("Compute node requires input/output schema")
        self.output_names = list(self.io_schema.output_format.keys())
        self.input_kwargs = {
            k: None for k, v in self.io_schema.input_format.items()
        }       

        # add the specific attributes for the compute node

    def _init_output_node(self, kwargs: Dict) -> None:
        """Initialize output node specific attributes"""
        self.input_kwargs = kwargs.get("input_kwargs", {})
        # add the specific attributes for the output node

    def validate(self) -> bool:
        """Validate node configuration and connections"""
        try:
            # validate the basic attributes
            if not self.name:
                raise ValueError("Node name cannot be empty")
                
            # validate the specific requirements for the node type
            if self.node_type == NodeType.COMPUTE:
                if not self.io_schema:
                    raise ValueError("Compute node requires io_schema")
                if not self.engine_type:
                    raise ValueError("Compute node requires engine_type")
                    
            # validate the node connections
            if self.node_type == NodeType.INPUT and self.parents:
                raise ValueError("Input node should not have parents")
            if self.node_type == NodeType.OUTPUT and self.children:
                raise ValueError("Output node should not have children")
                
            return True
        except Exception as e:
            self.error_message = str(e)
            return False

    def reset(self) -> None:
        """Reset node state"""
        self.status = NodeStatus.PENDING
        self.error_message = None
        self.result_ref = None
        self.input_values.clear()
        self.input_kwargs.clear()
        self.input_key_from_parents.clear()
        self.input_key_to_parent.clear()

    @property
    def input_names(self) -> List[str]:
        """Get input field names"""
        if self.io_schema:
            return list(self.io_schema.input_format.keys())
        return list(self.input_kwargs.keys())
    
    def refresh_io_schema(self, IO_schema: NodeIOSchema) -> None:
        """Refresh the io_schema of the node"""
        self.io_schema = IO_schema
        self.output_names = list(self.io_schema.output_format.keys())
        self.input_kwargs = {
            k: None for k, v in self.io_schema.input_format.items()
        }


    def add_parent(self, parent: 'Node') -> None:
        """Add a parent node and establish input/output connections"""
        if parent not in self.parents:
            self.parents.append(parent)
            # Find matching input/output keys
            intersection = set(parent.output_names).intersection(set(self.input_names))
            logger.debug(f"parent: {parent.name}, child: {self.name}, intersection: {intersection}")
            if intersection:
                if len(intersection) > 1:
                    logger.warning(f"node: {self.name} has multiple input keys from parent: {parent.name} with intersection: {intersection}, please check the io_schema and the node connections")
                for key in intersection:
                    #here we assume the primitive would only provide one output  
                    self.input_key_from_parents[parent.name] = key
                    self.input_key_to_parent[key] = parent
            if self not in parent.children:
                parent.children.append(self)
    
    def add_child(self, child: 'Node') -> None:
        """Add a child node"""
        if child not in self.children:
            self.children.append(child)
            if self not in child.parents:
                child.add_parent(self)
    
    def __rshift__(self, other: 'Node') -> 'Node':
        """Implement >> operator for creating dependencies"""
        self.add_child(other)
        return other
    
    def is_splittable(self) -> bool:
        """Check if node is splittable"""
        return self.anno == NodeAnnotation.SPLITTABLE
    
    def is_batchable(self) -> bool:
        """Check if node is batchable"""
        return self.anno == NodeAnnotation.BATCHABLE
    
    @property
    def is_ready(self) -> bool:
        """Check if node is ready for execution"""
        if self.status != NodeStatus.PENDING:
            return False
        return all(parent.status == NodeStatus.COMPLETED for parent in self.parents)
    
    def get_engine_type(self) -> str:
        """Get the engine type for this node"""
        return self.engine_type
    
    def apply_shard(self, data: Any, shards: List[Union[slice, tuple]]) -> List[Any]:
        """Apply the shards to the data"""
        print(f"apply_shard for node: {self.name}, data type: {type(data)}, shards: {shards}")
        print(f"data len: {len(data)}")
        print(f"shards: {shards}")
        result = []
        if not isinstance(shards, list):
            shards=[shards]
        for s in shards:
            try:
                if isinstance(s, slice):
                    result.extend(data[s])
                elif isinstance(s, tuple) and len(s) == 2:
                    result.extend(data[slice(*s)])
                else:
                    raise ValueError(f"Invalid shard format: {s}, expected slice or (start, end) tuple")
            except (TypeError, IndexError) as e:
                raise ValueError(f"Failed to apply shard {s} to data of type {type(data)}: {e}")
        return result

    def update_input_kwargs(self, nodes_outputs: Dict[str, Any]) -> None:
        """Update input parameters from parent nodes' outputs"""
        if self.node_type == NodeType.COMPUTE:
            # validate and convert the inputs according to the schema
            for input_name, input_type in self.io_schema.input_format.items():
                if self.input_kwargs[input_name] is None:
                    for parent in self.parents:
                        if parent.name in nodes_outputs:
                            parent_output = nodes_outputs[parent.name]
                            print(f"parent_output type: {type(parent_output)} for node: {self.name}")
                            if input_name in parent_output:
                                if self.decomposed and input_name in self.input_shards_mapping:
                                    shards = self.input_shards_mapping[input_name]
                                    self.input_kwargs[input_name] = self.apply_shard(
                                        parent_output[input_name], 
                                        shards
                                    )
                                else:
                                    self.input_kwargs[input_name] = parent_output[input_name]
                else:
                    continue
                                
        elif self.node_type == NodeType.OUTPUT:
            # the output node directly copies data from the parent node
            if len(self.parents) == 1:
                parent = self.parents[0]
                if parent.name in nodes_outputs:
                    self.input_kwargs = {
                        k: nodes_outputs[parent.name][k] 
                        for k in self.input_kwargs.keys()
                        if k in nodes_outputs[parent.name]
                    }
    
    def clear_dependencies(self) -> None:
        """Clear all dependencies"""
        self.parents.clear()
        self.children.clear()
        self.input_key_from_parents.clear()
        self.input_key_to_parent.clear() 


    def to_dict(self) -> Dict:
        """Convert node to dictionary representation"""
        return {
            "name": self.name,
            "node_type": self.node_type.value,
            "engine_type": self.engine_type,
            "io_schema": vars(self.io_schema),
            "status": self.status.value
        }
    
    def __hash__(self) -> int:
        return hash((self.name, self.node_type))
    
    def __eq__(self, other: 'Node') -> bool:
        return self.name == other.name and self.node_type == other.node_type
    
    def __str__(self):
        return f"Node(name={self.name}, node_type={self.node_type.value})"
    
    def __repr__(self) -> str:
        return f"\n----Node({self.name})----\
                \nNodeType({self.node_type.value})\
                \nEngineType({self.engine_type})\
                \nOpType({self.op_type})\
                \nParents({[f'({x.name}, {x.node_type.value})' for x in self.parents]})\
                \nChildren({[f'({x.name}, {x.node_type.value})' for x in self.children]})\
                \nDepth({self.depth})\
                \nInput keys and types({[f'{k}: {type(v)}, shape: {self.get_shape_for_certain_types(v)}' for k, v in self.input_kwargs.items()]})\
                \nInput key from parents({self.input_key_from_parents})\
                \nInput mapping shards({self.input_shards_mapping})\
                \nOutput names({self.output_names})\
                \nConfig({self.config})"

    def get_shape_for_certain_types(self, data: Any) -> Tuple[int, ...]:
        """Get the shape of the data for certain types"""
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.shape
        elif isinstance(data, list):
            return (len(data),)
        elif isinstance(data, dict):
            return tuple(self.get_shape_for_certain_types(v) for v in data.values())
        elif isinstance(data, int):
            return (1,)
        elif isinstance(data, str):
            return (1,)
        elif isinstance(data, float):
            return (1,)
        elif isinstance(data, bool):
            return (1,)
        return None
    def get_attr(self, key: str) -> Any:
        """get attribute of the node"""
        return getattr(self, key)
        
    def set_attr(self, key: str, value: Any) -> None:
        """set attribute of the node"""
        setattr(self, key, value)


    def update_output_shape_info(self, output_data=None):
        """Update the output shape info of the node
        
        Args:
            output_data: optional actual output data, used to infer the shape
        """
        if output_data is not None:
            # update the shape info from the actual output data
            for output_name in self.output_names:
                if output_name in output_data:
                    data = output_data[output_name]
                    if hasattr(data, "shape"):
                        # if the data has shape attribute (e.g. numpy array or tensor)
                        self.output_shape_info[output_name]["shape"] = data.shape
                    elif isinstance(data, list):
                        # if the data is a list, infer the dimension
                        shape = [len(data)]
                        if data and isinstance(data[0], list):
                            shape.append(len(data[0]))
                        self.output_shape_info[output_name]["shape"] = tuple(shape)
        else:
            # infer the shape from the config or other information
            for output_name in self.output_names:
                # if the node is an input node, infer the shape from the input values
                if self.node_type == NodeType.INPUT and output_name in self.input_values:
                    data = self.input_values[output_name]
                    if hasattr(data, "shape"):
                        self.output_shape_info[output_name]["shape"] = data.shape
                    elif isinstance(data, list):
                        shape = [len(data)]
                        if data and isinstance(data[0], list):
                            shape.append(len(data[0]))
                        self.output_shape_info[output_name]["shape"] = tuple(shape)
                # infer the shape from the config
                elif "batch_size" in self.config:
                    batch_size = self.config["batch_size"]
                    # infer the shape from the different op types
                    if self.op_type == NodeOps.EMBEDDING:
                        feature_dim = self.config.get("embedding_dim", 768)
                        self.output_shape_info[output_name]["shape"] = (batch_size, feature_dim)
                    elif self.op_type == NodeOps.VECTORDB_SEARCHING:
                        top_k = self.config.get("top_k", None)
                        self.output_shape_info[output_name]["shape"] = (batch_size, top_k)
                    # can add more shape inference logic for different op types
