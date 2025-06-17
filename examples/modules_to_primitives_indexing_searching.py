import time

from Ayo.dags.dag import DAG
from Ayo.modules.indexing import IndexingModule
from Ayo.modules.searching import SearchingModule
from Ayo.opt_pass.pruning_dependency import PruningDependencyPass
from Ayo.opt_pass.stage_decomposition import StageDecompositionPass
from Ayo.vis.vis_graph import visualize_dag_with_node_types

indexing_module = IndexingModule()
searching_module = SearchingModule()


indexing_nodes = indexing_module.to_primitive_nodes()
searching_nodes = searching_module.to_primitive_nodes()

indexing_nodes[-1] >> searching_nodes[0]

chained_nodes = indexing_nodes + searching_nodes
print(chained_nodes)

dag = DAG(dag_id="test_module_to_primitives")

dag.register_nodes(*chained_nodes)

dag.set_query_inputs(
    {
        "passages": [
            "passages1",
            "passages2",
            "passages3",
            "passages4",
            "passages5",
        ]
        * 100,
        "queries": [
            "query1",
        ],
    }
)

print(dag.get_full_dag_nodes_info())

begin_time = time.time()
dag.optimize([PruningDependencyPass(), StageDecompositionPass()])
end_time = time.time()

print(f"\033[91mTime taken: {end_time - begin_time} seconds\033[0m")

print(dag.get_full_dag_nodes_info())


visualize_dag_with_node_types(dag, "test_module_to_primitives.png")
