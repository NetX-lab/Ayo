import time
from typing import List

from Ayo.dags.dag import DAG
from Ayo.modules.indexing import IndexingModule
from Ayo.modules.query_expanding import QueryExpandingModule
from Ayo.modules.reranking import RerankingModule
from Ayo.modules.searching import SearchingModule
from Ayo.opt_pass.decoding_pipeling import LLMDecodingPipeliningPass
from Ayo.opt_pass.pruning_dependency import PruningDependencyPass
from Ayo.opt_pass.stage_decomposition import StageDecompositionPass
from Ayo.utils import print_key_info
from Ayo.vis.vis_graph import visualize_dag_with_node_types

indexing_module = IndexingModule(
    input_format={"passages": List[str]}, output_format={"index_status": bool}
)
query_expanding_module = QueryExpandingModule(
    input_format={"query": str},
    output_format={"expanded_queries": List[str]},
    config={"expanded_query_num": 3},
)
searching_module = SearchingModule(
    input_format={"index_status": bool, "expanded_queries": List[str]},
    output_format={"searching_results": List[str]},
)
reranking_module = RerankingModule(
    input_format={"query": str, "searching_results": List[str]},
    output_format={"reranking_results": List[str]},
)


indexing_nodes = indexing_module.to_primitive_nodes()
query_expanding_nodes = query_expanding_module.to_primitive_nodes()
searching_nodes = searching_module.to_primitive_nodes()
reranking_nodes = reranking_module.to_primitive_nodes()


indexing_nodes[-1] >> query_expanding_nodes[0]

query_expanding_nodes[-1] >> searching_nodes[0]

searching_nodes[-1] >> reranking_nodes[0]

dag = DAG(dag_id="test_embed_ingest_search_reranking")

dag.register_nodes(
    *indexing_nodes, *query_expanding_nodes, *searching_nodes, *reranking_nodes
)

dag.set_query_inputs(
    {
        "passages": [
            "passages1",
            "passages2",
            "passages3",
        ]
        * 100,
        "query": "What is the capital of France?",
    }
)

print(dag.get_full_dag_nodes_info())

begin_time = time.time()

dag.optimize([PruningDependencyPass()])

dag.optimize([StageDecompositionPass()])

dag.optimize([LLMDecodingPipeliningPass()])

end_time = time.time()

print_key_info(f"Time taken: {end_time - begin_time} seconds")


visualize_dag_with_node_types(dag, output_path="test_embed_ingest_search_reranking.png")
