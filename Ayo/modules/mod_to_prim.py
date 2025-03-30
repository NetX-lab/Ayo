from Ayo.modules.base_module import BaseModule
from Ayo.modules.indexing import IndexingModule
from Ayo.modules.query_expanding import QueryExpandingModule
from Ayo.modules.searching import SearchingModule
from Ayo.modules.reranking import RerankingModule
from typing import List

def transform_mod_to_prim(mods: List[BaseModule]):
    """
    Transform a chain of modules to a list of primitive nodes
    """
    mods_2_nodes={}
    for mod in mods:
        mods_2_nodes[mod]=mod.to_primitive_nodes()
    
    for mod in mods:
        for post_mod in mod.post_dependencies:
            mods_2_nodes[mod][-1]>>mods_2_nodes[post_mod][0]
    
    node_list=[]
    for mod in mods:
        node_list.extend(mods_2_nodes[mod])
    return node_list


if __name__=="__main__": 
    indexing_module = IndexingModule(input_format={"passages": List[str]}, output_format={"index_status": bool})
    query_expanding_module=QueryExpandingModule(input_format={"query": str}, output_format={"expanded_queries": List[str]},config={"expanded_query_num": 3})
    searching_module = SearchingModule(input_format={"index_status": bool, "expanded_queries": List[str]}, output_format={"searching_results": List[str]})
    reranking_module=RerankingModule(input_format={"searching_results": List[str]}, output_format={"reranking_results": List[str]})


    indexing_module>>query_expanding_module>>searching_module>>reranking_module 


    node_list=transform_mod_to_prim([indexing_module,query_expanding_module,searching_module,reranking_module])


    print(node_list)

