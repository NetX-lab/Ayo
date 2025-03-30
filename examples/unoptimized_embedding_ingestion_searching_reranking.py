from Ayo.app import APP
from Ayo.dags.node import Node, NodeAnnotation, NodeType, NodeIOSchema, NodeOps
from Ayo.dags.dag import DAG
from Ayo.queries.query import Query
from Ayo.configs.config import EngineConfig
from Ayo.engines.engine_types import ENGINE_REGISTRY, EngineType
import time
import asyncio
import ray
from copy import deepcopy
from typing import List, Any, Dict, Optional, Union
import traceback


default_embed_config = ENGINE_REGISTRY.get_default_config(EngineType.EMBEDDER)
default_embed_config.update(
    {
        "model_name": "BAAI/bge-large-en-v1.5",
        "max_batch_size": 1024,
    }
)
embed_config = EngineConfig(
    name="embedding_service",
    engine_type=EngineType.EMBEDDER,
    resources={},
    num_gpus=1,
    num_cpus=1,
    instances=2,
    model_config={
        **default_embed_config,
        "device": "cuda"
    },
    latency_profile={     
        "timeout": 300,
        "batch_wait": 0.1
    }
)


vectordb_config = EngineConfig(
    name="vector_db_service",
    engine_type=EngineType.VECTOR_DB,
    resources={},
    num_gpus=0,
    num_cpus=4,
    instances=1,
    model_config={
        "host": "localhost",
        "port": 5432,
        "user": "asplos25",
        "password": "123456",
        "database": "database_asplos",
        "vector_dim": 1024,
        "max_batch_size": 1000
    },
    latency_profile={
        "timeout": 60,
        "batch_wait": 0.05
    }
)


default_reranker_config = ENGINE_REGISTRY.get_default_config(EngineType.RERANKER)
default_reranker_config.update(
    {
        "model_name": "BAAI/bge-reranker-large",
        "max_batch_size": 32,
    }
)
reranker_config = EngineConfig(
    name="reranker_service",
    engine_type=EngineType.RERANKER,
    resources={},
    num_gpus=1,
    num_cpus=1,
    instances=1,
    model_config={
        **default_reranker_config,
        "device": "cuda"
    },
    latency_profile={
        "timeout": 60,
        "batch_wait": 0.1
    }
)

def create_base_dag():
    """Create basic DAG"""
    base_dag = DAG(dag_id="random-test")
    
    passages_embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str]},
            output_format={"passages_embeddings": List[List[float]]}
        ),
        op_type=NodeOps.EMBEDDING,
        anno=NodeAnnotation.BATCHABLE,
        config={

        }
    )

    ingestion_node = Node(
        name="Ingestion",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str], "passages_embeddings": List[List[float]]},
            output_format={"index_status": bool}
        ),
        op_type=NodeOps.VECTORDB_INGESTION,
        anno=NodeAnnotation.BATCHABLE,
        config={

        }
    )
    
    query_embedding_node = Node(
        name="QueryEmbedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"queries": List[str]},
            output_format={"queries_embeddings": List[List[float]]}
        ),
        op_type=NodeOps.EMBEDDING,
        anno=NodeAnnotation.BATCHABLE,
        config={

        }
    )
    
    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={"queries_embeddings": List[List[float]], "index_status": bool},
            output_format={"search_results": List[List[str]]}
        ),
        op_type=NodeOps.VECTORDB_SEARCHING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "top_k": 30  # Add more retrieval to provide more candidates for reranking
        }
    )
    
    reranking_node = Node(
        name="Reranking",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        io_schema=NodeIOSchema(
            input_format={"queries": str, "search_results": Union[List[List[str]],List[str]]},
            output_format={"reranked_results": List[List[str]]}
        ),
        op_type=NodeOps.RERANKING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "top_k": 10  # The number of results to return
        }
    )

    base_dag.set_query_inputs(
        {
                 "passages": [
                    "AI is a branch of computer science..." * 15,
                    "Machine learning is a subset of AI..." * 15,
                    "Deep learning revolutionized AI..." * 15,
                    "I have a question about AI..." * 15,  
                    "I have a question about machine learning..." * 15,
                    'What is the latest news about AI?' * 15,
                    'What is the latest news about machine learning?' * 15,
                    'What is the latest news about deep learning?' * 15,
                    'What is the latest news about AI?' * 15,
                    'What is the latest news about machine learning?' * 15,
                    'What is the latest news about deep learning?' * 15,
                ]*60 + [
                    "OSDI is a conference about operating systems..." * 15, 
                    "ASPLOS is a conference about computer architecture..." * 15,
                    "MICRO is a conference about computer architecture..." * 15,
                    "HPCA is a conference about computer architecture..." * 15,
                    "MLSYS is a conference about machine learning..." * 15, 
                    "Machine learning system design is a conference about machine ..." * 15, 
                ],
                "queries": [
                    "I want to know some system conferences."
                ]
        },
    )


    passages_embedding_node >> ingestion_node >> query_embedding_node >> search_node >> reranking_node
    ingestion_node >> search_node
    base_dag.register_nodes(passages_embedding_node, ingestion_node, query_embedding_node, search_node, reranking_node)
    
    return base_dag


async def process_query(app, queries: List[str], passages: List[str], dag, query_id: str):
    """Add a query to the app and process it"""
    try:
        query = Query(
            uuid=f"random-test-{query_id}",
            query_id=f"random-test-{query_id}",
            query_inputs={
                "passages": passages,
                "queries": queries
            },
            DAG=deepcopy(dag)
        )

        future = await app.submit_query(
            query=query,
            timeout=300
        )
        
        result = await asyncio.wait_for(future, timeout=300)
        return result
            
    except Exception as e:
        print(f"Query {query_id} processing failed:\n{traceback.format_exc()}")
        raise Exception(f"Query {query_id} processing failed: {str(e)}")       


async def run_app(dag):
    try:
        # Initialize the application
        app = APP.init()
        app.register_engine(embed_config)
        app.register_engine(vectordb_config)
        app.register_engine(reranker_config)  # Register the reranker engine
        
        
        # Start the application
        await app.start()
        await asyncio.sleep(5)

        async def delayed_query(query_data, index):
            if index > 0:
                await asyncio.sleep(3 * index)
            return await process_query(
                app,
                query_data["queries"],
                query_data["passages"],
                dag,
                str(index)
            )

        test_queries = [
            {
                 "passages": [
                    "AI is a branch of computer science..." * 15,
                    "Machine learning is a subset of AI..." * 15,
                    "Deep learning revolutionized AI..." * 15,
                    "I have a question about AI..." * 15,  
                    "I have a question about machine learning..." * 15,
                    'What is the latest news about AI?' * 15,
                    'What is the latest news about machine learning?' * 15,
                    'What is the latest news about deep learning?' * 15,
                    'What is the latest news about AI?' * 15,
                    'What is the latest news about machine learning?' * 15,
                    'What is the latest news about deep learning?' * 15,
                ]*60 + [
                    "OSDI is a conference about operating systems..." * 15, 
                    "ASPLOS is a conference about computer architecture..." * 15,
                    "MICRO is a conference about computer architecture..." * 15,
                    "HPCA is a conference about computer architecture..." * 15,
                    "MLSYS is a conference about machine learning..." * 15, 
                    "Machine learning system design is a conference about machine ..." * 15, 
                ],
                "queries": [
                    "I want to know some system conferences."
                ]
            },
        ]
        
        # create all tasks
        tasks = [
            delayed_query(query_data, i)
            for i, query_data in enumerate(test_queries)
        ]
        
        # wait for all queries to complete
        results = await asyncio.gather(*tasks)
        
        # print results
        for i, result in enumerate(results):
            print(f"\nQuery {i} results: {result}")
            for key, value in result.items(): 
                if 'search' in key.lower():
                    print(f"Search results length: {len(value)}")
                elif 'reranking' in key.lower():
                    print(f"Reranking results length: {len(value)}")

    except Exception as e:
        print(f"Main program error stack:\n{traceback.format_exc()}")
        print(f"Main program error: {e}")
        raise
    finally:
        # clean up
        try:
            await cleanup(app)
        except Exception as e:
            print(f"Cleanup process error: {e}") 

async def cleanup(app):
    """Clean up resources"""
    try:
        await app.stop()
        app.shutdown()
    except Exception as e:
        print(f"Cleanup failed:\n{traceback.format_exc()}")
        print(f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    from Ayo.vis.vis_graph import visualize_dag_with_node_types
    
    dag = create_base_dag()


    visualize_dag_with_node_types(dag, "unoptimized_embedding_ingestion_search_reranking.png")


    asyncio.run(run_app(dag))
