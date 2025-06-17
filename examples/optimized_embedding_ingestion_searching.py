import asyncio
import traceback
from copy import deepcopy
from typing import List

from Ayo.app import APP
from Ayo.configs.config import EngineConfig
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node, NodeAnnotation, NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import ENGINE_REGISTRY, EngineType
from Ayo.queries.query import Query

# Create embedding engine config
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
    instances=1,
    model_config={**default_embed_config, "device": "cuda"},
    latency_profile={"timeout": 300, "batch_wait": 0.1},
)

# Create search engine config
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
        "max_batch_size": 1000,
    },
    latency_profile={"timeout": 60, "batch_wait": 0.05},
)


def create_base_dag():
    """Create basic DAG"""
    base_dag = DAG(dag_id="random-test")

    # Create embedding node
    passages_embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"passages": List[str]},
            output_format={"passages_embeddings": List[List[float]]},
        ),
        op_type=NodeOps.EMBEDDING,
        anno=NodeAnnotation.BATCHABLE,
        config={"batch_size": embed_config.model_config.get("max_batch_size", 1024)},
    )

    ingestion_node = Node(
        name="Ingestion",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={
                "passages": List[str],
                "passages_embeddings": List[List[float]],
            },
            output_format={"index_status": bool},
        ),
        op_type=NodeOps.VECTORDB_INGESTION,
        anno=NodeAnnotation.BATCHABLE,
        config={"batch_size": embed_config.model_config.get("max_batch_size", 256)},
    )

    query_embedding_node = Node(
        name="QueryEmbedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"queries": List[str]},
            output_format={"queries_embeddings": List[List[float]]},
        ),
        op_type=NodeOps.EMBEDDING,
        anno=NodeAnnotation.BATCHABLE,
        config={"batch_size": embed_config.model_config.get("max_batch_size", 256)},
    )

    # Create search node
    search_node = Node(
        name="Search",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.VECTOR_DB,
        io_schema=NodeIOSchema(
            input_format={
                "queries_embeddings": List[List[float]],
                "index_status": bool,
            },
            output_format={"search_results": List[List[str]]},
        ),
        op_type=NodeOps.VECTORDB_SEARCHING,
        anno=NodeAnnotation.BATCHABLE,
        config={"batch_size": 16, "top_k": 5},
    )

    base_dag.set_query_inputs(
        {
            "passages": [
                "OSDI is a conference about operating systems..." * 20,
                "MICRO is a conference about computer architecture..." * 20,
                "HPCA is a conference about computer architecture..." * 20,
                "MLSYS is a conference about machine learning..." * 20,
                "Machine learning system design is a conference about ..." * 20,
                "AI is a branch of computer science..." * 20,
                "Machine learning is a subset of AI using statistical models " * 20,
                "Deep learning revolutionized AI using neural networks " * 20,
                "The sun is the largest planet in the solar system..." * 20,
                "The moon is the only natural satellite of the earth..." * 20,
                "The earth is the third planet from the sun..." * 20,
            ]
            * 80,
            "queries": [
                "I want to know some knowledge about the top computer system conferences."
            ],
        }
    )

    passages_embedding_node >> ingestion_node >> query_embedding_node >> search_node
    # query_embedding_node >> search_node
    base_dag.register_nodes(
        passages_embedding_node, ingestion_node, query_embedding_node, search_node
    )

    return base_dag


async def process_query(
    app, queries: List[str], passages: List[str], dag, query_id: str
):
    """Add a query to the app and process it"""
    try:
        query = Query(
            uuid=f"random-test-{query_id}",
            query_id=f"random-test-{query_id}",
            query_inputs={"passages": passages, "queries": queries},
            DAG=deepcopy(dag),
        )

        future = await app.submit_query(query=query, timeout=300)

        result = await asyncio.wait_for(future, timeout=300)
        return result

    except Exception as e:
        print(f"Query {query_id} processing failed:\n{traceback.format_exc()}")
        raise Exception(f"Query {query_id} processing failed: {str(e)}")


async def run_app(dag):
    try:
        # initialize the app
        app = APP.init()
        app.register_engine(embed_config)

        app.register_engine(vectordb_config)

        app.update_template(dag)

        # start the app
        await app.start()
        await asyncio.sleep(5)

        async def delayed_query(query_data, index):
            if index > 0:
                await asyncio.sleep(3 * index)
            return await process_query(
                app, query_data["queries"], query_data["passages"], dag, str(index)
            )

            # prepare test data

        test_queries = [
            {
                "passages": [
                    "OSDI is a conference about operating systems..." * 20,
                    "MICRO is a conference about computer architecture..." * 20,
                    "HPCA is a conference about computer architecture..." * 20,
                    "MLSYS is a conference about machine learning..." * 20,
                    "Machine learning system design is a conference about ..." * 20,
                    "AI is a branch of computer science..." * 20,
                    "Machine learning is a subset of AI using statistical models " * 20,
                    "Deep learning revolutionized AI using neural networks " * 20,
                    "The sun is the largest planet in the solar system..." * 20,
                    "The moon is the only natural satellite of the earth..." * 20,
                    "The earth is the third planet from the sun..." * 20,
                ]
                * 80,
                "queries": [
                    "I want to know some knowledge about the top computer system conferences."
                ],
            },
        ]

        # Create all tasks
        tasks = [
            delayed_query(query_data, i) for i, query_data in enumerate(test_queries)
        ]

        # Wait for all queries to complete
        results = await asyncio.gather(*tasks)

        # Print results
        for i, result in enumerate(results):
            print(f"\nQuery {i} results:")
            print(f"Results: {result}")

    except Exception as e:
        print(f"Main program error stack:\n{traceback.format_exc()}")
        print(f"Main program error: {e}")
        raise
    finally:
        # Cleanup
        try:
            await cleanup(app)
        except Exception as e:
            print(f"Cleanup process error: {e}")


async def cleanup(app):
    """clear resources"""
    try:
        await app.stop()
        app.shutdown()
    except Exception as e:
        print(f"Cleanup failed:\n{traceback.format_exc()}")
        print(f"Cleanup failed: {str(e)}")


if __name__ == "__main__":
    from Ayo.vis.vis_graph import visualize_dag_with_node_types

    dag = create_base_dag()

    print(dag.get_full_dag_nodes_info())

    visualize_dag_with_node_types(dag, "before_optimize_embedd_ingest_search.png")

    # optimize the dag
    from Ayo.opt_pass.pruning_dependency import PruningDependencyPass
    from Ayo.opt_pass.stage_decomposition import StageDecompositionPass

    dag.optimize([PruningDependencyPass(), StageDecompositionPass()])

    # print the dag
    print(dag.get_full_dag_nodes_info())

    visualize_dag_with_node_types(dag, "optimize_embedd_ingest_search.png")

    asyncio.run(run_app(dag))
