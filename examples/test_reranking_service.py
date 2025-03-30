from Ayo.app import APP
from Ayo.dags.node import Node, NodeAnnotation, NodeType, NodeIOSchema
from Ayo.dags.dag import DAG
from Ayo.queries.query import Query
from Ayo.configs.config import EngineConfig
from Ayo.engines.engine_types import ENGINE_REGISTRY, EngineType
import time
import asyncio
import ray
from copy import deepcopy
from typing import List, Any, Dict
import traceback

# Create reranker engine config
default_rerank_config = ENGINE_REGISTRY.get_default_config(EngineType.RERANKER)
rerank_config = EngineConfig(
    name="reranker_service",
    engine_type=EngineType.RERANKER,
    resources={},
    num_gpus=1,  # GPU number for each instance
    num_cpus=1,  # CPU number for each instance
    instances=1, # Run 2 instances
    model_config={
        **default_rerank_config,  # Use default config
        "device": "cuda",         # Override specific config
        "max_batch_size": 128     # Batch size
    },
    latency_profile={     
        "timeout": 300,    # Timeout (seconds)
        "batch_wait": 0.1  # Batch wait time (seconds)
    }
)

def create_base_dag():
    """Create base DAG"""
    base_dag = DAG(dag_id="reranker_service")
    
    # Create reranker node
    reranker_node = Node(
        name="Reranker",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.RERANKER,
        io_schema=NodeIOSchema(
            input_format={
                "query": str,
                "passages": List[str]
            },
            output_format={
                "ranked_results": List[Dict[str, Any]]
            }
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
            'top_k': 5
        }
    )
    
    # Register nodes to DAG
    base_dag.register_nodes(reranker_node)
    
    return base_dag

async def get_reranking(app, query: str, passages: List[str], dag, query_id: str):
    """Get reranking results"""
    try:
        # Create query
        query = Query(
            uuid=f"rerank_{query_id}",
            query_id=f"rerank_{query_id}",
            query_inputs={
                "query": query,
                "passages": passages
            },
            DAG=deepcopy(dag)
        )

        # Submit query and get result
        future = await app.submit_query(
            query=query,
            timeout=rerank_config.latency_profile.get("timeout", 30)
        )
        
        try:
            # Wait for result
            result = await asyncio.wait_for(
                future,
                timeout=rerank_config.latency_profile.get("timeout", 30)
            )
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Query timed out after {rerank_config.latency_profile.get('timeout', 30)} seconds")
            
    except Exception as e:
        print(f"Reranking processing error stack:\n{traceback.format_exc()}")
        raise Exception(f"Reranking processing failed: {str(e)}")

async def cleanup(app):
    """Cleanup resources"""
    try:
        await app.stop()
        app.shutdown()
    except Exception as e:
        print(f"Cleanup error stack:\n{traceback.format_exc()}")
        print(f"Cleanup failed: {str(e)}")

async def main():
    try:
        # Initialize application
        app = APP.init()
        app.register_engine(rerank_config)
        dag = create_base_dag()
        app.update_template(dag)
        
        # Start application
        await app.start()
        await asyncio.sleep(5)  # Wait for system initialization
        
        # Prepare test data
        test_queries = [
            {
                "query": "What is artificial intelligence?" * 5,
                "passages": [
                    "AI is a branch of computer science..." * 50,
                    "Machine learning is a subset of AI..." * 50,
                    "Deep learning revolutionized AI..." * 50
                ] * 12
            },
            {
                "query": "How does natural language processing work?" * 5,
                "passages": [
                    "NLP combines linguistics and machine learning..." * 50,
                    "Language models are key to NLP..." * 50,
                    "Transformers architecture changed NLP..." * 50
                ] * 12
            }
        ]
        
        # Create delayed query tasks
        async def delayed_reranking(query_data, index):
            if index > 0:
                await asyncio.sleep(0 * index)
            return await get_reranking(
                app,
                query_data["query"],
                query_data["passages"],
                dag,
                str(index)
            )

        # Create all tasks
        tasks = [
            delayed_reranking(query_data, i)
            for i, query_data in enumerate(test_queries)
        ]
        
        # Wait for all queries to complete
        results = await asyncio.gather(*tasks)
        
        # Print results
        for i, ranked_results in enumerate(results):
            print(f"\nQuery {i} reranking results:")
            print(ranked_results)
        
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

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error stack:\n{traceback.format_exc()}")
        print(f"Fatal error: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()
