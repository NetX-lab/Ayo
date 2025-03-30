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
from typing import List, Any
import traceback  

# Create engine config, using the default config of ENGINE_REGISTRY
default_embed_config = ENGINE_REGISTRY.get_default_config(EngineType.EMBEDDER)
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

def create_base_dag():
    """Create base DAG"""
    base_dag = DAG(dag_id="embedding_service")
    
    # Create embedding node with correct parameters
    embedding_node = Node(
        name="Embedding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.EMBEDDER,
        io_schema=NodeIOSchema(
            input_format={"texts": List[str]},
            output_format={"embeddings": List[Any]}
        ),
        anno=NodeAnnotation.BATCHABLE,
        config={
            "batch_size": embed_config.model_config.get("max_batch_size", 256)
        }
    )
    
    # Register node to DAG
    base_dag.register_nodes(embedding_node)
    
    return base_dag

async def get_embeddings(app, texts, dag, query_id):
    try:
        # Create query
        query = Query(
            uuid=f"embed_{query_id}",
            query_id=f"embed_{query_id}",
            query_inputs={"texts": texts},
            DAG=deepcopy(dag)
        )

        # Submit query and get result
        future = await app.submit_query(
            query=query,
            timeout=embed_config.latency_profile.get("timeout", 30)
        )
        
        try:
            # Wait for result
            result = await asyncio.wait_for(
                future,
                timeout=embed_config.latency_profile.get("timeout", 30)
            )
            return result
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Query timed out after {embed_config.latency_profile.get('timeout', 30)} seconds")
            
    except Exception as e:
        print(f"Embedding processing error stack:\n{traceback.format_exc()}")
        raise Exception(f"Embedding processing failed: {str(e)}")

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
        app.register_engine(embed_config)
        dag = create_base_dag()
        app.update_template(dag)
        
        # Start application
        await app.start()
        await asyncio.sleep(5)  # Wait for system initialization
        
        # Prepare test text groups
        text_groups = [
            [
                "Artificial Intelligence is the future" *50,
                "Machine Learning is the important part of Artificial Intelligence" *50,
                "Deep Learning is the important part of Machine Learning" *50
            ]*80,
            [
                "Natural Language Processing is essential" *50,
                "Computer Vision has many applications" *50,
                "Reinforcement Learning is fascinating" *50
            ]*80
        ]
        
        # Create delayed query tasks
        async def delayed_embedding(texts, index):
            if index > 0:
                await asyncio.sleep(2 * index)
            return await get_embeddings(app, texts, dag, str(index))

        # Create all tasks
        tasks = [
            delayed_embedding(texts, i)
            for i, texts in enumerate(text_groups)
        ]
        
        # Wait for all queries to complete
        results = await asyncio.gather(*tasks)
        
        # Print results
        for i, embeddings in enumerate(results):
            print(f"Query {i} embedding result: {embeddings}")
        
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