import asyncio
import time
import traceback
import uuid
from copy import deepcopy

from Ayo.app import APP
from Ayo.configs.config import EngineConfig
from Ayo.dags.dag import DAG
from Ayo.dags.node import Node, NodeAnnotation, NodeIOSchema, NodeOps, NodeType
from Ayo.engines.engine_types import ENGINE_REGISTRY, EngineType
from Ayo.opt_pass.decoding_pipeling import LLMDecodingPipeliningPass
from Ayo.opt_pass.prefilling_split import PrefillingSpiltPass
from Ayo.opt_pass.pruning_dependency import PruningDependencyPass
from Ayo.opt_pass.stage_decomposition import StageDecompositionPass
from Ayo.queries.query import Query
from Ayo.utils import print_key_info

# Create LLM engine configuration
default_llm_config = ENGINE_REGISTRY.get_default_config(EngineType.LLM)
default_llm_config.update(
    {
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "max_batch_size": 8,
    }
)

llm_config = EngineConfig(
    name="llm_service",
    engine_type=EngineType.LLM,
    resources={},
    num_gpus=1,
    num_cpus=4,
    instances=1,
    model_config={**default_llm_config, "device": "cuda"},
    latency_profile={"timeout": 300, "batch_wait": 0.1},
)


def create_base_dag():
    """Create base DAG"""
    dag = DAG(dag_id="dual_llm_workflow")

    # Generate unique IDs for each LLM
    answer_llm_id = str(uuid.uuid4())
    enrich_llm_id = str(uuid.uuid4())

    # First group: answer the query
    answer_prefilling_node = Node(
        name="AnswerPrefilling",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"query": str}, output_format={"answer_prefill_state": bool}
        ),
        op_type=NodeOps.LLM_PREFILLING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "prompt_template": "Please answer the following question:\n\n{query}\n\nAnswer:",
            "prompt": "Please answer the following question:\n\n{query}\n\nAnswer:",
            "parse_json": False,
            "partial_output": False,
            "partial_prefilling": False,
            "llm_internal_id": answer_llm_id,
            "max_tokens": 512,
        },
    )

    answer_decoding_node = Node(
        name="AnswerDecoding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"answer_prefill_state": bool}, output_format={"answer": str}
        ),
        op_type=NodeOps.LLM_DECODING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "prompt_template": "Please answer the following question:\n\n{query}\n\nAnswer:",
            "prompt": "Please answer the following question:\n\n{query}\n\nAnswer:",
            "parse_json": False,
            "partial_output": False,
            "partial_prefilling": False,
            "llm_internal_id": answer_llm_id,
            "max_tokens": 512,
        },
    )

    # Second group: enrich the answer
    enrich_prefilling_node = Node(
        name="EnrichPrefilling",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"query": str, "answer": str},
            output_format={"enrich_prefill_state": bool},
        ),
        op_type=NodeOps.LLM_PREFILLING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "prompt_template": "Original question: {query}\n\nInitial answer: {answer}\n\nPlease enrich and expand the above answer, adding more details and examples:",
            "prompt": "Original question: {query}\n\nInitial answer: {answer}\n\nPlease enrich and expand the above answer, adding more details and examples:",
            "parse_json": False,
            "partial_output": False,
            "partial_prefilling": False,
            "llm_internal_id": enrich_llm_id,
            "max_tokens": 1024,
        },
    )

    enrich_decoding_node = Node(
        name="EnrichDecoding",
        node_type=NodeType.COMPUTE,
        engine_type=EngineType.LLM,
        io_schema=NodeIOSchema(
            input_format={"enrich_prefill_state": bool},
            output_format={"enriched_answer": str},
        ),
        op_type=NodeOps.LLM_DECODING,
        anno=NodeAnnotation.BATCHABLE,
        config={
            "prompt_template": "Original question: {query}\n\nInitial answer: {answer}\n\nPlease enrich and expand the above answer, adding more details and examples:",
            "prompt": "Original question: {query}\n\nInitial answer: {answer}\n\nPlease enrich and expand the above answer, adding more details and examples:",
            "parse_json": False,
            "partial_output": False,
            "partial_prefilling": False,
            "llm_internal_id": enrich_llm_id,
            "max_tokens": 1024,
        },
    )

    # Connect nodes
    (
        answer_prefilling_node
        >> answer_decoding_node
        >> enrich_prefilling_node
        >> enrich_decoding_node
    )

    # Register all nodes
    dag.register_nodes(
        answer_prefilling_node,
        answer_decoding_node,
        enrich_prefilling_node,
        enrich_decoding_node,
    )

    # Set input
    dag.set_query_inputs(
        {"query": "What is the impact of artificial intelligence on future society?"}
    )

    return dag


async def process_query(app, query_input: str, dag, query_id: str):
    """Add a query to the app and process it"""
    try:
        query = Query(
            uuid=f"dual-llm-workflow-{query_id}",
            query_id=f"dual-llm-workflow-{query_id}",
            query_inputs={"query": query_input},
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
        # Initialize the application
        app = APP.init()
        app.register_engine(llm_config)

        # Start the application
        await app.start()
        await asyncio.sleep(5)

        async def delayed_query(query_text, index):
            if index > 0:
                await asyncio.sleep(3 * index)
            return await process_query(app, query_text, dag, str(index))

        # Prepare test data
        test_queries = [
            "What is the impact of artificial intelligence on future society?",
            # "How can we balance technological advancement with ethical considerations?",
            # "What are the limitations of large language models?"
        ]

        # Create all tasks
        tasks = [
            delayed_query(query_text, i) for i, query_text in enumerate(test_queries)
        ]

        # Wait for all queries to complete
        results = await asyncio.gather(*tasks)

        # Print results
        for i, result in enumerate(results):
            print(f"\nQuery {i} results:")
            print(result)

    except Exception as e:
        print(f"Main program error stack:\n{traceback.format_exc()}")
        print(f"Main program error: {e}")
        raise
    finally:
        # Clean up resources
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
    import os

    from Ayo.vis.vis_graph import visualize_dag_with_node_types

    # Create DAG
    dag = create_base_dag()

    print("Original DAG node information:")
    print(dag.get_full_dag_nodes_info())

    visualize_dag_with_node_types(dag, "before_optimize_dual_llm_workflow.png")

    # remove the png files
    os.remove("before_optimize_dual_llm_workflow.png")

    # Optimize DAG
    begin_time = time.time()
    dag.optimize(
        [
            PruningDependencyPass(),
            StageDecompositionPass(),
            PrefillingSpiltPass(),
            LLMDecodingPipeliningPass(),
        ]
    )
    end_time = time.time()

    print_key_info(f"Optimization time: {end_time - begin_time} seconds")
    print("Optimized DAG node information:")
    print(dag.get_full_dag_nodes_info())

    visualize_dag_with_node_types(dag, "optimized_dual_llm_workflow.png")
    os.remove("optimized_dual_llm_workflow.png")

    # Run the application
    asyncio.run(run_app(dag))
