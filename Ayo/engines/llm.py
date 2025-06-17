import asyncio
import datetime
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import ray
import torch

from Ayo.dags.node_commons import NodeOps
from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class LLMRequest:
    """LLM request class"""

    request_id: str
    query_id: str  # the group id of the same query
    llm_internal_id: str
    prompt: str
    llm_op_type: NodeOps
    llm_partial_decoding_idx: int = -1
    sampling_params: Optional[SamplingParams] = None
    callback_ref: Any = None  # Ray ObjectRef for result callback
    timestamp: float = time.time()


@ray.remote(num_gpus=1)
class LLMEngine:
    """
    We use vLLM as the LLM engine.
    A async engine is created and wrapped by Ray.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 256,
        max_queue_size: int = 1000,
        trust_remote_code: bool = False,
        dtype: str = "auto",
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):
        # print GPU information
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of available GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU device: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")

        self.model_name = model_name
        self.max_num_seqs = max_num_seqs
        self.max_queue_size = max_queue_size

        self.name = kwargs.get("name", None)

        # initialize the vLLM asynchronous engine
        engine_args = AsyncEngineArgs(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            engine_use_ray=False,
        )

        print(f"Initializing vLLM engine with model: {model_name}")
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        time.sleep(10)

        print(f"vLLM engine initialized with model: {model_name}")

        # asynchronous queue
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)

        # track requests
        self.query_requests: Dict[str, List[LLMRequest]] = {}

        # start processing tasks
        self.running = True
        self.tasks = [asyncio.create_task(self._process_requests())]

        self.scheduler_ref = scheduler_ref

        self.result_generator_tracker = {}

        self.partial_results = {}
        self.partial_results_lock = asyncio.Lock()

    def is_ready(self):
        """Check if the engine is ready"""
        return True

    async def submit_request(
        self,
        request_id: str,
        query_id: str,
        llm_internal_id: str,
        prompt: str,
        llm_op_type: NodeOps,
        llm_partial_decoding_idx: int = -1,
        sampling_params: SamplingParams = None,
    ) -> ray.ObjectRef:
        """Submit a new LLM request"""
        request = LLMRequest(
            request_id=request_id,
            query_id=query_id,
            llm_internal_id=llm_internal_id,
            prompt=prompt,
            llm_op_type=llm_op_type,
            llm_partial_decoding_idx=llm_partial_decoding_idx,
            sampling_params=sampling_params,
        )

        if self.request_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue is full")

        await self.request_queue.put(request)

        if query_id not in self.query_requests:
            self.query_requests[query_id] = []
        self.query_requests[query_id].append(request)

    async def _process_requests(self):
        """The task of processing requests asynchronously"""
        active_tasks = set()  # Track active generation tasks

        while self.running:
            try:
                # Try to get a new request from the queue, but do not block for too long
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(), timeout=0.005
                    )

                    # Create a processing task and add it to the active task set
                    task = asyncio.create_task(
                        self._handle_request(request, request.sampling_params)
                    )
                    active_tasks.add(task)
                    task.add_done_callback(lambda t: active_tasks.remove(t))

                except asyncio.TimeoutError:
                    # The queue is temporarily empty, continue checking active tasks
                    pass

                # Wait for any completed tasks, but do not block
                if active_tasks:
                    done, _ = await asyncio.wait(
                        active_tasks, timeout=0.005, return_when=asyncio.FIRST_COMPLETED
                    )
                    # Process completed tasks (if necessary)
                    for task in done:
                        try:
                            await task
                        except Exception as e:
                            print(f"Task error: {e}")
                else:
                    # If there are no active tasks, sleep briefly to avoid CPU idle
                    await asyncio.sleep(0.005)

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in request processing loop: {e}")

    async def _handle_request(self, request, sampling_params):
        """Helper method for processing a single LLM request"""
        try:
            # Generate results - do not use await
            if request.llm_internal_id not in self.result_generator_tracker:
                self.result_generator_tracker[request.llm_internal_id] = None

            # for partial prefilling, we need to use the partial_output_pattern
            if request.llm_op_type == NodeOps.LLM_FULL_PREFILLING:
                # only need to call the add_prompt_chunk_for_existing_request
                await self.engine.add_prompt_chunk_for_existing_request(
                    request_id=request.llm_internal_id,
                    prompt_chunk_str=request.prompt,
                    is_last_chunk=True,
                )

                formatted_time = datetime.datetime.fromtimestamp(time.time()).strftime(
                    "%m-%d %H:%M:%S"
                )
                logger.info(
                    f"Time {formatted_time}, LLM engine full prefilling: {request.llm_op_type} llm_internal_id: {request.llm_internal_id}, prompt: {request.prompt}, sampling_params: {sampling_params}"
                )

                if self.scheduler_ref is not None:
                    await self.scheduler_ref.on_result.remote(
                        request.request_id, request.query_id, True
                    )

            elif request.llm_op_type in [
                NodeOps.LLM_PARTIAL_PREFILLING,
                NodeOps.LLM_PREFILLING,
            ]:

                if isinstance(self.engine, ray.actor.ActorHandle):
                    # for ray engine, we need to use remote call
                    result_generator = await self.engine.generate_with_stream.remote(
                        prompt=request.prompt,
                        sampling_params=sampling_params,
                        request_id=request.llm_internal_id,
                    )
                else:
                    # for local engine, we can directly call the method
                    result_generator = await self.engine.generate_with_stream(
                        prompt=request.prompt,
                        sampling_params=sampling_params,
                        request_id=request.llm_internal_id,
                    )

                # store the stream object for later use
                self.result_generator_tracker[request.llm_internal_id] = (
                    result_generator
                )

                # Send the result to the scheduler
                if self.scheduler_ref is not None:
                    await self.scheduler_ref.on_result.remote(
                        request.request_id, request.query_id, True
                    )

            elif request.llm_op_type == NodeOps.LLM_DECODING:

                while True:
                    result_generator = self.result_generator_tracker[
                        request.llm_internal_id
                    ]
                    if result_generator is None:
                        await asyncio.sleep(0.01)
                        continue
                    else:
                        break

                async for result in result_generator:
                    if result.finished:
                        # Create the ObjectRef for the result
                        result_ref = ray.put(result.outputs[0].text)

                        print(
                            f"Result for decoding request {request.request_id}: {result.outputs[0].text}"
                        )

                        # Send the result to the scheduler
                        if self.scheduler_ref is not None:
                            await self.scheduler_ref.on_result.remote(
                                request.request_id, request.query_id, result_ref
                            )

                        # Clean up request tracking
                        if request.query_id in self.query_requests:
                            self.query_requests[request.query_id].remove(request)
                            if not self.query_requests[request.query_id]:
                                del self.query_requests[request.query_id]

            elif request.llm_op_type == NodeOps.LLM_PARTIAL_DECODING:
                # Ensure the state dictionary exists

                while True:
                    result_generator = self.result_generator_tracker[
                        request.llm_internal_id
                    ]
                    if result_generator is None:
                        await asyncio.sleep(0.01)
                        continue
                    else:
                        break

                result_generator = self.result_generator_tracker[
                    request.llm_internal_id
                ]
                partial_decode_idx = request.llm_partial_decoding_idx

                # Use the lock to safely initialize or access the state
                async with self.partial_results_lock:
                    if request.llm_internal_id not in self.partial_results:
                        self.partial_results[request.llm_internal_id] = {
                            "results": [],
                            "is_generating": False,
                            "finished": False,
                        }

                    state = self.partial_results[request.llm_internal_id]
                    is_first_request = not state["is_generating"]

                    if is_first_request:
                        state["is_generating"] = True

                # The first request is responsible for generating
                if is_first_request:
                    first_partial_result = None
                    try:
                        async for result in result_generator:
                            if result.outputs[0].partial_outputs:
                                async with self.partial_results_lock:
                                    state["results"] = result.outputs[0].partial_outputs

                                # Process the output of the current request
                                if (
                                    len(result.outputs[0].partial_outputs)
                                    > partial_decode_idx
                                ):
                                    if self.scheduler_ref is not None:
                                        await self.scheduler_ref.on_result.remote(
                                            request.request_id,
                                            request.query_id,
                                            result.outputs[0].partial_outputs[
                                                partial_decode_idx
                                            ],
                                        )

                                    if first_partial_result is None:
                                        first_partial_result = result.outputs[
                                            0
                                        ].partial_outputs[partial_decode_idx]
                                        print(
                                            f"Time: {time.time()}, Partial output: {first_partial_result} for partial decoding idx: {partial_decode_idx}"
                                        )

                            if result.finished:
                                async with self.partial_results_lock:
                                    state["finished"] = True
                                print(f"final output: {result.outputs[0].text}")
                                del self.result_generator_tracker[
                                    request.llm_internal_id
                                ]
                                break
                    except Exception as e:
                        print(f"Partial decoding failed: {e}")
                        async with self.partial_results_lock:
                            state["finished"] = True
                            state["is_generating"] = False

                # Other requests poll for results
                else:

                    while True:
                        current_results = []
                        is_finished = False

                        async with self.partial_results_lock:
                            current_results = (
                                state["results"].copy() if state["results"] else []
                            )
                            is_finished = state["finished"]

                        if len(current_results) > partial_decode_idx:
                            if self.scheduler_ref is not None:
                                await self.scheduler_ref.on_result.remote(
                                    request.request_id,
                                    request.query_id,
                                    current_results[partial_decode_idx],
                                )
                            print(
                                f"Time: {time.time()}, Partial output: {current_results[partial_decode_idx]} for partial decoding idx: {partial_decode_idx}"
                            )
                            return

                        if is_finished:
                            # The generation is finished but the result is not available
                            if self.scheduler_ref is not None:
                                await self.scheduler_ref.on_result.remote(
                                    request.request_id, request.query_id, None
                                )
                            return

                        # Wait for a short period of time
                        await asyncio.sleep(0.02)

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error processing request {request.request_id}: {e}")

    async def shutdown(self):
        """Shutdown the service"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if isinstance(self.engine, ray.actor.ActorHandle):
            try:
                if hasattr(self.engine, "_background_loop_unshielded"):
                    await ray.get(
                        self.engine._background_loop_unshielded.cancel.remote()
                    )

                try:
                    await ray.get(self.engine.abort_all_requests.remote())
                except Exception:
                    pass

                try:
                    await ray.get(
                        self.engine.engine._run_workers_async.remote(
                            "terminate", get_all_outputs=False
                        )
                    )
                except Exception:
                    pass

                ray.kill(self.engine)
            except Exception as e:
                print(f"Error shutting down engine: {e}")
                ray.kill(self.engine)
        else:
            if hasattr(self.engine, "_background_loop_unshielded"):
                self.engine._background_loop_unshielded.cancel()

            if hasattr(self.engine, "_request_tracker"):
                for request_id in list(
                    self.engine._request_tracker._request_streams.keys()
                ):
                    self.engine._abort(request_id)

            if hasattr(self.engine, "engine") and hasattr(
                self.engine.engine, "workers"
            ):
                for worker in self.engine.engine.workers:
                    if hasattr(worker, "terminate"):
                        worker.terminate()


if __name__ == "__main__":

    llm_engine = LLMEngine.remote()

    question = "How do you believe the increasing integration of artificial intelligence into everyday life will impact social connections and interpersonal relationships over the next decade, particularly considering the balance between technological convenience and authentic human interaction in different cultural contexts?"
    question_2 = "In what ways do you think the global shift toward sustainable energy and the increasing artificial intelligence into everyday life will transform urban infrastructure, economic systems, and individual lifestyle choices over the next thirty years, and what potential challenges might emerge during this transition that could require unprecedented cooperation between governments, private industries, and local communities?"

    refine_question_number = 5

    max_tokens = int(len(question.split(" ")) * refine_question_number * 1.5)

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.95,
        max_tokens=max_tokens,
        support_partial_output=True,
        support_partial_prefilling=True,
    )
    print("max_tokens:", sampling_params.max_tokens)

    prompt = f"""Please rewrite the following question into {refine_question_number} more refined one. \
        You should keep the original meaning of the question, but make it more suitable and clear for context retrieval. \
        The original question is: {question}? \
        Please output your answer in json format. \
        It should contain {refine_question_number} new refined questions.\
        For example, if the expaned number is 3, the json output should be like this: \
        {{\
            "revised question1": "[refined question 1]",\
            "revised question2": "[refined question 2]",\
            "revised question3": "[refined question 3]"\
        }}\
        You just need to output the json string, do not output any other information or additional text!!! \
        The json output:"""

    print(f"prompt: {prompt}")

    ray.get(
        llm_engine.submit_request.remote(
            "a",
            "llm::a",
            "1a2b3c",
            prompt[: int(len(prompt) * 0.8)],
            NodeOps.LLM_PARTIAL_PREFILLING,
            -1,
            sampling_params,
        )
    )

    time.sleep(3)
    print("begin full prefilling")
    ray.get(
        llm_engine.submit_request.remote(
            "b",
            "llm::b",
            "1a2b3c",
            prompt[int(len(prompt) * 0.8) :],
            NodeOps.LLM_FULL_PREFILLING,
            -1,
            sampling_params,
        )
    )

    # ray.get(llm_engine.submit_request.remote("1", "1", "xxx", prompt, NodeOps.LLM_DECODING, -1, sampling_params))
    ray.get(
        llm_engine.submit_request.remote(
            "c",
            "llm::c",
            "1a2b3c",
            prompt,
            NodeOps.LLM_PARTIAL_DECODING,
            0,
            sampling_params,
        )
    )
    ray.get(
        llm_engine.submit_request.remote(
            "d",
            "llm::d",
            "1a2b3c",
            prompt,
            NodeOps.LLM_PARTIAL_DECODING,
            1,
            sampling_params,
        )
    )
    ray.get(
        llm_engine.submit_request.remote(
            "e",
            "llm::e",
            "1a2b3c",
            prompt,
            NodeOps.LLM_PARTIAL_DECODING,
            2,
            sampling_params,
        )
    )
    ray.get(
        llm_engine.submit_request.remote(
            "f",
            "llm::f",
            "1a2b3c",
            prompt,
            NodeOps.LLM_PARTIAL_DECODING,
            3,
            sampling_params,
        )
    )
    ray.get(
        llm_engine.submit_request.remote(
            "g",
            "llm::g",
            "1a2b3c",
            prompt,
            NodeOps.LLM_PARTIAL_DECODING,
            4,
            sampling_params,
        )
    )

    time.sleep(20)
    ray.get(llm_engine.shutdown.remote())
