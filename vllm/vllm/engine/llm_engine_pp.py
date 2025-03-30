import copy
import queue
import threading
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple, Union

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.ray_utils import RayWorker, initialize_cluster, ray
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup,
                           SequenceGroupMetadata, SequenceGroupOutputs,
                           SequenceOutputs, SequenceStatus)
from vllm.transformers_utils.tokenizer import (detokenize_incrementally,
                                               get_tokenizer)
from vllm.utils import Counter


import multiprocessing as mp

import itertools



if ray:
    from ray.air.util.torch_dist import init_torch_dist_process_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

if TYPE_CHECKING:
    from ray.util.placement_group import PlacementGroup

logger = init_logger(__name__)

_LOGGING_INTERVAL_SEC = 5


class LLMEngine_PP:
    """An LLM engine that receives requests and generates texts.

    This is the main class for the vLLM engine. It receives requests
    from clients and generates texts from the LLM. It includes a tokenizer, a
    language model (possibly distributed across multiple GPUs), and GPU memory
    space allocated for intermediate states (aka KV cache). This class utilizes
    iteration-level scheduling and efficient memory management to maximize the
    serving throughput.

    The `LLM` class wraps this class for offline batched inference and the
    `AsyncLLMEngine` class wraps this class for online serving.

    NOTE: The config arguments are derived from the `EngineArgs` class. For the
    comprehensive list of arguments, see `EngineArgs`.

    Args:
        model_config: The configuration related to the LLM model.
        cache_config: The configuration related to the KV cache memory
            management.
        parallel_config: The configuration related to distributed execution.
        scheduler_config: The configuration related to the request scheduler.
        distributed_init_method: The initialization method for distributed
            execution. See `torch.distributed.init_process_group` for details.
        placement_group: Ray placement group for distributed execution.
            Required for distributed execution.
        log_stats: Whether to log statistics.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        logger.info(
            "Initializing an LLM engine with config: "
            f"model={model_config.model!r}, "
            f"tokenizer={model_config.tokenizer!r}, "
            f"tokenizer_mode={model_config.tokenizer_mode}, "
            f"revision={model_config.revision}, "
            f"tokenizer_revision={model_config.tokenizer_revision}, "
            f"trust_remote_code={model_config.trust_remote_code}, "
            f"dtype={model_config.dtype}, "
            f"max_seq_len={model_config.max_model_len}, "
            f"download_dir={model_config.download_dir!r}, "
            f"load_format={model_config.load_format}, "
            f"tensor_parallel_size={parallel_config.tensor_parallel_size}, "
            f"pipeline_parallel_size={parallel_config.pipeline_parallel_size}, "
            f"quantization={model_config.quantization}, "
            f"seed={model_config.seed})")
        # TODO(woosuk): Print more configs in debug mode.

        self.model_config = model_config
        self.cache_config = cache_config
        assert self.cache_config.sliding_window == getattr(
            self.model_config.hf_config, "sliding_window", None)
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.log_stats = log_stats
        self._verify_args()

        self.tokenizer = get_tokenizer(
            model_config.tokenizer,
            tokenizer_mode=model_config.tokenizer_mode,
            trust_remote_code=model_config.trust_remote_code,
            tokenizer_revision=model_config.tokenizer_revision,
            revision=model_config.revision)
        self.seq_counter = Counter()

        self.init_worker_done=False
        # Create the parallel GPU workers.
        if self.parallel_config.worker_use_ray:
            self._init_workers_ray(placement_group)
        else:
            self._init_workers(distributed_init_method)

        # Profile the memory usage and initialize the cache.

        begin= time.monotonic()
        self._init_cache()
        end=time.monotonic()
        print("init cache cost:",end-begin)

        # Create the scheduler.
        self.scheduler = Scheduler(scheduler_config, cache_config)

        # Logging.
        self.last_logging_time = 0.0
        # List of (timestamp, num_tokens)
        self.num_prompt_tokens: List[Tuple[float, int]] = []
        # List of (timestamp, num_tokens)
        self.num_generation_tokens: List[Tuple[float, int]] = []

        self.pp_size=parallel_config.pipeline_parallel_size
        self.tp_size=parallel_config.tensor_parallel_size

        self.init_worker_done=True 

        self.pp_only_mode= ( self.pp_size>1 and self.tp_size==1 )
        self.schedule_idx=0

        # self.scheduled_pipe_queue=queue.Queue()
        # self.output_pipe_queue=queue.Queue()
        
        self.timing=True

        self.async_pp=True 

        self.outstanding_execution_output_scheduled_output=[]
        self.unfinshed_sequence_groups=dict()

        self.use_threading=True
        
        
        print("original",self.workers)

        #mp.set_start_method("fork",force=True)
        if self.async_pp:
            if self.use_threading==True:
                self.scheduled_pipe_queue=queue.Queue()
                self.output_pipe_queue=queue.Queue()
                self.coordinator=threading.Thread(target=self._execute_model_pipeline)
                self.coordinator.start()
            else:
                self.scheduled_pipe_queue=mp.Queue()
                self.output_pipe_queue=mp.Queue()
                #mp.set_start_method('spawn')
                self.coordinator=mp.Process(target=execute_model_pipeline_process,args=( self.scheduled_pipe_queue,
                    self.output_pipe_queue,
                    self.workers,
                    self.parallel_config,
                    self.worker2rank))
                self.coordinator.start()

    def _init_workers(self, distributed_init_method: str):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        assert self.parallel_config.world_size == 1, (
            "Ray is required if parallel_config.world_size > 1.")

        self.workers: List[Worker] = []
        worker = Worker(
            self.model_config,
            self.parallel_config,
            self.scheduler_config,
            0,
            distributed_init_method,
        )
        self.workers.append(worker)
        
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )

        self._run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

    def _init_workers_ray(self, placement_group: "PlacementGroup",
                          **ray_remote_kwargs):
        # Lazy import the Worker to avoid importing torch.cuda/xformers
        # before CUDA_VISIBLE_DEVICES is set in the Worker
        from vllm.worker.worker import Worker

        self.workers: List[Worker] = []
        for bundle in placement_group.bundle_specs:
            if not bundle.get("GPU", 0):
                continue
            worker = ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_capture_child_tasks=True),
                **ray_remote_kwargs,
            )(RayWorker).remote(self.model_config.trust_remote_code)
            self.workers.append(worker)

        # Initialize torch distributed process group for the workers.
        init_torch_dist_process_group(self.workers, backend="nccl")
        model_config = copy.deepcopy(self.model_config)
        parallel_config = copy.deepcopy(self.parallel_config)
        scheduler_config = copy.deepcopy(self.scheduler_config)
        self._run_workers("init_worker",
                          get_all_outputs=True,
                          worker_init_fn=lambda: Worker(
                              model_config,
                              parallel_config,
                              scheduler_config,
                              None,
                              None,
                          ))
        self._run_workers(
            "init_model",
            get_all_outputs=True,
        )
        self._run_workers(
            "load_model",
            get_all_outputs=True,
            max_concurrent_workers=self.parallel_config.
            max_parallel_loading_workers,
        )

        self.rank2worker=dict()
        self.worker2rank=dict()
        for worker in self.workers:
            #print(worker.)
            rank=ray.get(worker.__getattr__.remote("rank"))

            self.rank2worker[rank]=worker
            self.worker2rank[worker]=rank 


    def _verify_args(self) -> None:
        self.model_config.verify_with_parallel_config(self.parallel_config)
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    def _init_cache(self) -> None:
        """Profiles the memory usage and initializes the KV cache."""
        # Get the maximum number of blocks that can be allocated on GPU and CPU.
        num_blocks = self._run_workers(
            "profile_num_available_blocks",
            get_all_outputs=True,
            block_size=self.cache_config.block_size,
            gpu_memory_utilization=self.cache_config.gpu_memory_utilization,
            cpu_swap_space=self.cache_config.swap_space_bytes,
        )

        # Since we use a shared centralized controller, we take the minimum
        # number of blocks across all workers to make sure all the memory
        # operators can be applied to all workers.
        num_gpu_blocks = min(b[0] for b in num_blocks)
        num_cpu_blocks = min(b[1] for b in num_blocks)
        # FIXME(woosuk): Change to debug log.
        logger.info(f"# GPU blocks: {num_gpu_blocks}, "
                    f"# CPU blocks: {num_cpu_blocks}")

        if num_gpu_blocks <= 0:
            raise ValueError("No available memory for the cache blocks. "
                             "Try increasing `gpu_memory_utilization` when "
                             "initializing the engine.")

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

        # Initialize the cache.
        print("init cache")
        self._run_workers("init_cache_engine", cache_config=self.cache_config)

    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> "LLMEngine_PP":
        """Creates an LLM engine from the engine arguments."""
        # Create the engine configs.
        engine_configs = engine_args.create_engine_configs()
        parallel_config = engine_configs[2]
        # Initialize the cluster.
        distributed_init_method, placement_group = initialize_cluster(
            parallel_config)
        # Create the LLM engine.
        engine = cls(*engine_configs,
                     distributed_init_method,
                     placement_group,
                     log_stats=not engine_args.disable_log_stats)
        return engine

    def add_request(
        self,
        request_id: str,
        prompt: Optional[str],
        sampling_params: SamplingParams,
        prompt_token_ids: Optional[List[int]] = None,
        arrival_time: Optional[float] = None,
    ) -> None:
        """Add a request to the engine's request pool.

        The request is added to the request pool and will be processed by the
        scheduler as `engine.step()` is called. The exact scheduling policy is
        determined by the scheduler.

        Args:
            request_id: The unique ID of the request.
            prompt: The prompt string. Can be None if prompt_token_ids is
                provided.
            sampling_params: The sampling parameters for text generation.
            prompt_token_ids: The token IDs of the prompt. If None, we
                use the tokenizer to convert the prompts to token IDs.
            arrival_time: The arrival time of the request. If None, we use
                the current monotonic time.
        """
        if arrival_time is None:
            arrival_time = time.monotonic()
        if prompt_token_ids is None:
            assert prompt is not None
            prompt_token_ids = self.tokenizer.encode(prompt)

        # Create the sequences.
        block_size = self.cache_config.block_size
        seq_id = next(self.seq_counter)
        seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

        # Create the sequence group.
        seq_group = SequenceGroup(request_id, [seq], sampling_params,
                                  arrival_time)

        # Add the sequence group to the scheduler.
        self.scheduler.add_seq_group(seq_group)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        """Aborts a request(s) with the given ID.

        Args:
            request_id: The ID(s) of the request to abort.
        """
        self.scheduler.abort_seq_group(request_id)

    def get_model_config(self) -> ModelConfig:
        """Gets the model configuration."""
        return self.model_config

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_seq_groups()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_seqs()

    def _schedule(
        self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[RequestOutput]]:
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list, scheduler_outputs, [
            RequestOutput.from_seq_group(seq_group)
            for seq_group in scheduler_outputs.ignored_seq_groups
        ]

    def _schedule_pp(
            self
    ) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs,
               List[RequestOutput]]:
               
        seq_group_metadata_list, scheduler_outputs = self.scheduler.schedule()
        return seq_group_metadata_list,scheduler_outputs

    def _check_beam_search_early_stopping(
        self,
        early_stopping: Union[bool, str],
        sampling_params: SamplingParams,
        best_running_seq: Sequence,
        current_worst_seq: Sequence,
    ) -> bool:
        assert sampling_params.use_beam_search
        length_penalty = sampling_params.length_penalty
        if early_stopping is True:
            return True

        current_worst_score = (current_worst_seq.get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id))
        if early_stopping is False:
            highest_attainable_score = (best_running_seq.get_beam_search_score(
                length_penalty=length_penalty,
                eos_token_id=self.tokenizer.eos_token_id))
        else:
            assert early_stopping == "never"
            if length_penalty > 0.0:
                # If length_penalty > 0.0, beam search will prefer longer
                # sequences. The highest attainable score calculation is
                # based on the longest possible sequence length in this case.
                max_possible_length = max(
                    best_running_seq.get_prompt_len() +
                    sampling_params.max_tokens,
                    self.scheduler_config.max_model_len)
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id,
                        seq_len=max_possible_length))
            else:
                # Otherwise, beam search will prefer shorter sequences. The
                # highest attainable score calculation is based on the current
                # sequence length.
                highest_attainable_score = (
                    best_running_seq.get_beam_search_score(
                        length_penalty=length_penalty,
                        eos_token_id=self.tokenizer.eos_token_id))
        return current_worst_score >= highest_attainable_score

    def _process_sequence_group_outputs(self, seq_group: SequenceGroup,
                                        outputs: SequenceGroupOutputs) -> None:
        # Process prompt logprobs
        prompt_logprobs = outputs.prompt_logprobs
        if prompt_logprobs is not None:
            seq_group.prompt_logprobs = prompt_logprobs

        # Process samples
        samples = outputs.samples
        parent_seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        existing_finished_seqs = seq_group.get_finished_seqs()
        parent_child_dict = {
            parent_seq.seq_id: []
            for parent_seq in parent_seqs
        }
        for sample in samples:
            parent_child_dict[sample.parent_seq_id].append(sample)
        # List of (child, parent)
        child_seqs: List[Tuple[Sequence, Sequence]] = []

        # Process the child samples for each parent sequence
        for parent in parent_seqs:
            child_samples: List[SequenceOutputs] = parent_child_dict[
                parent.seq_id]
            if len(child_samples) == 0:
                # This parent sequence has no children samples. Remove
                # the parent sequence from the sequence group since it will
                # not be used in the future iterations.
                parent.status = SequenceStatus.FINISHED_ABORTED
                seq_group.remove(parent.seq_id)
                self.scheduler.free_seq(parent)
                continue
            # Fork the parent sequence if there are multiple child samples.
            for child_sample in child_samples[:-1]:
                new_child_seq_id = next(self.seq_counter)
                child = parent.fork(new_child_seq_id)
                child.append_token_id(child_sample.output_token,
                                      child_sample.logprobs)
                child_seqs.append((child, parent))
            # Continue the parent sequence for the last child sample.
            # We reuse the parent sequence here to reduce redundant memory
            # copies, especially when using non-beam search sampling methods.
            last_child_sample = child_samples[-1]
            parent.append_token_id(last_child_sample.output_token,
                                   last_child_sample.logprobs)
            child_seqs.append((parent, parent))

        for seq, _ in child_seqs:
            self._decode_sequence(seq, seq_group.sampling_params)
            self._check_stop(seq, seq_group.sampling_params)

        # Non-beam search case
        if not seq_group.sampling_params.use_beam_search:
            # For newly created child sequences, add them to the sequence group
            # and fork them in block manager if they are not finished.
            for seq, parent in child_seqs:
                if seq is not parent:
                    seq_group.add(seq)
                    if not seq.is_finished():
                        self.scheduler.fork_seq(parent, seq)

            # Free the finished and selected parent sequences' memory in block
            # manager. Keep them in the sequence group as candidate output.
            # NOTE: we need to fork the new sequences before freeing the
            # old sequences.
            for seq, parent in child_seqs:
                if seq is parent and seq.is_finished():
                    self.scheduler.free_seq(seq)
            return

        # Beam search case
        # Select the child sequences to keep in the sequence group.
        selected_child_seqs = []
        unselected_child_seqs = []
        beam_width = seq_group.sampling_params.best_of
        length_penalty = seq_group.sampling_params.length_penalty

        # Select the newly finished sequences with the highest scores
        # to replace existing finished sequences.
        # Tuple of (seq, parent, is_new)
        existing_finished_seqs = [(seq, None, False)
                                  for seq in existing_finished_seqs]
        new_finished_seqs = [(seq, parent, True) for seq, parent in child_seqs
                             if seq.is_finished()]
        all_finished_seqs = existing_finished_seqs + new_finished_seqs
        # Sort the finished sequences by their scores.
        all_finished_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                               reverse=True)
        for seq, parent, is_new in all_finished_seqs[:beam_width]:
            if is_new:
                # A newly generated child sequence finishes and has a high
                # score, so we will add it into the sequence group.
                selected_child_seqs.append((seq, parent))
        for seq, parent, is_new in all_finished_seqs[beam_width:]:
            if is_new:
                # A newly generated child sequence finishes but has a low
                # score, so we will not add it into the sequence group.
                # Additionally, if this sequence is a continuation of a
                # parent sequence, we will need remove the parent sequence
                # from the sequence group.
                unselected_child_seqs.append((seq, parent))
            else:
                # An existing finished sequence has a low score, so we will
                # remove it from the sequence group.
                seq_group.remove(seq.seq_id)

        # select the top beam_width sequences from the running
        # sequences for the next iteration to continue the beam
        # search.
        running_child_seqs = [(seq, parent) for seq, parent in child_seqs
                              if not seq.is_finished()]
        # Sort the running sequences by their scores.
        running_child_seqs.sort(key=lambda x: x[0].get_beam_search_score(
            length_penalty=length_penalty,
            eos_token_id=self.tokenizer.eos_token_id),
                                reverse=True)

        # Check if we can stop the beam search.
        if len(running_child_seqs) == 0:
            # No running sequences, stop the beam search.
            stop_beam_search = True
        elif len(all_finished_seqs) < beam_width:
            # Not enough finished sequences, continue the beam search.
            stop_beam_search = False
        else:
            # Check the early stopping criteria
            best_running_seq = running_child_seqs[0][0]
            current_worst_seq = all_finished_seqs[beam_width - 1][0]
            stop_beam_search = self._check_beam_search_early_stopping(
                seq_group.sampling_params.early_stopping,
                seq_group.sampling_params, best_running_seq, current_worst_seq)

        if stop_beam_search:
            # Stop the beam search and remove all the running sequences from
            # the sequence group.
            unselected_child_seqs.extend(running_child_seqs)
        else:
            # Continue the beam search and select the top beam_width sequences
            # to continue the beam search.
            selected_child_seqs.extend(running_child_seqs[:beam_width])
            # The remaining running sequences will not be used in the next
            # iteration. Again, if these sequences are continuations of
            # parent sequences, we will need to remove the parent sequences
            # from the sequence group.
            unselected_child_seqs.extend(running_child_seqs[beam_width:])

        # For newly created child sequences, add them to the sequence group
        # and fork them in block manager if they are not finished.
        for seq, parent in selected_child_seqs:
            if seq is not parent:
                seq_group.add(seq)
                if not seq.is_finished():
                    self.scheduler.fork_seq(parent, seq)

        # Free the finished and selected parent sequences' memory in block
        # manager. Keep them in the sequence group as candidate output.
        for seq, parent in selected_child_seqs:
            if seq is parent and seq.is_finished():
                self.scheduler.free_seq(seq)

        # Remove the unselected parent sequences from the sequence group and
        # free their memory in block manager.
        for seq, parent in unselected_child_seqs:
            if seq is parent:
                # Remove the parent sequence if it is not selected for next
                # iteration
                seq_group.remove(seq.seq_id)
                self.scheduler.free_seq(seq)

    def _process_model_outputs(
            self, output: SamplerOutput,
            scheduler_outputs: SchedulerOutputs,
            real_scheduled_groups=None) -> List[RequestOutput]:
        # Update the scheduled sequence groups with the model outputs.
        scheduled_seq_groups = scheduler_outputs.scheduled_seq_groups

        if real_scheduled_groups!=None:
            scheduled_seq_groups=real_scheduled_groups
        #print("process model output:",output)

        #print("process output:",scheduled_seq_groups)

        for seq_group, outputs in zip(scheduled_seq_groups, output):
            #print("seq group:",seq_group)
            self._process_sequence_group_outputs(seq_group, outputs)
            #print("seq group is finished",seq_group.is_finished())

        # Free the finished sequence groups.
        self.scheduler.free_finished_seq_groups()

        # Create the outputs.
        request_outputs: List[RequestOutput] = []
        for seq_group in (scheduled_seq_groups +
                          scheduler_outputs.ignored_seq_groups):
            request_output = RequestOutput.from_seq_group(seq_group)
            request_outputs.append(request_output)

        # if self.log_stats:
        #     # Log the system stats.
        #     self._log_system_stats(scheduler_outputs.prompt_run,
        #                            scheduler_outputs.num_batched_tokens)
        return request_outputs

    def step(self) -> List[RequestOutput]:
        """Performs one decoding iteration and returns newly generated results.

        This function performs one decoding iteration of the engine. It first
        schedules the sequences to be executed in the next iteration and the
        token blocks to be swapped in/out/copy. Then, it executes the model
        and updates the scheduler with the model outputs. Finally, it decodes
        the sequences and returns the newly generated results.
        """

        duplicated_rid=[]
        if self.async_pp==False:
            seq_group_metadata_list, scheduler_outputs, ignored = self._schedule()
        else:
            seq_group_metadata_list, scheduler_outputs = self._schedule_pp()
            
            # print(seq_group_metadata_list)
            # print("-"*20)
            # print(scheduler_outputs)
            for meta in seq_group_metadata_list[::-1]:
                # print("meta rid",meta.request_id)
                # print("unfinished:",self.unfinshed_sequence_groups.keys())
                if meta.request_id in self.unfinshed_sequence_groups:
                    seq_group_metadata_list.remove(meta)
                    duplicated_rid.append(meta.request_id)

        
            # print("*"*20)
            # print("scheduled result:",seq_group_metadata_list)
            # print("scheduled rid:",[x.request_id for x in seq_group_metadata_list])
            # print("duplicated rid:",duplicated_rid)
            # print("*"*20) 
        #print("seq_group_metadata_list",seq_group_metadata_list)
        #print("scheduler_outputs",scheduler_outputs)

        

        if scheduler_outputs.is_empty() and self.async_pp==False:
            return ignored

        #print("scheduled seq_group_metadata_list",seq_group_metadata_list)
        # Execute the model.

        if self.pp_only_mode==True and self.async_pp==False:
            output = self._run_workers(
                "execute_model_pp_only",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                get_all_outputs=True,
            )

        elif self.pp_only_mode==True and self.async_pp==True:
            if len(seq_group_metadata_list)!=0:
                self.scheduled_pipe_queue.put((seq_group_metadata_list,scheduler_outputs.blocks_to_swap_in,scheduler_outputs.blocks_to_swap_out,scheduler_outputs.blocks_to_copy))
        else:
            output = self._run_workers(
                "execute_model",
                seq_group_metadata_list=seq_group_metadata_list,
                blocks_to_swap_in=scheduler_outputs.blocks_to_swap_in,
                blocks_to_swap_out=scheduler_outputs.blocks_to_swap_out,
                blocks_to_copy=scheduler_outputs.blocks_to_copy,
                get_all_outputs=True,
            )

        if self.async_pp == False:
            if isinstance(output[0],SequenceGroupOutputs):
                output=output
            else:
                output = output[-1]  # The last pipeline stage returns the output.

            return self._process_model_outputs(output, scheduler_outputs)
        
        else:
            
            output=[]
            real_scheduler_groups=[]
            candidate:List[SequenceGroup]=list(self.unfinshed_sequence_groups.values())+[x for x in scheduler_outputs.scheduled_seq_groups if x.request_id not in duplicated_rid]
            candidate={seq_group.request_id:seq_group for seq_group in candidate}

            size=self.output_pipe_queue.qsize()
            count=0

            while(count<size):
                rid,res_obj,t=self.output_pipe_queue.get()
                # print("geting rid:",rid)
                # res,finish_time=ray.get(res_obj)


                #real_scheduler_groups.append(candidate[rid])
                # print("request id ",rid,res[0],"submit time:",t)
                #print("one iteration time cost:",finish_time-t)
                # output.append(res[0])
                # del candidate[rid]
                ready_refs, remaining_refs = ray.wait([res_obj], num_returns=1, timeout=0.001)
                if len(ready_refs)>0:
                    res,finish_time=ray.get(ready_refs[0])
                    if isinstance(rid,list):
                        for idx,result in zip(rid,res):
                            output.append(result)
                            real_scheduler_groups.append(candidate[idx])
                            print("request id ",idx,"submit time:",t)
                            print("one iteration time cost:",finish_time-t)
                            del candidate[idx]
                    else:
                        output.append(res[0])
                        real_scheduler_groups.append(candidate[rid])
                        print("request id ",rid,res[0],"submit time:",t)
                        print("one iteration time cost:",finish_time-t)
                        del candidate[rid]
                else:
                    self.output_pipe_queue.put((rid,res_obj,t))

                count+=1
                
            self.unfinshed_sequence_groups=candidate
            
            # print("unfinished squence groups:",[x for x in self.unfinshed_sequence_groups])
            # print("real_scheduler_groups:",real_scheduler_groups)
            scheduler_outputs.scheduled_seq_groups=real_scheduler_groups
            if len(output)==0:
                return []
            
            else:
                return self._process_model_outputs(output, scheduler_outputs,real_scheduler_groups)

    def _log_system_stats(
        self,
        prompt_run: bool,
        num_batched_tokens: int,
    ) -> None:
        now = time.monotonic()
        # Log the number of batched input tokens.
        if prompt_run:
            self.num_prompt_tokens.append((now, num_batched_tokens))
        else:
            self.num_generation_tokens.append((now, num_batched_tokens))

        elapsed_time = now - self.last_logging_time
        if elapsed_time < _LOGGING_INTERVAL_SEC:
            return

        # Discard the old stats.
        self.num_prompt_tokens = [(t, n) for t, n in self.num_prompt_tokens
                                  if now - t < _LOGGING_INTERVAL_SEC]
        self.num_generation_tokens = [(t, n)
                                      for t, n in self.num_generation_tokens
                                      if now - t < _LOGGING_INTERVAL_SEC]

        if len(self.num_prompt_tokens) > 1:
            total_num_tokens = sum(n for _, n in self.num_prompt_tokens[:-1])
            window = now - self.num_prompt_tokens[0][0]
            avg_prompt_throughput = total_num_tokens / window
        else:
            avg_prompt_throughput = 0.0
        if len(self.num_generation_tokens) > 1:
            total_num_tokens = sum(n
                                   for _, n in self.num_generation_tokens[:-1])
            window = now - self.num_generation_tokens[0][0]
            avg_generation_throughput = total_num_tokens / window
        else:
            avg_generation_throughput = 0.0

        total_num_gpu_blocks = self.cache_config.num_gpu_blocks
        num_free_gpu_blocks = (
            self.scheduler.block_manager.get_num_free_gpu_blocks())
        num_used_gpu_blocks = total_num_gpu_blocks - num_free_gpu_blocks
        gpu_cache_usage = num_used_gpu_blocks / total_num_gpu_blocks

        total_num_cpu_blocks = self.cache_config.num_cpu_blocks
        if total_num_cpu_blocks > 0:
            num_free_cpu_blocks = (
                self.scheduler.block_manager.get_num_free_cpu_blocks())
            num_used_cpu_blocks = total_num_cpu_blocks - num_free_cpu_blocks
            cpu_cache_usage = num_used_cpu_blocks / total_num_cpu_blocks
        else:
            cpu_cache_usage = 0.0

        logger.info("Avg prompt throughput: "
                    f"{avg_prompt_throughput:.1f} tokens/s, "
                    "Avg generation throughput: "
                    f"{avg_generation_throughput:.1f} tokens/s, "
                    f"Running: {len(self.scheduler.running)} reqs, "
                    f"Swapped: {len(self.scheduler.swapped)} reqs, "
                    f"Pending: {len(self.scheduler.waiting)} reqs, "
                    f"GPU KV cache usage: {gpu_cache_usage * 100:.1f}%, "
                    f"CPU KV cache usage: {cpu_cache_usage * 100:.1f}%")
        self.last_logging_time = now

    def _decode_sequence(self, seq: Sequence, prms: SamplingParams) -> None:
        """Decodes the new token for a sequence."""
        (new_tokens, new_output_text, prefix_offset,
         read_offset) = detokenize_incrementally(
             self.tokenizer,
             all_input_ids=seq.get_token_ids(),
             prev_tokens=seq.tokens,
             prefix_offset=seq.prefix_offset,
             read_offset=seq.read_offset,
             skip_special_tokens=prms.skip_special_tokens,
             spaces_between_special_tokens=prms.spaces_between_special_tokens,
         )
        if seq.tokens is None:
            seq.tokens = new_tokens
        else:
            seq.tokens.extend(new_tokens)
        seq.prefix_offset = prefix_offset
        seq.read_offset = read_offset
        seq.output_text += new_output_text

    def _check_stop(self, seq: Sequence,
                    sampling_params: SamplingParams) -> None:
        """Stop the finished sequences."""
        for stop_str in sampling_params.stop:
            if seq.output_text.endswith(stop_str):
                # Truncate the output text so that the stop string is
                # not included in the output.
                seq.output_text = seq.output_text[:-len(stop_str)]
                seq.status = SequenceStatus.FINISHED_STOPPED
                return
        if seq.get_last_token_id() in sampling_params.stop_token_ids:
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if the sequence has reached max_model_len.
        if seq.get_len() > self.scheduler_config.max_model_len:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has reached max_tokens.
        if seq.get_output_len() == sampling_params.max_tokens:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return

        # Check if the sequence has generated the EOS token.
        if ((not sampling_params.ignore_eos)
                and seq.get_last_token_id() == self.tokenizer.eos_token_id):
            seq.status = SequenceStatus.FINISHED_STOPPED
            return

    def _run_workers_in_batch(
        self,
        workers,
        method: str,
        *args,
        **kwargs,
    ):
        all_outputs = []
        for worker in workers:
            if self.parallel_config.worker_use_ray:
                #print("use ray")
                executor = partial(worker.execute_method.remote, method)
            else:
                executor = getattr(worker, method)

            output = executor(*args, **kwargs)
            all_outputs.append(output)

        begin=time.monotonic()
        if self.parallel_config.worker_use_ray:
            all_outputs = ray.get(all_outputs)
        print("batch execution cost:",time.monotonic()-begin)

        return all_outputs

    def _run_workers(
        self,
        method: str,
        *args,
        get_all_outputs: bool = False,
        max_concurrent_workers: Optional[int] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers."""
        all_outputs = []
        if max_concurrent_workers:
            work_groups = [
                self.workers[i:i + max_concurrent_workers]
                for i in range(0, len(self.workers), max_concurrent_workers)
            ]
        else:
            work_groups = [self.workers]

        # print("work groups:",work_groups)
        # print("# worker groups",len(work_groups))

        # print("args",args)
        # print("kwargs",kwargs)

        if  method!="execute_model_pp_only":
            for workers in work_groups:
                all_outputs.extend(
                    self._run_workers_in_batch(workers, method, *args, **kwargs))

            if get_all_outputs:
                return all_outputs

            # Make sure all workers have the same results.
            output = all_outputs[0]
            for other_output in all_outputs[1:]:
                assert output == other_output
            return output
        
        else:
            seq_group_metadata_list=kwargs.get("seq_group_metadata_list")

            all_seq_group_state=[metadata.is_prompt for metadata in seq_group_metadata_list]

            ### no prompts in the scheduled requests
            if all(all_seq_group_state)==False:
                #print("not prompt")
                outputs=self._run_workers_in_batch(self.workers, 'execute_model', *args, **kwargs)

                return outputs

            else:
                seq_group_metadata_list=kwargs.pop("seq_group_metadata_list")

                last_pp_outputs=[]

                world_size=self.parallel_config.world_size  

                begin=time.monotonic()

                #print(self.worker2rank)

                for i,seq_group_metadata in enumerate(seq_group_metadata_list):
                    #print(i,seq_group_metadata)
                    print("schedule id",i)
                    for worker in self.workers:
                        if self.parallel_config.worker_use_ray:
                            #print("use ray")
                            pretecher=partial(worker.execute_method.remote, "fetech_input_activation")
                            executor = partial(worker.execute_method.remote, "execute_model_pp_only")
                            sender=partial(worker.execute_method.remote, "send_hidden_states")
                            origin= partial(worker.execute_method.remote, "execute_model")
                        else:
                            executor = getattr(worker, method)


                        prefetch_res=pretecher(seq_group_metadata_list=[seq_group_metadata],schedule_idx=self.schedule_idx, **kwargs)
                        exec_res = executor(self.schedule_idx)
                        send_res = sender(self.schedule_idx)

                        #send_res=origin(seq_group_metadata_list=[seq_group_metadata], **kwargs)
                        if self.worker2rank[worker]==world_size-1:
                            #print("here")
                            last_pp_outputs.append(send_res)

                    self.schedule_idx+=1

                end=time.monotonic()

                #print("cost:",end-begin)
                last_pp_outputs = ray.get(last_pp_outputs) 
                print("one batch cost:",time.monotonic()-end)


                outputs=list(itertools.chain.from_iterable(last_pp_outputs))

                #print(outputs)


                return outputs


    def _execute_model_pipeline(
        self,
        #method: str,
        # *args,
        # get_all_outputs: bool = False,
        # max_concurrent_workers: Optional[int] = None,
        # **kwargs,
        scheduled_pipe_queue=None,
        output_pipe_queue=None
    ):
        
        schedule_idx=0
        
        if self.use_threading==True:
            scheduled_pipe_queue=self.scheduled_pipe_queue
            output_pipe_queue=self.output_pipe_queue

        while(True):
            
            kwargs=scheduled_pipe_queue.get()

            print("worker get kwargs:",kwargs)

            seq_group_metadata_list,blocks_to_swap_in,blocks_to_swap_out,blocks_to_copy=kwargs

            #print(seq_group_metadata_list)
            all_seq_group_state=[metadata.is_prompt for metadata in seq_group_metadata_list]
            request_ids=[metadata.request_id for metadata in seq_group_metadata_list]
            begin=time.monotonic()
            ### no prompts in the scheduled requests
            if all(all_seq_group_state)==False:
                print("not prompt","seq group size:",len(all_seq_group_state))

                last_pp_outputs=[]
                world_size=self.parallel_config.world_size  
                begin=time.monotonic()
                 
                for worker in self.workers:
                    assert self.parallel_config.worker_use_ray==True
                        #print("use ray")
                    pretecher=partial(worker.execute_method.remote, "fetech_input_activation")
                    executor = partial(worker.execute_method.remote, "execute_model_pp_only")
                    sender=partial(worker.execute_method.remote, "send_hidden_states")
                    origin= partial(worker.execute_method.remote, "execute_model")
                    if isinstance(seq_group_metadata_list,SequenceGroupMetadata):
                        seq_group_metadatas=[seq_group_metadata_list]
                    else:
                        seq_group_metadatas=seq_group_metadata_list
                    prefetch_res=pretecher(seq_group_metadata_list=seq_group_metadatas,schedule_idx=schedule_idx, 
                                           blocks_to_swap_in=blocks_to_swap_in,
                                           blocks_to_swap_out=blocks_to_swap_out,
                                           blocks_to_copy=blocks_to_copy)
                    exec_res = executor(schedule_idx)
                    send_res = sender(schedule_idx)

                    if self.worker2rank[worker]==world_size-1:
                        #print("here")
                        if self.timing==True:
                            seq_group_metadata:SequenceGroupMetadata
                            last_pp_outputs.append((request_ids,send_res,begin))
                        else:
                            last_pp_outputs.append(send_res)
                schedule_idx+=1

                for output in last_pp_outputs:
                    print(output[0])
                    self.output_pipe_queue.put(output)

            else:
                #seq_group_metadata_list=kwargs.pop("seq_group_metadata_list")
                last_pp_outputs=[]
                world_size=self.parallel_config.world_size  

                #print(self.worker2rank)
                for i,seq_group_metadata in enumerate(seq_group_metadata_list):
                    #print(i,seq_group_metadata)
                    print("schedule id",i)
                    for worker in self.workers:

                        assert self.parallel_config.worker_use_ray==True

                            #print("use ray")
                        pretecher=partial(worker.execute_method.remote, "fetech_input_activation")
                        executor = partial(worker.execute_method.remote, "execute_model_pp_only")
                        sender=partial(worker.execute_method.remote, "send_hidden_states")
                        origin= partial(worker.execute_method.remote, "execute_model")

                        if isinstance(seq_group_metadata,SequenceGroupMetadata):
                            seq_group_metadatas=[seq_group_metadata]
                        else:
                            seq_group_metadatas=seq_group_metadata

                        prefetch_res=pretecher(seq_group_metadata_list=seq_group_metadatas,schedule_idx=schedule_idx, 
                                               blocks_to_swap_in=blocks_to_swap_in,
                                               blocks_to_swap_out=blocks_to_swap_out,
                                               blocks_to_copy=blocks_to_copy)
                        exec_res = executor(schedule_idx)
                        send_res = sender(schedule_idx)
                        #ray.get(send_res)
                        #send_res=origin(seq_group_metadata_list=[seq_group_metadata], **kwargs)
                        if self.worker2rank[worker]==world_size-1:
                            #print("here")
                            if self.timing==True:
                                seq_group_metadata:SequenceGroupMetadata
                                last_pp_outputs.append((seq_group_metadata.request_id,send_res,begin))
                            else:
                                last_pp_outputs.append(send_res)
                    schedule_idx+=1
                end=time.monotonic()
                print("cost:",end-begin)
                if self.timing:
                    print("timing")
                    # for x in last_pp_outputs:
                    #      ray.get(x[1])
                    #      print("request latency:",time.monotonic()-x[2])
                else:
                    ray.get(last_pp_outputs)

                #print("one batch cost:",time.monotonic()-end)
                #outputs=list(itertools.chain.from_iterable(last_pp_outputs))

                #print(outputs)
                for output in last_pp_outputs:
                    output_pipe_queue.put(output)
                #return outputs 
                
                
                
def execute_model_pipeline_process(
        scheduled_pipe_queue,
        output_pipe_queue,
        workers,
        parallel_config,
        worker2rank
        
    ):
        ray.init()
        schedule_idx=0
        
        print()
       
        while(True):
            
            kwargs=scheduled_pipe_queue.get()

            print("worker get kwargs:",kwargs)

            seq_group_metadata_list,blocks_to_swap_in,blocks_to_swap_out,blocks_to_copy=kwargs
            begin=time.monotonic()
            #print(seq_group_metadata_list)
            all_seq_group_state=[metadata.is_prompt for metadata in seq_group_metadata_list]
            print("exec",workers)
            ### no prompts in the scheduled requests
            if all(all_seq_group_state)==False:
                print("not prompt")

                last_pp_outputs=[]
                world_size=parallel_config.world_size  
                begin=time.monotonic()
                 
                for worker in workers:
                    assert parallel_config.worker_use_ray==True
                        #print("use ray")
                    pretecher=partial(worker.execute_method.remote, "fetech_input_activation")
                    executor = partial(worker.execute_method.remote, "execute_model_pp_only")
                    sender=partial(worker.execute_method.remote, "send_hidden_states")
                    origin= partial(worker.execute_method.remote, "execute_model")
                    if isinstance(seq_group_metadata,SequenceGroupMetadata):
                        seq_group_metadatas=[seq_group_metadata]
                    else:
                        seq_group_metadatas=seq_group_metadata
                    prefetch_res=pretecher(seq_group_metadata_list=seq_group_metadatas,schedule_idx=self.schedule_idx, 
                                           blocks_to_swap_in=blocks_to_swap_in,
                                           blocks_to_swap_out=blocks_to_swap_out,
                                           blocks_to_copy=blocks_to_copy)
                    exec_res = executor(schedule_idx)
                    send_res = sender(schedule_idx)

                    if worker2rank[worker]==world_size-1:
                        #print("here")
                        if timing==True:
                            seq_group_metadata:SequenceGroupMetadata
                            last_pp_outputs.append((seq_group_metadata.request_id,send_res,begin))
                        else:
                            last_pp_outputs.append(send_res)
                schedule_idx+=1

                for output in last_pp_outputs:
                    output_pipe_queue.put(output)

            else:
                #seq_group_metadata_list=kwargs.pop("seq_group_metadata_list")
                last_pp_outputs=[]
                world_size=parallel_config.world_size  

                #print(self.worker2rank)
                for i,seq_group_metadata in enumerate(seq_group_metadata_list):
                    #print(i,seq_group_metadata)
                    print("schedule id",i)
                    for worker in workers:

                        assert parallel_config.worker_use_ray==True

                            #print("use ray")
                        pretecher=partial(worker.execute_method.remote, "fetech_input_activation")
                        executor = partial(worker.execute_method.remote, "execute_model_pp_only")
                        sender=partial(worker.execute_method.remote, "send_hidden_states")
                        origin= partial(worker.execute_method.remote, "execute_model")

                        if isinstance(seq_group_metadata,SequenceGroupMetadata):
                            seq_group_metadatas=[seq_group_metadata]
                        else:
                            seq_group_metadatas=seq_group_metadata

                        prefetch_res=pretecher(seq_group_metadata_list=seq_group_metadatas,schedule_idx=schedule_idx, 
                                               blocks_to_swap_in=blocks_to_swap_in,
                                               blocks_to_swap_out=blocks_to_swap_out,
                                               blocks_to_copy=blocks_to_copy)
                        exec_res = executor(schedule_idx)
                        send_res = sender(schedule_idx)
                        #ray.get(send_res)
                        #send_res=origin(seq_group_metadata_list=[seq_group_metadata], **kwargs)
                        if worker2rank[worker]==world_size-1:
                            #print("here")
                            if True:
                                seq_group_metadata:SequenceGroupMetadata
                                last_pp_outputs.append((seq_group_metadata.request_id,send_res,begin))
                            else:
                                last_pp_outputs.append(send_res)
                    schedule_idx+=1
                end=time.monotonic()
                print("cost:",end-begin)
                if True:
                    print("timing")
                    for x in last_pp_outputs:
                         print("get ray")
                         print(x[1])
                         ray.get(x[1])
                         print("request latency:",time.monotonic()-x[2])
                else:
                    ray.get(last_pp_outputs)
                print("one batch cost:",time.monotonic()-end)
                #outputs=list(itertools.chain.from_iterable(last_pp_outputs))

                #print(outputs)
                for output in last_pp_outputs:
                    output_pipe_queue.put(output)
                #return outputs 