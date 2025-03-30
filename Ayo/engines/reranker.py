from typing import Dict, List, Optional, Any, Tuple
import ray
import torch
import asyncio
import time
from dataclasses import dataclass
from collections import deque
from Ayo.logger import get_logger, GLOBAL_INFO_LEVEL

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)

@dataclass
class RerankerRequest:
    """Data class for reranker requests"""
    request_id: str
    query_id: str
    query_text: str
    passages: List[str]
    top_k: int 
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = time.time()



@ray.remote
class RerankerEngine:
    """Ray Actor for serving reranking requests with async processing
    
    Features:
    - Async request handling
    - Batches requests for efficient processing
    - Groups requests from same query
    """
    
    def __init__(self, 
                 model_name: str = "BAAI/bge-reranker-large",
                 max_batch_size: int = 128,
                 max_queue_size: int = 2000,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 scheduler_ref: Optional[ray.actor.ActorHandle] = None,
                 batch_strategy: str = "request",  # add new parameter: "request" or "pair"
                 **kwargs):
        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.device = device
        self.batch_strategy = batch_strategy  # add new parameter: "request" or "pair"
        
        self.name = kwargs.get("name", None)
        
        # Initialize model
        self.model = self._load_model()
        
        # Async queues
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Track requests by query_id
        self.query_requests: Dict[str, List[RerankerRequest]] = {}
        
        # Start processing tasks
        self.running = True
        self.tasks = [
            asyncio.create_task(self._batch_requests()),
            asyncio.create_task(self._process_batches())
        ]
        
        self.scheduler_ref = scheduler_ref


    def is_ready(self):
        """Check if the engine is ready"""
        return True
    
    
    def _load_model(self):
        """Load the reranker model"""
        from sentence_transformers import CrossEncoder
        
        model = CrossEncoder(
            model_name=self.model_name,
            device=self.device
        )
        model.model.half()

        res=model.predict(["hello", "world"])  # Warm up
        print(res)
        return model
    
    async def submit_request(self, 
                           request_id: str, 
                           query_id: str,
                           query: str,
                           passages: List[str],
                           top_k: int) -> None:
        """Submit a new reranking request"""
        request = RerankerRequest(
            request_id=request_id,
            query_id=query_id,
            query_text=query,
            passages=passages,
            top_k=top_k,
            callback_ref=None
        )
        
        if self.request_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        await self.request_queue.put(request)
        
        if query_id not in self.query_requests:
            self.query_requests[query_id] = []
        self.query_requests[query_id].append(request)
    
    async def _batch_requests(self):
        """Task for batching requests"""
        while self.running:
            try:
                batch_requests, batch_data = await self._get_next_batch()
                if batch_requests:
                    await self.batch_queue.put((batch_requests, batch_data))
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in batching task: {e}")
                continue
    
    async def _process_batches(self):
        """Task for processing batches"""
        while self.running:
            try:
                try:
                    batch_requests, (queries, passages) = await asyncio.wait_for(
                        self.batch_queue.get(), 
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process batch
                try:
                    with torch.no_grad():
                        # create pairs 
                        #pairs=[[query, passage], [query, passage], ...]
                        pairs = [[queries[i], p] for i, (_, query_passages) in enumerate(zip(batch_requests, passages)) for p in query_passages]
                        # pair_indices=[0, 0, 0, 1, 1, 1, 2, 2, 2, ...]
                        pair_indices = [i for i, (_, query_passages) in enumerate(zip(batch_requests, passages)) for _ in query_passages]
                        
                        # Compute scores in sub-batches
                        all_scores = []

                        max_batch_size=len(pairs)

                        for i in range(0, len(pairs), max_batch_size):
                            sub_batch = pairs[i:i + max_batch_size]
                            begin_time = time.time()
                            scores = self.model.predict(
                                sub_batch,
                                convert_to_numpy=True,
                                batch_size=self.max_batch_size
                            )
                            end_time = time.time()
                            print(f"Reranking time for batch {i} to {i + max_batch_size}: {end_time - begin_time} seconds")
                            all_scores.extend(scores.tolist())
                        
                        # Group scores by request
                        request_scores = {}
                        for score, idx in zip(all_scores, pair_indices):
                            if idx not in request_scores:
                                request_scores[idx] = []
                            request_scores[idx].append(score)
                        
                        # Create or update results for each request
                        for i, request in enumerate(batch_requests):
                            scores = request_scores[i]
                            
                            # If using pair strategy, need special handling for partial results
                            if self.batch_strategy == "pair":
                                # Get or create a temporary result storage for the request
                                if not hasattr(request, 'partial_results'):
                                    request.partial_results = []
                                    request.processed_passages = 0
                                
                                # Add the results of this batch processing
                                for passage, score in zip(passages[i], scores):
                                    request.partial_results.append({
                                        "passage": passage,
                                        "score": score
                                    })
                                
                                request.processed_passages += len(passages[i])
                                
                                # Check if all passages are processed
                                if request.processed_passages < len(request.passages):
                                    # There are remaining passages, skip result submission
                                    continue
                                
                                # All passages are processed, use the accumulated results
                                results = request.partial_results
                            else:
                                # For request strategy, directly create results
                                results = [
                                    {
                                        "passage": passage,
                                        "score": score
                                    }
                                    for passage, score in zip(passages[i], scores)
                                ]

                            top_k=request.top_k
                            
                            # Sort the results
                            results.sort(key=lambda x: x["score"], reverse=True)

                            logger.debug(f"original results length: {len(results)}; after top_k: {top_k}")
                            results = results[:top_k]


                            # Create result ObjectRef
                            result_ref = ray.put(results)


                            #TODO: Handle the situation for pair strategy
                            if self.scheduler_ref is not None:
                                await self.scheduler_ref.on_result.remote(
                                    request.request_id,
                                    request.query_id,
                                    result_ref
                                )
                            
                            # Clean up request tracking
                            if request.query_id in self.query_requests:
                                self.query_requests[request.query_id].remove(request)
                                if not self.query_requests[request.query_id]:
                                    del self.query_requests[request.query_id]
                
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    
            except Exception as e:
                print(f"Error in process loop: {e}")
                continue
    
    async def _get_next_batch(self) -> Tuple[List[RerankerRequest], Tuple[List[str], List[List[str]]]]:
        """Get the next batch of requests to process"""
        if self.batch_strategy == "request":
            return await self._get_next_batch_by_request()
        else:  # "pair"
            return await self._get_next_batch_by_pair()
    
    async def _get_next_batch_by_request(self) -> Tuple[List[RerankerRequest], Tuple[List[str], List[List[str]]]]:
        """Get the next batch of requests to process - process one request at a time"""
        #TODO: Support batch multiple requests at a time
        batch_requests = []
        batch_queries = []
        batch_passages = []
        
        try:
            # Try to get a request, if the queue is empty, wait for a short time
            request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=0.01
            )
            
            # Get the request immediately and return
            batch_requests.append(request)
            batch_queries.append(request.query_text)
            batch_passages.append(request.passages)
            
        except asyncio.TimeoutError:
            # The queue is empty, return empty batch
            pass
                
        return batch_requests, (batch_queries, batch_passages)
    
    async def _get_next_batch_by_pair(self) -> Tuple[List[RerankerRequest], Tuple[List[str], List[List[str]]]]:
        """Get the next batch of requests to process - process one query-passage pair at a time"""
        batch_requests = []
        batch_queries = []
        batch_passages = []
        
        # Track the passage indices of each request
        request_passage_indices = {}  # {request_id: [passage_indices]}
        
        current_pairs = 0  # Track the total number of query-passage pairs
        
        pending_requests = deque()  # Store the requests that are not fully processed
        
        # Try to get a request from the queue
        try:
            while current_pairs < self.max_batch_size:
                if not pending_requests:
                    try:
                        # Get a new request
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=0.01
                        )
                        pending_requests.append((request, 0))  # (request, start_index)
                    except asyncio.TimeoutError:
                        break
                
                if not pending_requests:
                    break
                
                # Process the pending requests
                request, start_idx = pending_requests.popleft()
                
                # Calculate how many passages can be added for this request
                remaining_passages = len(request.passages) - start_idx
                available_slots = self.max_batch_size - current_pairs
                passages_to_add = min(remaining_passages, available_slots)
                
                if passages_to_add <= 0:
                    continue
                
                # Add the request to the batch
                if request.request_id not in request_passage_indices:
                    batch_requests.append(request)
                    batch_queries.append(request.query_text)
                    batch_passages.append([])
                    request_passage_indices[request.request_id] = []
                    request_idx = len(batch_requests) - 1
                else:
                    # Find the index of this request in the batch
                    request_idx = next(i for i, r in enumerate(batch_requests) if r.request_id == request.request_id)
                
                # Add the passages
                end_idx = start_idx + passages_to_add
                batch_passages[request_idx].extend(request.passages[start_idx:end_idx])
                request_passage_indices[request.request_id].extend(list(range(start_idx, end_idx)))
                current_pairs += passages_to_add
                
                # If the request has remaining passages, put it back to the pending requests
                if end_idx < len(request.passages):
                    pending_requests.append((request, end_idx))
        
        except Exception as e:
            print(f"Error in batch construction: {e}")
        
        # If there are pending requests, put them back to the request queue
        for request, start_idx in pending_requests:
            # Create a new request, only containing the remaining passages
            new_request = RerankerRequest(
                request_id=request.request_id,
                query_id=request.query_id,
                query_text=request.query_text,
                passages=request.passages[start_idx:],
                callback_ref=request.callback_ref,
                timestamp=request.timestamp
            )
            asyncio.create_task(self.request_queue.put(new_request))
        
        return batch_requests, (batch_queries, batch_passages)
    
    
    
    async def shutdown(self):
        """Shutdown the service"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass 