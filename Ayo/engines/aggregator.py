from typing import Dict, List, Optional, Any, Tuple, Union
import ray
import asyncio
import time
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# This engine would not really be used in the pipeline; 
# Rather we do the in-place aggregation in the engine scheduler slide

@dataclass
class AggregateRequest:
    """Data class for aggregate requests"""
    request_id: str
    query_id: str  # Group requests by query ID
    agg_mode: str  # Aggregation mode: concat, merge_dicts, select_best, custom
    data_sources: List[Any]  # Data sources to aggregate
    callback_ref: Any = None  # Ray ObjectRef for result callback
    timestamp: float = time.time()

@ray.remote
class AggregateEngine:
    """Ray Actor, used to handle aggregate requests
    
    Features:
    - Asynchronous request processing
    - Supports multiple aggregation modes
    - Groups requests by query ID
    """
    
    def __init__(self,
                 max_batch_size: int = 32,
                 max_queue_size: int = 1000,
                 scheduler_ref: Optional[ray.actor.ActorHandle] = None,
                 **kwargs):
        
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        
        # Asynchronous queue
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Track requests by query ID
        self.query_requests: Dict[str, List[AggregateRequest]] = {}
        
        # Start processing tasks
        self.running = True
        self.tasks = [
            asyncio.create_task(self._batch_requests()),
            asyncio.create_task(self._process_batches())
        ]
        
        self.scheduler_ref = scheduler_ref
    
    async def submit_request(self,
                           request_id: str,
                           query_id: str,
                           agg_mode: str,
                           data_sources: List[Any]) -> None:
        """Submit new aggregate request"""
        request = AggregateRequest(
            request_id=request_id,
            query_id=query_id,
            agg_mode=agg_mode,
            data_sources=data_sources,
            callback_ref=None
        )
        
        if self.request_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue is full")
        
        await self.request_queue.put(request)
        
        if query_id not in self.query_requests:
            self.query_requests[query_id] = []
        self.query_requests[query_id].append(request)
    
    async def _batch_requests(self):
        """Asynchronous task for batching requests"""
        while self.running:
            try:
                batch_requests = await self._get_next_batch()
                if batch_requests:
                    await self.batch_queue.put(batch_requests)
                else:
                    await asyncio.sleep(0.01)  # Avoid busy waiting
            except Exception as e:
                print(f"Error in batch processing task: {e}")
                continue
    
    async def _process_batches(self):
        """Asynchronous task for processing batches"""
        while self.running:
            try:
                try:
                    batch_requests = await asyncio.wait_for(
                        self.batch_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process each request
                for request in batch_requests:
                    try:
                        # Process data based on aggregation mode
                        result = await self._aggregate_data(
                            request.agg_mode,
                            request.data_sources
                        )
                        
                        # Create ObjectRef for result
                        result_ref = ray.put(result)
                        
                        # If scheduler is set, send result to scheduler
                        if self.scheduler_ref is not None:
                            await self.scheduler_ref.on_result.remote(
                                request.request_id,
                                request.query_id,
                                result_ref
                            )
                        
                        # Clean up request records
                        if request.query_id in self.query_requests:
                            self.query_requests[request.query_id].remove(request)
                            if not self.query_requests[request.query_id]:
                                del self.query_requests[request.query_id]
                                
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error in processing single request: {e}")
                        continue
                        
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error in processing batch: {e}")
                continue
    
    async def _get_next_batch(self) -> List[AggregateRequest]:
        """Get next batch of requests to process"""
        batch_requests = []
        processed_queries = set()
        
        while len(batch_requests) < self.max_batch_size:
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=0.01
                )
                
                if request.query_id in processed_queries:
                    # Process requests from the same query
                    pending_requests = self.query_requests[request.query_id]
                    batch_requests.extend(pending_requests)
                else:
                    batch_requests.append(request)
                    processed_queries.add(request.query_id)
                    
            except asyncio.TimeoutError:
                break
                
        return batch_requests
    
    async def _aggregate_data(self, agg_mode: str, data_sources: List[Any]) -> Any:
        """Aggregate data based on aggregation mode"""
        if agg_mode == "concat":
            # Simple list concatenation
            result = []
            for source in data_sources:
                if isinstance(source, list):
                    result.extend(source)
                else:
                    result.append(source)
            return result
            
        elif agg_mode == "merge_dicts":
            # Merge multiple dictionaries
            result = {}
            for source in data_sources:
                if isinstance(source, dict):
                    result.update(source)
            return result
            
        elif agg_mode == "select_best":
            # Select best result (assuming each source has a score field)
            if not data_sources:
                return None
                
            best_source = None
            best_score = float('-inf')
            
            for source in data_sources:
                if isinstance(source, dict) and 'score' in source:
                    if source['score'] > best_score:
                        best_score = source['score']
                        best_source = source
            
            return best_source
            
        elif agg_mode == "topk":
            # Select top k results with highest scores
            # datasource format: 
            if not data_sources:
                return []
                
            # Use n parameter from request as k value, default to 3 if not provided
            k = 3
                
            # Filter valid data sources (   must be dictionaries and contain score field)
            valid_sources = [
                source for source in data_sources 
                if isinstance(source, dict) and 'score' in source
            ]
            
            # Sort by score in descending order and return top k
            sorted_sources = sorted(
                valid_sources, 
                key=lambda x: x['score'], 
                reverse=True
            )
            
            return sorted_sources[:k]
            
        elif agg_mode == "custom":
            # Custom aggregation function (needs function and data in data_sources)
            if len(data_sources) >= 2 and callable(data_sources[0]):
                custom_func = data_sources[0]
                data = data_sources[1:]
                return custom_func(*data)
            return None
            
        else:
            # Default return original data sources
            return data_sources
    
    async def shutdown(self):
        """Shutdown service"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass 