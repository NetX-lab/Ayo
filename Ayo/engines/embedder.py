import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch

from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


@dataclass
class EmbeddingRequest:
    """Data class for embedding requests"""

    request_id: str
    query_id: str
    texts: List[str]
    callback_ref: Any
    timestamp: float = field(default_factory=time.time)


@ray.remote(num_gpus=1)
class EmbeddingEngine:
    """Ray Actor for serving embedding requests with async processing

    Features:
    - Async request handling
    - Batches requests for efficient processing
    - Groups requests from same query
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        max_batch_size: int = 512,
        max_queue_size: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):

        logger.info(f"CUDA is available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"Number of available GPUs: {torch.cuda.device_count()}")
            logger.info(f"Current GPU device: {torch.cuda.current_device()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name()}")

        self.model_name = model_name
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.device = device

        self.name = kwargs.get("name", None)

        self.model = self._load_model()

        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.batch_queue = asyncio.Queue(maxsize=max_queue_size)

        self.query_requests: Dict[str, List[EmbeddingRequest]] = {}

        self.loop = asyncio.get_event_loop()

        self.running = True
        self.tasks = [
            self.loop.create_task(self._batch_requests()),
            self.loop.create_task(self._process_batches()),
        ]

        self.scheduler_ref = scheduler_ref

    def is_ready(self):
        """Check if the engine is ready"""
        return True

    def _load_model(self):
        """Load the embedding model"""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            model_name_or_path=self.model_name, device=self.device
        )
        model.half()
        model.eval()

        warm_up_embeddings = model.encode(["hello", "world"])
        logger.debug(f"warm_up_embeddings: {warm_up_embeddings}")

        logger.debug(
            f"Load and warm up embedding model:{self.model_name} successfully on {self.device}"
        )

        return model

    async def submit_request(
        self, request_id: str, query_id: str, texts: List[str]
    ) -> ray.ObjectRef:
        """Submit a new embedding request"""

        request = EmbeddingRequest(
            request_id=request_id, query_id=query_id, texts=texts, callback_ref=None
        )

        if self.request_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Request queue is full")

        await self.request_queue.put(request)

        if query_id not in self.query_requests:
            self.query_requests[query_id] = []

        self.query_requests[query_id].append(request)

    async def _batch_requests(self):
        """Async task for batching requests"""
        while self.running:
            try:
                batch_requests, batch_texts = await self._get_next_batch()
                if batch_requests:
                    await self.batch_queue.put((batch_requests, batch_texts))
                else:
                    await asyncio.sleep(0)
            except Exception as e:
                logger.error(f"Error in batching task: {e}")
                continue

    async def _process_batches(self):
        """Async task for processing batches"""
        while self.running:
            try:
                try:
                    batch_requests, batch_texts = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=0.1
                    )
                except asyncio.TimeoutError:
                    continue

                try:
                    embeddings = await self.loop.run_in_executor(
                        None, self._compute_embeddings, batch_texts
                    )

                    start_idx = 0
                    for request in batch_requests:
                        try:
                            end_idx = start_idx + len(request.texts)
                            request_embeddings = embeddings[start_idx:end_idx]

                            logger.debug(
                                f"request_embeddings shape: {request_embeddings.shape}"
                            )

                            result_ref = ray.put(request_embeddings)

                            if self.scheduler_ref is not None:
                                await self.scheduler_ref.on_result.remote(
                                    request.request_id, request.query_id, result_ref
                                )
                            else:
                                ray.get(
                                    ray.put(
                                        request_embeddings, _owner=request.callback_ref
                                    )
                                )

                            if request.query_id in self.query_requests:
                                try:
                                    self.query_requests[request.query_id].remove(
                                        request
                                    )
                                except ValueError:
                                    pass
                                if not self.query_requests[request.query_id]:
                                    del self.query_requests[request.query_id]

                            start_idx = end_idx
                        except Exception as e:
                            logger.error(f"Error processing individual request: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error computing embeddings: {e}")
                    continue

            except Exception as e:
                logger.error(f"Error in inference task: {e}")
                continue

    async def _get_next_batch(self) -> Tuple[List[EmbeddingRequest], List[str]]:
        """Get next batch of requests to process"""
        batch_requests = []
        batch_texts = []
        processed_queries = set()

        while len(batch_texts) == 0:
            try:
                request = await asyncio.wait_for(self.request_queue.get(), timeout=0.01)

                if request.query_id in processed_queries:
                    pending_requests = self.query_requests[request.query_id]
                    for pending_req in pending_requests:
                        if (
                            len(batch_texts) + len(pending_req.texts)
                            <= self.max_batch_size
                        ):
                            batch_requests.append(pending_req)
                            batch_texts.extend(pending_req.texts)
                else:
                    if len(batch_texts) + len(request.texts) <= self.max_batch_size:
                        batch_requests.append(request)
                        batch_texts.extend(request.texts)
                        processed_queries.add(request.query_id)
                    else:
                        await self.request_queue.put(request)
                        break

            except asyncio.TimeoutError:
                break

        return batch_requests, batch_texts

    def _compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts"""
        with torch.no_grad():

            assert isinstance(texts, list) or isinstance(texts, str)

            begin = time.time()

            batch_size = len(texts) if isinstance(texts, list) else 1
            embeddings = self.model.encode(
                texts, batch_size=batch_size, show_progress_bar=False
            )

            logger.debug(
                f"texts' type: {type(texts)}, len: {len(texts)}, embeddings shape: {embeddings.shape}"
            )
            end = time.time()
            logger.debug(f"embedding time for {len(texts)} texts: {end - begin}")
            return embeddings

    async def shutdown(self):
        """Shutdown the service"""
        self.running = False
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def cleanup_query(self, query_id: str):
        self.query_requests.pop(query_id, None)
