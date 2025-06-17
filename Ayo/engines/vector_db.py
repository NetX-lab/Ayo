import asyncio
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import asyncpg
import numpy as np
import ray
from pgvector.asyncpg import register_vector

from Ayo.logger import GLOBAL_INFO_LEVEL, get_logger

logger = get_logger(__name__, level=GLOBAL_INFO_LEVEL)


class RequestType(str, Enum):
    """Types of vector database operations"""

    INGESTION = "ingestion"  # Insert vectors and texts
    SEARCHING = "searching"  # Search for similar vectors


@dataclass
class VectorDBRequest:
    """Data class for vector database requests"""

    request_id: str
    query_id: str
    request_type: RequestType
    data: Union[
        List[Tuple[Union[np.ndarray, list], str]], Tuple[Union[np.ndarray, list], int]
    ]  # (embeddings, texts) for ingestion or (query_vector, top_k) for searching
    callback_ref: Any  # Ray ObjectRef for result
    timestamp: float = time.time()


@ray.remote
class VectorDBEngine:
    """Ray Actor for serving vector database operations

    Features:
    - Async request handling
    - Batched insertions
    - Per-query table management
    - Concurrent read/write operations
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        user: str = "asplos25",
        password: str = "123456",
        database: str = "database_asplos",
        max_batch_size: int = 1000,
        max_queue_size: int = 2000,
        vector_dim: int = 768,
        scheduler_ref: Optional[ray.actor.ActorHandle] = None,
        **kwargs,
    ):

        self.db_config = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database,
        }
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        self.vector_dim = vector_dim

        self.name = kwargs.get("name", None)

        # Async queues for different operations
        self.insert_queue = asyncio.Queue(maxsize=max_queue_size)
        self.search_queue = asyncio.Queue(maxsize=max_queue_size)

        # Track active tables and requests
        self.active_tables: Dict[str, str] = {}  # query_id -> table_name
        self.query_requests: Dict[str, List[VectorDBRequest]] = {}

        # Create event loop and connection pool
        self.pool = None

        # Start processing tasks
        self.running = True
        self.tasks = []
        asyncio.create_task(self._initialize())

        self.scheduler_ref = scheduler_ref

    def is_ready(self):
        """Check if the engine is ready"""
        return True

    async def _initialize(self):
        """Initialize database connection and start processing tasks"""
        # Create connection pool
        self.pool = await asyncpg.create_pool(**self.db_config)

        # Register vector type with asyncpg
        async with self.pool.acquire() as conn:
            await register_vector(conn)

        # Start processing tasks
        self.tasks = [
            asyncio.create_task(self._process_inserts()),
            asyncio.create_task(self._process_searches()),
        ]

    def _get_table_name(self, query_id: str) -> str:
        """Generate unique table name for query_id"""
        hash_obj = hashlib.md5(query_id.encode())
        return f"vectors_{hash_obj.hexdigest()}"

    async def _ensure_table(self, query_id: str) -> str:
        """Ensure table exists for query_id"""
        if query_id not in self.active_tables:
            logger.debug(f"ensure table for query_id: {query_id}")
            table_name = self._get_table_name(query_id)
            async with self.pool.acquire() as conn:

                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        embedding vector({self.vector_dim}),
                        text TEXT
                    )
                """
                )

                # Use simple vector index - use HNSW index, better for small dataset and no pre-training
                await conn.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                    ON {table_name}
                    USING hnsw (embedding vector_cosine_ops)
                """
                )
            self.active_tables[query_id] = table_name
        return self.active_tables[query_id]

    async def ingest(
        self,
        request_id: str,
        query_id: str,
        embeddings: List[np.ndarray],
        texts: List[str],
    ) -> ray.ObjectRef:
        """Ingest vectors and texts into database

        Args:
            request_id: Unique identifier for this request
            query_id: Query identifier for table management
            embeddings: List of vector embeddings
            texts: List of corresponding texts

        Returns:
            Ray ObjectRef for tracking completion
        """
        if len(embeddings) != len(texts):
            raise ValueError("Number of embeddings must match number of texts")

        data = list(zip(embeddings, texts))
        return await self.submit_request(
            request_id=request_id,
            query_id=query_id,
            request_type=RequestType.INGESTION,
            data=data,
        )

    async def search(
        self, request_id: str, query_id: str, query_vector: np.ndarray, top_k: int = 5
    ) -> ray.ObjectRef:
        """Search for similar vectors in database

        Args:
            request_id: Unique identifier for this request
            query_id: Query identifier for table lookup
            query_vector: Vector to search for
            top_k: Number of results to return

        Returns:
            Ray ObjectRef for results
        """
        return await self.submit_request(
            request_id=request_id,
            query_id=query_id,
            request_type=RequestType.SEARCHING,
            data=(query_vector, top_k),
        )

    async def submit_request(
        self,
        request_id: str,
        query_id: str,
        request_type: RequestType,
        data: Union[List[Tuple[np.ndarray, str]], Tuple[np.ndarray, int]],
    ) -> None:
        """Submit a new vector database request"""

        request = VectorDBRequest(
            request_id=request_id,
            query_id=query_id,
            request_type=request_type,
            data=data,
            callback_ref=None,
        )

        logger.info(
            f"submit request in vector_db: {request.request_type} {request.request_id} {request.query_id}"
        )

        # Route request to appropriate queue
        if request_type == RequestType.INGESTION:
            if self.insert_queue.qsize() >= self.max_queue_size:
                raise RuntimeError("Ingestion queue is full")
            logger.debug(f"put request {request.request_id} in vector_db insert queue")
            await self.insert_queue.put(request)
        else:  # RequestType.SEARCHING
            if self.search_queue.qsize() >= self.max_queue_size:
                raise RuntimeError("Search queue is full")
            await self.search_queue.put(request)

        # Track request
        if query_id not in self.query_requests:
            self.query_requests[query_id] = []
        self.query_requests[query_id].append(request)

    async def _process_inserts(self):
        """Process insertion requests"""
        while self.running:
            try:
                # Collect batch of insertion requests
                batch_requests = []
                batch_data = []

                # while len(batch_requests) < self.max_batch_size:
                while len(batch_requests) == 0:
                    try:
                        request = await asyncio.wait_for(
                            self.insert_queue.get(), timeout=0.1
                        )
                        batch_requests.append(request)
                        batch_data.append(request.data)

                        logger.debug(
                            f"vector_db insert batch_requests len: {len(batch_requests)}"
                        )

                    except asyncio.TimeoutError:
                        break
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        logger.error(f"vector_db insert error: {e}")
                        break

                if not batch_requests:
                    continue

                # Group requests by query_id
                query_groups: Dict[str, List[int]] = {}
                for i, request in enumerate(batch_requests):
                    if request.query_id not in query_groups:
                        query_groups[request.query_id] = []
                    query_groups[request.query_id].append(i)

                # Process each query group
                for query_id, indices in query_groups.items():
                    table_name = await self._ensure_table(query_id)
                    logger.debug(f"ingest in vector_db: {query_id} {table_name}")

                    # each item in batch_data is a list of (embedding, text) pairs
                    # data = [(embedding, text) for embedding, text in zip(embeddings, texts)]
                    # merge all items in batch_data into a single list of (embedding, text) pairs

                    group_data = []
                    for i in indices:
                        request_data = batch_data[i]
                        group_data.extend(request_data)

                    logger.debug(f"len of group_data: {len(group_data)}")

                    # Insert vectors in batch
                    begin = time.time()
                    async with self.pool.acquire() as conn:

                        if isinstance(group_data[0][0], list):

                            await conn.executemany(
                                f"INSERT INTO {table_name} (embedding, text) VALUES ($1, $2)",
                                [(embedding, text) for embedding, text in group_data],
                            )
                        else:

                            await conn.executemany(
                                f"INSERT INTO {table_name} (embedding, text) VALUES ($1, $2)",
                                [
                                    (embedding.tolist(), text)
                                    for embedding, text in group_data
                                ],
                            )

                        end = time.time()
                        logger.debug(f"insert time: {end - begin}")

                        count = await conn.fetchval(
                            f"SELECT COUNT(*) FROM {table_name}"
                        )
                        logger.debug(
                            f"Table {table_name} contains {count} records after ingest"
                        )
                    # Update results and clean up
                    for i in indices:
                        request = batch_requests[i]
                        result_ref = ray.put(True)

                        if self.scheduler_ref is not None:
                            await self.scheduler_ref.on_result.remote(
                                request.request_id, request.query_id, result_ref
                            )

                        if request.query_id in self.query_requests:
                            self.query_requests[request.query_id].remove(request)

            except Exception as e:
                # traceback
                import traceback

                traceback.print_exc()
                logger.error(f"Error in insert processing: {e}")
                continue

    async def _process_searches(self):
        """Process search requests"""
        while self.running:
            try:
                try:
                    request = await asyncio.wait_for(
                        self.search_queue.get(), timeout=0.02
                    )
                except asyncio.TimeoutError:
                    continue

                query_vectors, top_k = request.data
                table_name = await self._ensure_table(request.query_id)

                logger.debug(
                    f"search in vector_db: {request.request_id} {request.query_id} {table_name}"
                )

                # Execute vector search
                async with self.pool.acquire() as conn:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    logger.debug(f"Table {table_name} contains {count} records")

                    # Process multiple query vectors
                    all_results = []
                    begin = time.time()

                    for query_vector in query_vectors:
                        # Ensure query vector format is correct
                        if isinstance(query_vector, list):
                            query_vector = np.array(query_vector)

                        if len(query_vector.shape) > 1:
                            query_vector = query_vector.flatten()

                        # Execute single vector query
                        vector_results = await conn.fetch(
                            f"""
                            SELECT text, embedding <=> $1 as distance
                            FROM {table_name}
                            ORDER BY embedding <=> $1
                            LIMIT $2
                            """,
                            query_vector.tolist(),
                            top_k,
                        )

                        # Format single query result
                        search_results = [
                            {
                                "text": record["text"],
                                "score": 1
                                - float(
                                    record["distance"]
                                ),  # Convert distance to similarity score
                            }
                            for record in vector_results
                        ]
                        all_results.append(search_results)

                    end = time.time()
                    logger.debug(
                        f"search time for {len(query_vectors)} vectors: {end - begin}"
                    )

                # If there is only one query vector, return a single result list instead of a nested list
                if len(all_results) == 1:
                    search_results = all_results[0]
                    # sort and take top_k
                    search_results = sorted(
                        search_results, key=lambda x: x["score"], reverse=True
                    )[:top_k]
                else:
                    search_results = all_results
                    for i, result in enumerate(search_results):
                        search_results[i] = sorted(
                            result, key=lambda x: x["score"], reverse=True
                        )[:top_k]

                logger.debug(f"search results: {search_results}")

                # Update results and clean up
                result_ref = ray.put(search_results)

                if self.scheduler_ref is not None:
                    await self.scheduler_ref.on_result.remote(
                        request.request_id, request.query_id, result_ref
                    )

                if request.query_id in self.query_requests:
                    self.query_requests[request.query_id].remove(request)

            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error in search processing: {e}")
                continue

    async def cleanup_query(self, query_id: str):
        """Clean up resources for a query"""
        if query_id in self.active_tables:
            table_name = self.active_tables[query_id]
            async with self.pool.acquire() as conn:
                await conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            del self.active_tables[query_id]

    async def shutdown(self):
        """Shutdown the service"""
        self.running = False

        # Cancel processing tasks
        for task in self.tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Clean up all tables created by queries
        try:
            for query_id in list(self.active_tables.keys()):
                await self.cleanup_query(query_id)
            print("Successfully cleaned up all tables")
        except Exception as e:
            print(f"Error cleaning up tables: {e}")

        # Close database pool
        if self.pool:
            await self.pool.close()
