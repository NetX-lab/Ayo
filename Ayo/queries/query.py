import time
from typing import Any, Dict, Optional

from Ayo.dags.dag import DAG
from Ayo.queries.query_state import QueryStates, QueryStatus


class Query:
    """Base class for all query types in the system"""

    def __init__(
        self,
        uuid: str,
        query_id: str,
        query_inputs: Dict[str, Any],
        DAG: DAG,
        context: Optional[Dict] = None,
        uploaded_file: Optional[Any] = None,
        timeout: float = 30.0,
        # here some attributes maybe not used... to clear some in the future
    ):
        # Basic query information
        self.uuid = uuid
        self.query_id = query_id
        self.query_inputs = query_inputs
        self.context = context or {}
        self.DAG = DAG
        self.uploaded_file = uploaded_file

        # Initialize DAG information
        self._init_dag()

        # Query state management
        self.status = QueryStatus.INIT
        self.query_state = QueryStates.remote()
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.timeout = timeout
        self.error_message: Optional[str] = None

        # Results storage
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def _init_dag(self) -> None:
        """Initialize DAG query related information"""
        self.DAG.query_id = self.query_id
        self.DAG.set_query_inputs(self.query_inputs)
        self.DAG.create_input_nodes()

    def start(self):
        """Start query execution"""
        self.status = QueryStatus.RUNNING
        self.start_time = time.time()
        self.updated_at = self.start_time

    def complete(self):
        """Mark query as completed"""
        self.status = QueryStatus.COMPLETED
        self.end_time = time.time()
        self.updated_at = self.end_time

    def fail(self, error_message: str):
        """Mark query as failed"""
        self.status = QueryStatus.FAILED
        self.error_message = error_message
        self.end_time = time.time()
        self.updated_at = self.end_time

    def set_timeout(self):
        """Mark query as timed out"""
        self.status = QueryStatus.TIMEOUT
        self.error_message = "Query execution exceeded timeout"
        self.end_time = time.time()
        self.updated_at = self.end_time

    def is_timeout(self) -> bool:
        """Check if query has exceeded timeout"""
        if self.start_time is None:
            return False
        return (time.time() - self.start_time) > self.timeout

    def get_execution_time(self) -> Optional[float]:
        """Get query execution time in seconds"""
        if self.start_time is None:
            return None
        end = self.end_time or time.time()
        return end - self.start_time

    def _get_obj_name_recurse(self, name, obj):
        """Helper method for attribute access"""
        name = name.split(".", maxsplit=1)
        recurse = len(name) > 1
        next_name = name[1] if recurse else ""
        name = name[0]
        obj = self if obj is None else obj
        return obj, name, next_name, recurse

    def get_status(self) -> QueryStatus:
        """Get query status"""
        return self.status

    def get_remote_attr(self, __name: str, __obj: object = None):
        """Get remote attribute value"""
        obj, name, next_name, recurse = self._get_obj_name_recurse(__name, __obj)
        next_obj = getattr(obj, name)
        if recurse:
            next_obj = self.get_remote_attr(next_name, next_obj)
        return next_obj

    def set_remote_attr(self, __name: str, __value: Any, __obj: object = None):
        """Set remote attribute value"""
        obj, name, next_name, recurse = self._get_obj_name_recurse(__name, __obj)
        if recurse:
            next_obj = getattr(obj, name)
            self.set_remote_attr(next_name, __value, next_obj)
        else:
            if hasattr(obj, name):
                setattr(obj, name, __value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert query to dictionary representation"""
        return {
            "uuid": self.uuid,
            "query_id": self.query_id,
            "query_inputs": self.query_inputs,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_time": self.get_execution_time(),
            "error_message": self.error_message,
            "results": self.results,
            "context": self.context,
            "metadata": self.metadata,
        }

    def __str__(self):
        return f"Query(uuid={self.uuid}, status={self.status.value}, query_id='{self.query_id}', query_inputs={self.query_inputs})"

    def __repr__(self):
        return self.__str__()
