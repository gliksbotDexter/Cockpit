from __future__ import annotations
import json
import logging
import sqlite_utils
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OutboxMessage:
    id: int
    task: str
    payload: Dict[str, Any]
    status: str
    created_at: float
    retry_count: int = 0

class Outbox:
    """
    Transactional outbox for guaranteed at-least-once delivery of tasks.
    Implements the transactional outbox pattern to ensure tasks are never lost
    even if the worker crashes.
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            data_dir = Path(os.environ.get("DEXTER_DATA_DIR", "./data"))
            db_path = data_dir / "brain.db"
        
        self.db = sqlite_utils.Database(db_path)
        self._init_tables()
    
    def _init_tables(self) -> None:
        """Initialize outbox tables if they don't exist."""
        # Main outbox table
        self.db["outbox"].create({
            "id": int,
            "task": str,
            "payload": str,
            "status": str,
            "created_at": float,
            "retry_count": int
        }, pk="id", not_null={"task", "payload", "status", "created_at"}, defaults={"retry_count": 0}, if_not_exists=True)
        
        # Create indexes for performance
        self.db["outbox"].create_index(["status"], if_not_exists=True)
        self.db["outbox"].create_index(["created_at"], if_not_exists=True)
        
        logger.info("Outbox tables initialized")
    
    def add(self, task: str, payload: Dict[str, Any]) -> int:
        """
        Add a task to the outbox for guaranteed delivery.
        
        Args:
            task: Task name/type
            payload: Task parameters
            
        Returns:
            Message ID
        """
        message_data = {
            "task": task,
            "payload": json.dumps(payload),
            "status": "pending",
            "created_at": time.time(),
            "retry_count": 0
        }
        
        message_id = self.db["outbox"].insert(message_data).last_pk
        logger.info(f"Added task {task} to outbox with ID {message_id}")
        return message_id
    
    def get_pending(self, limit: int = 10) -> List[OutboxMessage]:
        """
        Get pending messages for processing.
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of pending messages
        """
        rows = list(self.db["outbox"].rows_where(
            "status = 'pending'",
            order_by="created_at",
            limit=limit
        ))
        
        messages = []
        for row in rows:
            messages.append(OutboxMessage(
                id=row["id"],
                task=row["task"],
                payload=json.loads(row["payload"]),
                status=row["status"],
                created_at=row["created_at"],
                retry_count=row.get("retry_count", 0)
            ))
        
        return messages
    
    def mark_sent(self, message_id: int) -> None:
        """Mark a message as successfully sent."""
        self.db["outbox"].update(message_id, {"status": "sent"})
        logger.debug(f"Marked message {message_id} as sent")
    
    def mark_failed(self, message_id: int, error: str) -> None:
        """Mark a message as failed and increment retry count."""
        self.db.execute(
            "UPDATE outbox SET status = 'failed', retry_count = retry_count + 1 WHERE id = ?",
            [message_id]
        )
        logger.warning(f"Marked message {message_id} as failed: {error}")
    
    def mark_retry(self, message_id: int) -> None:
        """Mark a message for retry (keeps it pending but increments retry count)."""
        self.db.execute(
            "UPDATE outbox SET retry_count = retry_count + 1 WHERE id = ?",
            [message_id]
        )
        logger.debug(f"Incremented retry count for message {message_id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get outbox statistics."""
        stats = {}
        
        for status in ["pending", "sent", "failed"]:
            count = self.db.execute(
                "SELECT COUNT(*) FROM outbox WHERE status = ?",
                [status]
            ).fetchone()[0]
            stats[status] = count
        
        stats["total"] = sum(stats.values())
        return stats
    
    def cleanup_old(self, days: int = 7) -> int:
        """
        Clean up old sent messages.
        
        Args:
            days: Age in days after which to clean up sent messages
            
        Returns:
            Number of messages cleaned up
        """
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        result = self.db.execute(
            "DELETE FROM outbox WHERE status = 'sent' AND created_at < ?",
            [cutoff_time]
        )
        
        deleted_count = result.rowcount
        logger.info(f"Cleaned up {deleted_count} old sent messages")
        return deleted_count
