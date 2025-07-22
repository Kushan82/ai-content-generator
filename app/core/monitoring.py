"""Enhanced monitoring and metrics collection"""
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

@dataclass
class TaskMetrics:
    """Metrics for individual task execution"""
    task_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None
    confidence_score: float = 0.0

class AgentMetricsCollector:
    """Enhanced metrics collection for agents"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.performance_history: List[TaskMetrics] = []
        
    def start_task(self, task_id: str):
        """Record task start"""
        self.task_metrics[task_id] = TaskMetrics(
            task_id=task_id,
            agent_id=self.agent_id,
            start_time=datetime.utcnow()
        )
    
    def complete_task(self, task_id: str, duration: float, confidence: float = 0.0):
        """Record successful task completion"""
        if task_id in self.task_metrics:
            metrics = self.task_metrics[task_id]
            metrics.end_time = datetime.utcnow()
            metrics.duration = duration
            metrics.success = True
            metrics.confidence_score = confidence
            
            # Move to history
            self.performance_history.append(metrics)
            del self.task_metrics[task_id]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {}
            
        recent_tasks = [t for t in self.performance_history 
                       if t.start_time > datetime.utcnow() - timedelta(hours=24)]
        
        return {
            "total_tasks": len(self.performance_history),
            "recent_tasks_24h": len(recent_tasks),
            "success_rate": sum(1 for t in self.performance_history if t.success) / len(self.performance_history),
            "average_duration": sum(t.duration or 0 for t in self.performance_history) / len(self.performance_history),
            "average_confidence": sum(t.confidence_score for t in self.performance_history) / len(self.performance_history)
        }
