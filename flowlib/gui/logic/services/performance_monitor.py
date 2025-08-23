"""
Performance monitoring and profiling service for GUI operations.

Provides comprehensive performance tracking, memory monitoring, and
operation profiling for the GUI application.
"""

import time
import logging
import asyncio
import psutil
import threading
from typing import List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from PySide6.QtCore import QObject, Signal, QTimer

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_delta: Optional[float] = None
    metadata: dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    
    def finish(self, end_time: Optional[float] = None):
        """Mark metric as finished."""
        self.end_time = end_time or time.perf_counter()
        self.duration = self.end_time - self.start_time
        
        if self.memory_before is not None:
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            self.memory_after = current_memory
            self.memory_delta = self.memory_after - self.memory_before


@dataclass
class SystemMetrics:
    """System-wide performance metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gui_memory_mb: float
    active_operations: int
    
    def to_dict(self) -> dict[str, Union[str, int, float, bool]]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'gui_memory_mb': self.gui_memory_mb,
            'active_operations': self.active_operations
        }


class PerformanceProfiler:
    """Context manager for profiling individual operations."""
    
    def __init__(self, monitor: 'PerformanceMonitor', operation_name: str, metadata: Optional[dict] = None):
        self.monitor = monitor
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.metric: Optional[PerformanceMetric] = None
    
    def __enter__(self):
        """Start profiling."""
        self.metric = self.monitor.start_operation(self.operation_name, self.metadata)
        return self.metric
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling."""
        if self.metric:
            self.monitor.end_operation(self.operation_name)
            if exc_type:
                self.metric.metadata['error'] = str(exc_val)
                self.metric.metadata['error_type'] = exc_type.__name__


class PerformanceMonitor(QObject):
    """
    Comprehensive performance monitoring system for GUI operations.
    
    Tracks operation performance, memory usage, and system metrics.
    """
    
    # Signals for performance events
    performance_alert = Signal(str, dict)  # alert_type, data
    metrics_updated = Signal(dict)  # latest metrics
    operation_completed = Signal(str, float)  # operation_name, duration
    
    def __init__(self):
        super().__init__()
        
        # Performance tracking
        self.active_operations: dict[str, PerformanceMetric] = {}
        self.completed_operations: deque = deque(maxlen=1000)  # Keep last 1000 operations
        self.operation_stats: dict[str, List[float]] = defaultdict(list)
        
        # System monitoring
        self.system_metrics_history: deque = deque(maxlen=100)  # Keep last 100 system snapshots
        self.monitoring_enabled = False
        self.monitoring_interval = 5.0  # seconds
        
        # Performance thresholds
        self.slow_operation_threshold = 2.0  # seconds
        self.memory_warning_threshold = 80.0  # percent
        self.cpu_warning_threshold = 90.0  # percent
        
        # Qt Timer for system monitoring
        self.system_monitor_timer = QTimer()
        self.system_monitor_timer.timeout.connect(self._collect_system_metrics)
        
        # Background thread for heavy monitoring
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        logger.info("Performance monitor initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """Start system performance monitoring."""
        self.monitoring_interval = interval
        self.monitoring_enabled = True
        
        # Start Qt timer for regular system metrics
        self.system_monitor_timer.start(int(interval * 1000))
        
        # Start background monitoring thread
        self.stop_monitoring.clear()
        self.monitor_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Performance monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop system performance monitoring."""
        self.monitoring_enabled = False
        self.system_monitor_timer.stop()
        
        if self.monitor_thread:
            self.stop_monitoring.set()
            self.monitor_thread.join(timeout=1.0)
        
        logger.info("Performance monitoring stopped")
    
    def start_operation(self, operation_name: str, metadata: Optional[dict] = None) -> PerformanceMetric:
        """Start tracking a new operation."""
        if operation_name in self.active_operations:
            logger.warning(f"Operation '{operation_name}' already active, overwriting")
        
        # Get current memory usage
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        metric = PerformanceMetric(
            name=operation_name,
            start_time=time.perf_counter(),
            memory_before=current_memory,
            metadata=metadata or {}
        )
        
        self.active_operations[operation_name] = metric
        logger.debug(f"Started tracking operation: {operation_name}")
        
        return metric
    
    def end_operation(self, operation_name: str) -> Optional[PerformanceMetric]:
        """End tracking an operation."""
        if operation_name not in self.active_operations:
            logger.warning(f"Operation '{operation_name}' not found in active operations")
            return None
        
        metric = self.active_operations.pop(operation_name)
        metric.finish()
        
        # Store completed operation
        self.completed_operations.append(metric)
        self.operation_stats[operation_name].append(metric.duration)
        
        # Check for performance issues
        self._check_performance_thresholds(metric)
        
        # Emit signal
        self.operation_completed.emit(operation_name, metric.duration)
        
        logger.debug(f"Completed operation: {operation_name} in {metric.duration:.3f}s")
        
        return metric
    
    def profile_operation(self, operation_name: str, metadata: Optional[dict] = None) -> PerformanceProfiler:
        """Get a context manager for profiling an operation."""
        return PerformanceProfiler(self, operation_name, metadata)
    
    @asynccontextmanager
    async def profile_async_operation(self, operation_name: str, metadata: Optional[dict] = None):
        """Async context manager for profiling async operations."""
        metric = self.start_operation(operation_name, metadata)
        try:
            yield metric
        except Exception as e:
            metric.metadata['error'] = str(e)
            metric.metadata['error_type'] = type(e).__name__
            raise
        finally:
            self.end_operation(operation_name)
    
    def get_operation_stats(self, operation_name: str) -> dict[str, Union[str, int, float, bool]]:
        """Get statistics for a specific operation."""
        durations = self.operation_stats[operation_name] if operation_name in self.operation_stats else []
        
        if not durations:
            return {"operation": operation_name, "count": 0}
        
        return {
            "operation": operation_name,
            "count": len(durations),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "recent_duration": durations[-1] if durations else 0
        }
    
    def get_all_operation_stats(self) -> List[dict[str, Union[str, int, float, bool]]]:
        """Get statistics for all tracked operations."""
        return [self.get_operation_stats(op_name) for op_name in self.operation_stats.keys()]
    
    def get_slow_operations(self, threshold: Optional[float] = None) -> List[PerformanceMetric]:
        """Get operations that exceeded the slow operation threshold."""
        threshold = threshold or self.slow_operation_threshold
        
        slow_ops = []
        for metric in self.completed_operations:
            if metric.duration and metric.duration > threshold:
                slow_ops.append(metric)
        
        return sorted(slow_ops, key=lambda x: x.duration, reverse=True)
    
    def get_memory_intensive_operations(self, threshold_mb: float = 10.0) -> List[PerformanceMetric]:
        """Get operations that used significant memory."""
        memory_ops = []
        for metric in self.completed_operations:
            if metric.memory_delta and abs(metric.memory_delta) > threshold_mb:
                memory_ops.append(metric)
        
        return sorted(memory_ops, key=lambda x: abs(x.memory_delta or 0), reverse=True)
    
    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=None),
            memory_percent=system_memory.percent,
            memory_used_mb=system_memory.used / 1024 / 1024,
            memory_available_mb=system_memory.available / 1024 / 1024,
            gui_memory_mb=memory_info.rss / 1024 / 1024,
            active_operations=len(self.active_operations)
        )
    
    def get_system_metrics_history(self, minutes: int = 10) -> List[SystemMetrics]:
        """Get system metrics history for the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        return [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_performance_summary(self) -> dict[str, Union[str, int, float, bool]]:
        """Get comprehensive performance summary."""
        current_metrics = self.get_current_system_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": current_metrics.to_dict(),
            "active_operations": len(self.active_operations),
            "completed_operations": len(self.completed_operations),
            "slow_operations": len(self.get_slow_operations()),
            "operation_stats": self.get_all_operation_stats(),
            "memory_usage": {
                "gui_memory_mb": current_metrics.gui_memory_mb,
                "system_memory_percent": current_metrics.memory_percent,
                "memory_intensive_ops": len(self.get_memory_intensive_operations())
            },
            "performance_alerts": self._get_current_alerts()
        }
    
    def clear_history(self):
        """Clear performance history."""
        self.completed_operations.clear()
        self.operation_stats.clear()
        self.system_metrics_history.clear()
        logger.info("Performance history cleared")
    
    def _collect_system_metrics(self):
        """Collect current system metrics (Qt timer callback)."""
        try:
            metrics = self.get_current_system_metrics()
            self.system_metrics_history.append(metrics)
            
            # Emit metrics update
            self.metrics_updated.emit(metrics.to_dict())
            
            # Check for system-level alerts
            self._check_system_thresholds(metrics)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    def _background_monitoring(self):
        """Background thread for intensive monitoring tasks."""
        while not self.stop_monitoring.is_set():
            try:
                # Perform heavy monitoring tasks here
                self._analyze_memory_trends()
                self._cleanup_old_metrics()
                
                # Sleep for monitoring interval
                self.stop_monitoring.wait(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")
                self.stop_monitoring.wait(1.0)
    
    def _check_performance_thresholds(self, metric: PerformanceMetric):
        """Check if operation exceeded performance thresholds."""
        if metric.duration and metric.duration > self.slow_operation_threshold:
            alert_data = {
                "operation": metric.name,
                "duration": metric.duration,
                "threshold": self.slow_operation_threshold,
                "metadata": metric.metadata
            }
            self.performance_alert.emit("slow_operation", alert_data)
    
    def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check if system metrics exceed thresholds."""
        if metrics.memory_percent > self.memory_warning_threshold:
            alert_data = {
                "memory_percent": metrics.memory_percent,
                "threshold": self.memory_warning_threshold,
                "gui_memory_mb": metrics.gui_memory_mb
            }
            self.performance_alert.emit("high_memory", alert_data)
        
        if metrics.cpu_percent > self.cpu_warning_threshold:
            alert_data = {
                "cpu_percent": metrics.cpu_percent,
                "threshold": self.cpu_warning_threshold
            }
            self.performance_alert.emit("high_cpu", alert_data)
    
    def _analyze_memory_trends(self):
        """Analyze memory usage trends."""
        if len(self.system_metrics_history) < 5:
            return
        
        recent_metrics = list(self.system_metrics_history)[-5:]
        memory_trend = [m.gui_memory_mb for m in recent_metrics]
        
        # Simple trend detection
        if len(memory_trend) >= 3:
            if all(memory_trend[i] < memory_trend[i+1] for i in range(len(memory_trend)-1)):
                # Consistent memory increase
                alert_data = {
                    "trend": "increasing",
                    "current_memory": memory_trend[-1],
                    "increase_rate": memory_trend[-1] - memory_trend[0]
                }
                self.performance_alert.emit("memory_trend", alert_data)
    
    def _cleanup_old_metrics(self):
        """Clean up old performance data."""
        # Remove operation stats for operations not seen recently
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # This is a simple cleanup - in a real implementation,
        # you might want more sophisticated data retention policies
        pass
    
    def _get_current_alerts(self) -> List[dict[str, Union[str, int, float, bool]]]:
        """Get current performance alerts."""
        alerts = []
        
        # Check for currently slow operations
        slow_ops = self.get_slow_operations()
        if slow_ops:
            alerts.append({
                "type": "slow_operations",
                "count": len(slow_ops),
                "operations": [op.name for op in slow_ops[:5]]
            })
        
        # Check current system metrics
        current_metrics = self.get_current_system_metrics()
        if current_metrics.memory_percent > self.memory_warning_threshold:
            alerts.append({
                "type": "high_memory",
                "value": current_metrics.memory_percent,
                "threshold": self.memory_warning_threshold
            })
        
        return alerts