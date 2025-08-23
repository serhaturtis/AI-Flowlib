#!/usr/bin/env python3
"""
Performance optimization examples for the GUI application.

Demonstrates how to use the performance monitoring and optimization services
to improve application performance.
"""

import sys
import asyncio
import time
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from PySide6.QtWidgets import QApplication
from logic.services.service_factory import ServiceFactory
from logic.services.performance_monitor import PerformanceMonitor
from logic.services.optimization_service import OptimizationService


async def example_expensive_operation():
    """Simulate an expensive operation."""
    await asyncio.sleep(0.1)  # Simulate work
    return {"result": "success", "data": list(range(1000))}


async def example_cached_operation(optimization_service: OptimizationService):
    """Example of using cached operations."""
    
    @optimization_service.cached_operation(ttl_seconds=60)
    async def expensive_computation(n: int):
        """Expensive computation that benefits from caching."""
        print(f"Computing for n={n} (this should only happen once per value)")
        await asyncio.sleep(0.1)  # Simulate expensive work
        return sum(range(n))
    
    # First call - will compute
    result1 = await expensive_computation(1000)
    print(f"First call result: {result1}")
    
    # Second call - will use cache
    result2 = await expensive_computation(1000)
    print(f"Second call result: {result2} (from cache)")
    
    # Different parameter - will compute
    result3 = await expensive_computation(500)
    print(f"Third call result: {result3}")


async def example_batched_operations(optimization_service: OptimizationService):
    """Example of batching operations for efficiency."""
    
    @optimization_service.batched_operation("data_processing", batch_size=5)
    async def process_data_item(item_id: int):
        """Process a single data item (batched automatically)."""
        print(f"Processing item {item_id}")
        await asyncio.sleep(0.01)  # Simulate processing
        return f"processed_{item_id}"
    
    # These calls will be batched together
    tasks = []
    for i in range(12):  # Will create 3 batches of 5, 5, and 2 items
        task = asyncio.create_task(process_data_item(i))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    print(f"Batched processing results: {results}")


async def example_lazy_loading(optimization_service: OptimizationService):
    """Example of lazy loading resources."""
    
    @optimization_service.lazy_loader("expensive_resource")
    async def load_expensive_resource():
        """Load an expensive resource only when needed."""
        print("Loading expensive resource...")
        await asyncio.sleep(0.2)  # Simulate loading
        return {"resource_data": "large_dataset", "size": "100MB"}
    
    # First access - will load
    resource1 = await load_expensive_resource()
    print(f"First access: {resource1}")
    
    # Second access - will use cached version
    resource2 = await load_expensive_resource()
    print(f"Second access: {resource2} (cached)")


async def example_debounced_operations(optimization_service: OptimizationService):
    """Example of debouncing rapid operations."""
    
    @optimization_service.debounce(delay=0.1)
    async def search_operation(query: str):
        """Search operation that should be debounced."""
        print(f"Searching for: {query}")
        await asyncio.sleep(0.05)  # Simulate search
        return f"results_for_{query}"
    
    # Rapid successive calls - only the last one will execute
    print("Making rapid search calls...")
    task1 = asyncio.create_task(search_operation("hello"))
    task2 = asyncio.create_task(search_operation("hello_w"))
    task3 = asyncio.create_task(search_operation("hello_wo"))
    task4 = asyncio.create_task(search_operation("hello_wor"))
    task5 = asyncio.create_task(search_operation("hello_world"))
    
    # Wait for all tasks (most will be cancelled)
    results = await asyncio.gather(task1, task2, task3, task4, task5, return_exceptions=True)
    print(f"Debounced search results: {results}")


async def example_performance_monitoring():
    """Example of comprehensive performance monitoring."""
    
    # Create services
    service_factory = ServiceFactory()
    performance_monitor = service_factory.get_performance_monitor()
    optimization_service = service_factory.get_optimization_service()
    
    # Start monitoring
    performance_monitor.start_monitoring(interval=1.0)
    
    print("=== Performance Monitoring Example ===")
    
    # Example 1: Manual operation tracking
    print("\n1. Manual operation tracking:")
    with performance_monitor.profile_operation("manual_operation"):
        await asyncio.sleep(0.2)
        print("Manual operation completed")
    
    # Example 2: Async context manager
    print("\n2. Async operation tracking:")
    async with performance_monitor.profile_async_operation("async_operation"):
        await example_expensive_operation()
        print("Async operation completed")
    
    # Example 3: Cached operations
    print("\n3. Cached operations:")
    await example_cached_operation(optimization_service)
    
    # Example 4: Batched operations
    print("\n4. Batched operations:")
    await example_batched_operations(optimization_service)
    
    # Example 5: Lazy loading
    print("\n5. Lazy loading:")
    await example_lazy_loading(optimization_service)
    
    # Example 6: Debounced operations
    print("\n6. Debounced operations:")
    await example_debounced_operations(optimization_service)
    
    # Wait a bit for monitoring data
    await asyncio.sleep(2)
    
    # Get performance statistics
    print("\n=== Performance Statistics ===")
    
    # Operation statistics
    operation_stats = performance_monitor.get_all_operation_stats()
    for stats in operation_stats:
        print(f"Operation: {stats['operation']}")
        print(f"  Count: {stats['count']}")
        print(f"  Avg Duration: {stats['avg_duration']:.3f}s")
        print(f"  Min/Max: {stats['min_duration']:.3f}s / {stats['max_duration']:.3f}s")
    
    # Slow operations
    slow_ops = performance_monitor.get_slow_operations(threshold=0.15)
    if slow_ops:
        print(f"\nSlow operations (>0.15s): {len(slow_ops)}")
        for op in slow_ops[:3]:  # Show top 3
            print(f"  {op.name}: {op.duration:.3f}s")
    
    # System metrics
    system_metrics = performance_monitor.get_current_system_metrics()
    print(f"\nSystem Metrics:")
    print(f"  CPU: {system_metrics.cpu_percent:.1f}%")
    print(f"  Memory: {system_metrics.memory_percent:.1f}%")
    print(f"  GUI Memory: {system_metrics.gui_memory_mb:.1f}MB")
    print(f"  Active Operations: {system_metrics.active_operations}")
    
    # Optimization statistics
    opt_stats = optimization_service.get_optimization_stats()
    print(f"\nOptimization Statistics:")
    print(f"  Cache hits: {sum(opt_stats['cache_hits'].values())}")
    print(f"  Cache misses: {sum(opt_stats['cache_misses'].values())}")
    print(f"  Loaded resources: {opt_stats['loaded_resources']}")
    
    # Performance analysis
    analysis = await optimization_service.analyze_performance_bottlenecks()
    print(f"\nPerformance Analysis:")
    if analysis['bottlenecks']:
        print("  Bottlenecks:")
        for bottleneck in analysis['bottlenecks']:
            print(f"    - {bottleneck}")
    
    if analysis['recommendations']:
        print("  Recommendations:")
        for rec in analysis['recommendations']:
            print(f"    - {rec}")
    
    # Performance summary
    summary = performance_monitor.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total operations completed: {summary['completed_operations']}")
    print(f"  Slow operations: {summary['slow_operations']}")
    print(f"  Memory usage: {summary['memory_usage']['gui_memory_mb']:.1f}MB")
    
    # Stop monitoring
    performance_monitor.stop_monitoring()
    
    print("\n=== Example completed ===")


async def main():
    """Main example function."""
    # Create Qt application (required for signals)
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        await example_performance_monitoring()
    except Exception as e:
        print(f"Error in example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())