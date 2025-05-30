"""
Live Data Collector - IO component for real-time metrics collection

This module provides a background collector that samples system metrics
and feeds them into the alignment vector space as a live data source.
"""

import queue
import threading
import time
import logging
from typing import Any, Dict, Optional, List, Generator

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available. Live metrics collection will be limited.")

try:
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy/SciPy not available. Using fallback implementations for smoothing.")

logger = logging.getLogger("constitutional_dynamics.io.live")

# Default configuration
DEFAULT_INTERVAL = 1.0  # seconds
GAUSS_SIGMA = 2.0       # Gaussian window for smoothing
MAX_QUEUE = 512         # rolling buffer


class LiveMetricsCollector(threading.Thread):
    """
    Background worker that samples host metrics and provides them as a
    stream of vectors for alignment analysis.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        callback: Optional[callable] = None,
        baseline: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize the live metrics collector.

        Args:
            config: Configuration dictionary
            callback: Optional callback function to receive metrics
            baseline: Optional baseline metrics for normalization
        """
        super().__init__(name="constitutional_dynamics-collector", daemon=True)

        self.config = config or {}
        self.callback = callback

        # Configuration
        self.interval = float(self.config.get("interval", DEFAULT_INTERVAL))
        self.sigma = float(self.config.get("gauss_sigma", GAUSS_SIGMA))
        self.max_queue = int(self.config.get("max_queue", MAX_QUEUE))

        # Thread control
        self._running = threading.Event()
        self._buf: "queue.Queue[Dict[str, float]]" = queue.Queue(self.max_queue)

        # Baseline for normalization
        self._baseline = baseline or {}

        # Metrics to collect
        self.metrics = self.config.get("metrics", ["cpu", "mem", "net", "disk"])

        if not PSUTIL_AVAILABLE and any(m in self.metrics for m in ["cpu", "mem", "net", "disk"]):
            logger.warning("Some requested metrics require psutil which is not available")

    def run(self) -> None:
        """Thread main loop"""
        self._running.set()
        while self._running.is_set():
            self._sample_once()
            self._process_if_ready()
            time.sleep(self.interval)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the collector thread"""
        self._running.clear()
        self.join(timeout=timeout)

    def _sample_once(self) -> None:
        """Grab a raw metrics snapshot and push to the rolling buffer"""
        if not PSUTIL_AVAILABLE:
            # Fallback metrics if psutil not available
            snap = {
                "timestamp": time.time(),
                "random1": time.time() % 60 / 60.0,  # 0-1 value cycling every minute
                "random2": time.time() % 300 / 300.0,  # 0-1 value cycling every 5 minutes
            }
        else:
            # Full metrics with psutil
            snap = {
                "timestamp": time.time(),
            }

            # CPU metrics
            if "cpu" in self.metrics:
                snap["cpu"] = psutil.cpu_percent(interval=None)

            # Memory metrics
            if "mem" in self.metrics:
                mem = psutil.virtual_memory()
                snap["mem"] = mem.percent
                snap["mem_available"] = mem.available / mem.total * 100

            # Network metrics
            if "net" in self.metrics:
                net = psutil.net_io_counters()
                snap["net_tx"] = net.bytes_sent
                snap["net_rx"] = net.bytes_recv

            # Disk metrics
            if "disk" in self.metrics:
                try:
                    disk = psutil.disk_io_counters()
                    snap["disk_read"] = disk.read_bytes
                    snap["disk_write"] = disk.write_bytes
                except Exception as e:
                    logger.debug(f"Could not collect disk metrics: {e}")

        try:
            self._buf.put_nowait(snap)
        except queue.Full:
            # drop the oldest sample, keep buffer bounded
            self._buf.get_nowait()
            self._buf.put_nowait(snap)

    def _process_if_ready(self) -> None:
        """When we have enough samples, process and emit them"""
        if self._buf.qsize() < 5:
            return

        window = []
        while not self._buf.empty():
            window.append(self._buf.get())

        # Process the window
        smoothed = self._smooth_window(window)

        # Calculate rates for cumulative metrics
        if len(window) >= 2:
            first, last = window[0], window[-1]
            time_diff = last["timestamp"] - first["timestamp"]

            if time_diff > 0:
                # Network rates
                if "net_tx" in last and "net_tx" in first:
                    smoothed["net_tx_rate"] = (last["net_tx"] - first["net_tx"]) / time_diff
                if "net_rx" in last and "net_rx" in first:
                    smoothed["net_rx_rate"] = (last["net_rx"] - first["net_rx"]) / time_diff

                # Disk rates
                if "disk_read" in last and "disk_read" in first:
                    smoothed["disk_read_rate"] = (last["disk_read"] - first["disk_read"]) / time_diff
                if "disk_write" in last and "disk_write" in first:
                    smoothed["disk_write_rate"] = (last["disk_write"] - first["disk_write"]) / time_diff

        # Convert to vector
        metrics_vector = self._to_vector(smoothed)

        # Call callback if provided
        if self.callback:
            self.callback(metrics_vector, smoothed)

    def _smooth_window(self, window: List[Dict[str, float]]) -> Dict[str, float]:
        """Apply smoothing to a window of metrics"""
        if not window:
            return {}

        # Extract metrics that are present in all samples
        keys = set(window[0].keys())
        for sample in window[1:]:
            keys &= set(sample.keys())

        # Remove timestamp from smoothing
        if "timestamp" in keys:
            keys.remove("timestamp")

        if not NUMPY_AVAILABLE or not keys:
            # Simple averaging if NumPy not available
            result = {"timestamp": window[-1]["timestamp"]}
            for key in keys:
                values = [sample[key] for sample in window]
                result[key] = sum(values) / len(values)
            return result

        # NumPy-based smoothing
        # Extract values into arrays
        arrays = {}
        for key in keys:
            arrays[key] = np.array([sample[key] for sample in window])

        # Apply Gaussian smoothing
        result = {"timestamp": window[-1]["timestamp"]}
        for key, values in arrays.items():
            # Skip smoothing for non-numeric values
            if not np.issubdtype(values.dtype, np.number):
                result[key] = window[-1][key]
                continue

            # Apply smoothing
            smoothed = gaussian_filter1d(values, sigma=self.sigma)
            result[key] = float(smoothed[-1])

        return result

    def _to_vector(self, metrics: Dict[str, float]) -> List[float]:
        """Convert metrics dictionary to a vector for alignment analysis"""
        # Define the order of metrics in the vector
        vector_keys = [
            "cpu", "mem", "mem_available",
            "net_tx_rate", "net_rx_rate",
            "disk_read_rate", "disk_write_rate"
        ]

        # Build vector with available metrics
        vector = []
        for key in vector_keys:
            if key in metrics:
                # Normalize if baseline available
                if key in self._baseline and self._baseline[key] > 0:
                    value = metrics[key] / self._baseline[key]
                else:
                    value = metrics[key]
                vector.append(value)
            else:
                vector.append(0.0)

        return vector

    def get_metrics_stream(self) -> Generator[List[float], None, None]:
        """
        Generator that yields metrics vectors as they become available.

        Returns:
            Generator yielding metrics vectors
        """
        metrics_queue = queue.Queue(100)

        def queue_callback(vector, _):
            try:
                metrics_queue.put_nowait(vector)
            except queue.Full:
                # Drop if full
                pass

        # Set callback
        self.callback = queue_callback

        # Start if not running
        if not self._running.is_set():
            self.start()

        # Yield metrics as they become available
        while self._running.is_set():
            try:
                vector = metrics_queue.get(timeout=1.0)
                yield vector
            except queue.Empty:
                # No metrics available, continue waiting
                pass


# Factory function for easy instantiation
def create_collector(config: Optional[Dict[str, Any]] = None) -> LiveMetricsCollector:
    """
    Create a new LiveMetricsCollector instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        LiveMetricsCollector instance
    """
    return LiveMetricsCollector(config)
