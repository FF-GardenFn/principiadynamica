# IO API Reference

The `northstar.io` module provides input/output functionality for loading embeddings, detecting time series, and collecting live metrics.

## Loaders

The `northstar.io.loaders` module provides functions for loading embeddings and aligned examples from various file formats.

### `load_embeddings`

```python
load_embeddings(file_path: str) -> Dict[str, Union[List[float], Dict[str, List[float]]]]
```

Load embeddings from a file in various formats.

**Parameters**:
- **file_path** (`str`): Path to the embeddings file

**Returns**:
- `Dict[str, Union[List[float], Dict[str, List[float]]]]`: Dictionary mapping identifiers to embeddings

**Supported Formats**:
- Direct mapping of ID to embedding vector
- Report format with summaries containing embeddings
- List of objects with path and embedding
- File similarities format

### `aligned_examples`

```python
aligned_examples(file_path: str, dimension: int = 1024) -> List[List[float]]
```

Load examples of aligned behavior from a file.

**Parameters**:
- **file_path** (`str`): Path to the aligned examples file
- **dimension** (`int`, default=1024): Dimensionality of the embedding space

**Returns**:
- `List[List[float]]`: List of aligned example vectors

**Supported Formats**:
- List of embeddings
- Dictionary with "aligned_examples" key
- Dictionary mapping IDs to embeddings
- List of objects with embedding

## Time Series

The `northstar.io.timeseries` module provides functions for detecting and ordering time-series data from embeddings.

### `detect_and_order_time_series`

```python
detect_and_order_time_series(embeddings: Dict[str, List[float]], prefix_pattern: Optional[str] = None) -> List[Dict[str, Any]]
```

Detect and order embeddings as a time series if they follow a pattern.

**Parameters**:
- **embeddings** (`Dict[str, List[float]]`): Dictionary of embeddings
- **prefix_pattern** (`str`, optional): Optional prefix pattern to identify time series

**Returns**:
- `List[Dict[str, Any]]`: List of ordered states with the following keys:
  - `id`: Identifier for the state
  - `embedding`: Embedding vector
  - `timestamp`: Timestamp for the state
  - `index`: Index in the time series

**Detection Methods**:
- Numeric suffix pattern (e.g., "state_1", "state_2", etc.)
- Timestamp pattern in keys (e.g., "2023-01-01T12:00:00")
- Fallback to arbitrary order if no pattern detected

### `extract_subsequence`

```python
extract_subsequence(time_series: List[Dict[str, Any]], start_idx: Optional[int] = None, end_idx: Optional[int] = None, window_size: Optional[int] = None) -> List[Dict[str, Any]]
```

Extract a subsequence from a time series.

**Parameters**:
- **time_series** (`List[Dict[str, Any]]`): List of time series states
- **start_idx** (`int`, optional): Optional start index
- **end_idx** (`int`, optional): Optional end index
- **window_size** (`int`, optional): Optional window size (used if start_idx is provided but end_idx is not)

**Returns**:
- `List[Dict[str, Any]]`: Subsequence of the time series

## Live Metrics

The `northstar.io.live` module provides a background collector that samples system metrics and feeds them into the alignment vector space as a live data source.

### `LiveMetricsCollector`

```python
class LiveMetricsCollector(config=None, callback=None, baseline=None)
```

Background worker that samples host metrics and provides them as a stream of vectors for alignment analysis.

**Parameters**:
- **config** (`Dict[str, Any]`, optional): Configuration dictionary
- **callback** (`callable`, optional): Optional callback function to receive metrics
- **baseline** (`Dict[str, float]`, optional): Optional baseline metrics for normalization

#### Methods

##### `run`

```python
run() -> None
```

Thread main loop that collects metrics at regular intervals.

##### `stop`

```python
stop(timeout: float = 2.0) -> None
```

Stop the collector thread.

**Parameters**:
- **timeout** (`float`, default=2.0): Timeout for joining the thread

##### `get_metrics_stream`

```python
get_metrics_stream() -> Generator[List[float], None, None]
```

Generator that yields metrics vectors as they become available.

**Returns**:
- `Generator[List[float], None, None]`: Generator yielding metrics vectors

**Example**:
```python
collector = LiveMetricsCollector()
collector.start()
for vector in collector.get_metrics_stream():
    # Process the metrics vector
    print(vector)
```

### `create_collector`

```python
create_collector(config: Optional[Dict[str, Any]] = None) -> LiveMetricsCollector
```

Create a new LiveMetricsCollector instance.

**Parameters**:
- **config** (`Dict[str, Any]`, optional): Optional configuration dictionary

**Returns**:
- `LiveMetricsCollector`: LiveMetricsCollector instance

**Configuration Options**:
- `interval`: Seconds between samples (default: 1.0)
- `gauss_sigma`: Smoothing factor for Gaussian filter (default: 2.0)
- `max_queue`: Maximum queue size for samples (default: 512)
- `metrics`: List of metrics to collect (default: ["cpu", "mem", "net", "disk"])

**Example**:
```python
config = {
    "interval": 0.5,  # Sample every 0.5 seconds
    "gauss_sigma": 1.5,  # Less smoothing
    "metrics": ["cpu", "mem"]  # Only collect CPU and memory metrics
}
collector = create_collector(config)
```

## Metrics Format

The metrics vectors produced by the LiveMetricsCollector have the following format:

1. CPU usage (0-100%)
2. Memory usage (0-100%)
3. Memory available (0-100%)
4. Network TX rate (bytes/second)
5. Network RX rate (bytes/second)
6. Disk read rate (bytes/second)
7. Disk write rate (bytes/second)

These metrics are normalized if a baseline is provided, making them suitable for alignment analysis.
