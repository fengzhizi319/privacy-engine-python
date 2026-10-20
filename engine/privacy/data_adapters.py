"""数据格式适配器。

中文说明：
为 DP 等隐私原语提供统一的数据输入适配，支持从多种数据格式中提取目标列：

- Python list/tuple
- numpy ndarray
- pandas Series/DataFrame
- SecretFlow DataFrame / HDataFrame / VDataFrame / MixDataFrame / FedNdarray
- scipy.sparse 稀疏矩阵
- PyArrow IPC Stream 二进制流
- Polars / duck-typing 兼容格式

SecretFlow 相关依赖为可选依赖，未安装时不会影响其他格式的支持。

English Description:
Unified data input adapter for privacy primitives (DP, K-anonymity, etc.).
Extracts target columns from multiple data formats including Python sequences,
numpy, pandas, SecretFlow federated structures, scipy.sparse, PyArrow IPC,
and Polars. SecretFlow dependencies are optional and degrade gracefully.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..observability.logging_config import get_logger
from ..observability.metrics import DATA_EXTRACTION_TOTAL

if TYPE_CHECKING:
    from collections.abc import Iterable

# Module-level structured logger for data adapter events
logger = get_logger(__name__)


def _is_secretflow_available() -> bool:
    """检测 SecretFlow 是否可用 / Check if SecretFlow is Available.

    Returns:
        True 如果 secretflow 已安装 / True if secretflow is installed.
    """
    try:
        import secretflow  # noqa: F401
        return True
    except ImportError:
        return False


def _extract_from_secretflow(
    data: Any,
    column: str | None = None,
    party: str | None = None,
) -> np.ndarray:
    """从 SecretFlow 数据结构中抽取目标列 / Extract Column from SecretFlow Data.

    中文说明：
    支持 FedNdarray、HDataFrame、VDataFrame 及本地 sf.data.DataFrame。

    English Description:
    Extracts target column from SecretFlow federated data structures including
    FedNdarray, HDataFrame, VDataFrame, and local sf.data.DataFrame.

    Args:
        data: SecretFlow 数据结构 / SecretFlow data structure.
        column: 目标列名 / Target column name.
        party: 参与方标识 / Party identifier for HDataFrame.

    Returns:
        一维 np.ndarray / 1-D np.ndarray.

    Raises:
        ValueError: 缺少必要参数 / Missing required parameters.
        TypeError: 不支持的数据类型 / Unsupported data type.
    """
    import secretflow as sf

    # FedNdarray: 联邦 ndarray，底层是 numpy
    if hasattr(data, "partitions") and hasattr(data, "partition_way"):
        # HDataFrame / VDataFrame / MixDataFrame
        return _extract_from_sf_dataframe(data, column=column, party=party)

    # 本地 DataFrame（sf.data.DataFrame 底层通常是 pandas）
    if isinstance(data, sf.data.DataFrame):
        if column is None:
            raise ValueError(
                "column must be specified when input is a SecretFlow DataFrame"
            )
        return _to_numpy_array(data[column].to_numpy())

    raise TypeError(f"Unsupported SecretFlow data type: {type(data)}")


def _extract_from_sf_dataframe(
    data: Any,
    column: str | None = None,
    party: str | None = None,
) -> np.ndarray:
    """从 SecretFlow 联邦 DataFrame 中提取数据 / Extract from SecretFlow Federated DataFrame.

    中文说明：
    支持 HDataFrame（水平分割）、VDataFrame（垂直分割），MixDataFrame 不支持直接提取。

    English Description:
    Supports HDataFrame (horizontal partition) and VDataFrame (vertical partition).
    MixDataFrame is not directly supported.

    Args:
        data: SecretFlow 联邦 DataFrame / SecretFlow federated DataFrame.
        column: 目标列名 / Target column name.
        party: 参与方标识 / Party identifier.

    Returns:
        一维 np.ndarray / 1-D np.ndarray.
    """
    from secretflow.data.horizontal import HDataFrame
    from secretflow.data.mix import MixDataFrame
    from secretflow.data.vertical import VDataFrame

    if column is None:
        raise ValueError(
            "column must be specified when input is a SecretFlow federated DataFrame"
        )

    if isinstance(data, HDataFrame):
        return _extract_from_hdataframe(data, column=column, party=party)

    if isinstance(data, VDataFrame):
        return _extract_from_vdataframe(data, column=column)

    if isinstance(data, MixDataFrame):
        raise TypeError(
            "MixDataFrame is not directly supported; please convert to "
            "HDataFrame/VDataFrame or extract values manually"
        )

    raise TypeError(f"Unsupported SecretFlow federated DataFrame type: {type(data)}")


def _extract_from_hdataframe(
    data: Any,
    column: str,
    party: str | None = None,
) -> np.ndarray:
    """从 HDataFrame 中提取指定列 / Extract Column from HDataFrame.

    中文说明：
    HDataFrame 是水平分割：各参与方拥有相同样本空间的不同子集。
    若未指定 party 且只有一个 partition，则自动提取；否则必须指定 party。

    English Description:
    HDataFrame is horizontally partitioned: each party holds a different subset
    of the same sample space. Auto-selects if single partition; otherwise party
    must be specified.

    Args:
        data: HDataFrame 实例 / HDataFrame instance.
        column: 目标列名 / Target column name.
        party: 参与方标识 / Party identifier.

    Returns:
        一维 np.ndarray / 1-D np.ndarray.
    """
    partitions = dict(data.partitions)
    if party is None:
        if len(partitions) == 1:
            partition = next(iter(partitions.values()))
        else:
            raise ValueError(
                "party must be specified when HDataFrame has multiple partitions"
            )
    else:
        # party 可以是 str(device name) 或 PYU 对象
        key = None
        for k in partitions:
            if str(k) == party or getattr(k, "device_id", None) == party:
                key = k
                break
        if key is None:
            raise ValueError(f"party '{party}' not found in HDataFrame partitions")
        partition = partitions[key]

    pdf = partition.data
    if column not in pdf.columns:
        raise ValueError(f"column '{column}' not found in HDataFrame partition")
    return _to_numpy_array(pdf[column].to_numpy())


def _extract_from_vdataframe(data: Any, column: str) -> np.ndarray:
    """从 VDataFrame 中提取指定列 / Extract Column from VDataFrame.

    中文说明：
    VDataFrame 是垂直分割：列分布在不同参与方。系统会自动找到包含该列的 partition。

    English Description:
    VDataFrame is vertically partitioned: columns are distributed across parties.
    Automatically locates the partition containing the target column.

    Args:
        data: VDataFrame 实例 / VDataFrame instance.
        column: 目标列名 / Target column name.

    Returns:
        一维 np.ndarray / 1-D np.ndarray.
    """
    for partition in data.partitions.values():
        pdf = partition.data
        if column in pdf.columns:
            return _to_numpy_array(pdf[column].to_numpy())
    raise ValueError(f"column '{column}' not found in any VDataFrame partition")


def _is_sparse_matrix(data: Any) -> bool:
    """判断是否为 scipy.sparse 稀疏矩阵 / Check if Data is a Scipy Sparse Matrix.

    Returns:
        True 如果是稀疏矩阵 / True if data is a scipy sparse matrix.
    """
    try:
        import scipy.sparse as sp

        return bool(sp.issparse(data))
    except ImportError:
        return False


def _to_numpy_array(arr: Any) -> np.ndarray:
    """将输入数组转换为 np.float64 ndarray / Convert Input to np.float64 ndarray.

    中文说明：
    尝试将输入转换为 float64 一维数组，无法转数值时保留 object 类型。

    English Description:
    Attempts to convert input to a 1-D float64 array; preserves object dtype
    when numeric conversion is not possible.

    Args:
        arr: 输入数组 / Input array-like data.

    Returns:
        一维 np.ndarray / 1-D np.ndarray.
    """
    np_arr = np.asarray(arr)
    if np_arr.ndim != 1:
        np_arr = np_arr.ravel()
    if np.issubdtype(np_arr.dtype, np.number) or np.issubdtype(np_arr.dtype, np.bool_):
        return np_arr.astype(np.float64, copy=False)
    try:
        return np_arr.astype(np.float64)
    except (ValueError, TypeError):
        return np_arr


def _to_2d_numpy_array(data: Any) -> Any:
    """将多种表格或 2D 数据源转换为二维 np.ndarray / Convert Tabular Data to 2-D np.ndarray.

    中文说明：
    支持 pandas DataFrame/Series、numpy ndarray、scipy.sparse 矩阵等输入。

    English Description:
    Converts various tabular or 2-D data sources to a 2-D float64 np.ndarray
    or preserves scipy.sparse matrix format.

    Args:
        data: 输入数据 / Input data (DataFrame, ndarray, sparse matrix, etc.).

    Returns:
        二维 np.ndarray (float64) 或 scipy.sparse 矩阵 / 2-D np.ndarray or sparse matrix.
    """
    if _is_sparse_matrix(data):
        return data

    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.to_numpy(dtype=np.float64)
        if isinstance(data, pd.Series):
            return data.to_numpy(dtype=np.float64).reshape(-1, 1)
    except ImportError:
        pass

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        return arr.astype(np.float64, copy=False)
    try:
        return arr.astype(np.float64)
    except (ValueError, TypeError):
        return arr


def extract_values(
    data: Any,
    column: str | None = None,
    party: str | None = None,
) -> np.ndarray | Any:
    """从多种数据格式中提取数值或类别数组 / Extract Values from Multiple Data Formats.

    中文说明：
    统一的数据提取入口，支持 list/tuple/ndarray/Series/DataFrame/SecretFlow/sparse/arrow/polars。
    返回一维 np.ndarray 或 scipy.sparse 矩阵。

    English Description:
    Unified data extraction entry point supporting list/tuple/ndarray/Series/
    DataFrame/SecretFlow/sparse/arrow/polars formats. Returns a 1-D np.ndarray
    or scipy.sparse matrix.

    执行步骤 / Execution Steps:
    1. 检测 scipy.sparse 稀疏矩阵，直接返回。
       (Detect scipy.sparse matrix, return directly)
    2. 检测 numpy ndarray，转换并返回。
       (Detect numpy ndarray, convert and return)
    3. 检测 Python 原生序列，转换并返回。
       (Detect Python native sequence, convert and return)
    4. 检测 pandas Series/DataFrame，提取目标列。
       (Detect pandas Series/DataFrame, extract target column)
    5. 检测 SecretFlow 数据结构（可选依赖）。
       (Detect SecretFlow data structures, optional dependency)
    6. 通用 duck-typing 回退 (Polars/PyArrow)。
       (Generic duck-typing fallback for Polars/PyArrow)
    7. 记录指标并返回结果。
       (Record metrics and return result)

    Args:
        data: 输入数据 / Input data.
            支持 list/tuple/ndarray/Series/DataFrame/SecretFlow/sparse/arrow/polars 格式。
        column: 当 data 为表格类型且需要指定列时使用 / Column name for tabular input.
        party: 当 data 为 HDataFrame 且需要指定参与方时使用 / Party for HDataFrame.

    Returns:
        一维 np.ndarray 数组或 scipy.sparse 矩阵 / 1-D np.ndarray or scipy.sparse matrix.

    Raises:
        ValueError: 缺少必要的 column 参数 / Missing required column parameter.
        TypeError: 不支持的数据类型 / Unsupported data type.
    """
    # 0. scipy.sparse 稀疏矩阵
    if _is_sparse_matrix(data):
        DATA_EXTRACTION_TOTAL.labels(format="sparse", status="success").inc()
        return data

    # 1. numpy ndarray
    if isinstance(data, np.ndarray):
        DATA_EXTRACTION_TOTAL.labels(format="numpy", status="success").inc()
        return _to_numpy_array(data)

    # 2. Python 原生序列
    if isinstance(data, (list, tuple)):
        DATA_EXTRACTION_TOTAL.labels(format="sequence", status="success").inc()
        return _to_numpy_array(data)

    # 3. pandas Series / DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.Series):
            DATA_EXTRACTION_TOTAL.labels(format="pandas_series", status="success").inc()
            return _to_numpy_array(data.to_numpy())
        if isinstance(data, pd.DataFrame):
            if column is None:
                DATA_EXTRACTION_TOTAL.labels(format="pandas_dataframe", status="error").inc()
                raise ValueError(
                    "column must be specified when input is a pandas DataFrame"
                )
            if column not in data.columns:
                DATA_EXTRACTION_TOTAL.labels(format="pandas_dataframe", status="error").inc()
                raise ValueError(f"column '{column}' not found in DataFrame")
            DATA_EXTRACTION_TOTAL.labels(format="pandas_dataframe", status="success").inc()
            return _to_numpy_array(data[column].to_numpy())
    except ImportError:
        pass

    # 4. SecretFlow 数据结构（可选依赖）
    if _is_secretflow_available():
        DATA_EXTRACTION_TOTAL.labels(format="secretflow", status="success").inc()
        return _extract_from_secretflow(data, column=column, party=party)

    # 5. 通用 duck-typing 回退 (Polars / PyArrow / Zero-copy)
    if hasattr(data, "to_numpy"):
        try:
            DATA_EXTRACTION_TOTAL.labels(format="duck_typed", status="success").inc()
            return _to_numpy_array(data.to_numpy(zero_copy_only=False))
        except TypeError:
            return _to_numpy_array(data.to_numpy())
    if hasattr(data, "tolist"):
        DATA_EXTRACTION_TOTAL.labels(format="duck_typed", status="success").inc()
        return _to_numpy_array(data.tolist())

    # 6. PyArrow RecordBatch / Table Stream / Bytes
    if isinstance(data, (bytes, bytearray)):
        DATA_EXTRACTION_TOTAL.labels(format="arrow_ipc", status="success").inc()
        return parse_arrow_ipc_bytes(data, column=column)

    DATA_EXTRACTION_TOTAL.labels(format="unknown", status="error").inc()
    logger.warning(
        "data_extraction_unsupported_type",
        extra={"data_type": str(type(data))},
    )
    raise TypeError(f"Unsupported data type for DP values: {type(data)}")


def parse_arrow_ipc_bytes(b: bytes | bytearray, column: str | None = None) -> np.ndarray:
    """从 PyArrow IPC Stream 字节流解析数据 / Parse PyArrow IPC Stream Bytes.

    中文说明：
    从二进制 Arrow IPC RecordBatch Stream 中提取目标列为 np.ndarray。

    English Description:
    Parses binary Arrow IPC RecordBatch Stream bytes and extracts the target
    column as a np.ndarray.

    执行步骤 / Execution Steps:
    1. 使用 `pyarrow.ipc.open_stream` 读取二进制 Stream。
       (Read binary stream via pyarrow.ipc.open_stream)
    2. 将 RecordBatch Stream 还原为 `pyarrow.Table`。
       (Restore RecordBatch Stream to pyarrow.Table)
    3. 若指定 column 则提取目标列，否则默认提取第一列。
       (Extract target column if specified, otherwise first column)

    Args:
        b: Arrow IPC 二进制字节流 / Arrow IPC binary bytes.
        column: 目标列名 / Target column name (optional, defaults to first column).

    Returns:
        一维 np.ndarray / 1-D np.ndarray.

    Raises:
        ValueError: 指定列不存在 / Specified column not found.
    """
    import pyarrow.ipc as ipc

    reader = ipc.RecordBatchStreamReader(b)
    table = reader.read_all()
    if column is not None:
        if column not in table.column_names:
            raise ValueError(f"Column '{column}' not found in Arrow IPC Table")
        arr = table.column(column).to_numpy()
    else:
        if table.num_columns == 0:
            return np.array([], dtype=np.float64)
        arr = table.column(0).to_numpy()
    return _to_numpy_array(arr)


def table_to_arrow_ipc_bytes(table: Any) -> bytes:
    """将 PyArrow Table 转换为 Arrow IPC 二进制字节流 / Convert PyArrow Table to IPC Bytes.

    中文说明：
    将 PyArrow Table 序列化为 Arrow IPC RecordBatch Stream 二进制格式。

    English Description:
    Serializes a PyArrow Table into Arrow IPC RecordBatch Stream binary format.

    执行步骤 / Execution Steps:
    1. 使用 `pyarrow.ipc.new_stream` 打开二进制 Sink 缓存区。
       (Open binary sink buffer via pyarrow.ipc.new_stream)
    2. 写入 table 及其全量 Schema Metadata。
       (Write table with full schema metadata)
    3. 导出包含全量数据的二进制 bytes。
       (Export binary bytes containing all data)

    Args:
        table: PyArrow Table 实例 / PyArrow Table instance.

    Returns:
        Arrow IPC 二进制字节流 / Arrow IPC binary bytes.
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return cast("bytes", sink.getvalue().to_pybytes())


def extract_chunks(
    chunks: Iterable[Any],
    column: str | None = None,
    party: str | None = None,
) -> list[np.ndarray]:
    """对分块输入逐块提取数值 / Extract Values from Chunked Input.

    中文说明：
    对可迭代的分块输入逐块调用 extract_values，返回各块结果列表。

    English Description:
    Iterates over chunked input, calling extract_values on each chunk and
    returning a list of per-chunk results.

    Args:
        chunks: 可迭代的数据块 / Iterable of data chunks.
        column: 目标列名 / Target column name.
        party: 参与方标识 / Party identifier.

    Returns:
        各块提取结果列表 / List of per-chunk np.ndarray results.
    """
    return [extract_values(chunk, column=column, party=party) for chunk in chunks]


def _extract_dataframe_partition(data: Any) -> Any:
    """从 SecretFlow DataFrame 中提取单个 pandas DataFrame / Extract Single pandas DataFrame.

    中文说明：
    从 SecretFlow HDataFrame / VDataFrame 中提取单个 pandas DataFrame。
    HDataFrame 多 partition 时需要指定 party；单 partition 自动选择。

    English Description:
    Extracts a single pandas DataFrame from SecretFlow HDataFrame / VDataFrame.
    HDataFrame with multiple partitions requires party specification.

    Args:
        data: SecretFlow DataFrame 实例 / SecretFlow DataFrame instance.

    Returns:
        pandas DataFrame / pandas DataFrame.

    Raises:
        ValueError: 多 partition 未指定 party / Multiple partitions without party.
        TypeError: 不支持的数据类型 / Unsupported data type.
    """
    import secretflow as sf
    from secretflow.data.horizontal import HDataFrame
    from secretflow.data.vertical import VDataFrame

    if isinstance(data, HDataFrame):
        partitions = dict(data.partitions)
        if len(partitions) == 1:
            return next(iter(partitions.values())).data
        raise ValueError(
            "party must be specified when input HDataFrame has multiple partitions"
        )

    if isinstance(data, VDataFrame):
        # VDataFrame 中所有列都在同一个参与方，取第一个 partition
        return next(iter(data.partitions.values())).data

    if isinstance(data, sf.data.DataFrame):
        return data

    raise TypeError(f"Unsupported SecretFlow data type: {type(data)}")


def to_records(
    data: Any,
    party: str | None = None,
) -> list[dict[str, Any]]:
    """将多种表格型输入统一转换为记录列表 / Convert Tabular Input to Record List.

    中文说明：
    将多种表格型输入统一转换为 List[Dict[str, Any]] 记录列表。

    English Description:
    Converts various tabular inputs to a unified List[Dict[str, Any]] record list.

    执行步骤 / Execution Steps:
    1. 检测原生 dict 列表，直接返回。
       (Detect native dict list, return directly)
    2. 检测 pandas DataFrame，调用 to_dict(orient="records")。
       (Detect pandas DataFrame, call to_dict(orient="records"))
    3. 检测 SecretFlow 数据结构（可选依赖）。
       (Detect SecretFlow data structures, optional dependency)

    Args:
        data: 输入数据 / Input data (list of dict, DataFrame, SecretFlow).
        party: SecretFlow HDataFrame 多 partition 时指定参与方 / Party for multi-partition HDataFrame.

    Returns:
        记录列表 / List of record dicts.

    Raises:
        TypeError: 不支持的输入类型 / Unsupported input type.
    """
    # 1. 原生 dict 列表
    if isinstance(data, (list, tuple)):
        if not data:
            return []
        if all(isinstance(r, dict) for r in data):
            return list(data)
        raise TypeError("list input must contain dict records")

    # 2. pandas DataFrame
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return cast("list[dict[str, Any]]", data.to_dict(orient="records"))
    except ImportError:
        pass

    # 3. SecretFlow 数据结构（可选依赖）
    if _is_secretflow_available():
        pdf = _extract_dataframe_partition(data)
        if isinstance(pdf, pd.DataFrame):
            return cast("list[dict[str, Any]]", pdf.to_dict(orient="records"))
        # SecretFlow 本地 DataFrame 可能直接是 pandas-like
        return cast("list[dict[str, Any]]", pdf.to_dict(orient="records"))

    raise TypeError(f"Unsupported table input type: {type(data)}")


def from_records(
    records: list[dict[str, Any]], original: Any
) -> Any:
    """根据原始输入类型将 records 转换回对应格式 / Convert Records Back to Original Format.

    中文说明：
    根据原始输入类型将处理后的记录列表转换回对应格式。

    English Description:
    Converts processed record list back to the format matching the original input type.

    执行步骤 / Execution Steps:
    1. 检测原始输入是否为 pandas DataFrame，返回 DataFrame。
       (Check if original is pandas DataFrame, return DataFrame)
    2. 检测原始输入是否为 SecretFlow DataFrame，返回本地 pandas 副本。
       (Check if original is SecretFlow DataFrame, return local pandas copy)
    3. 其他情况返回原始 records 列表。
       (Otherwise return records list as-is)

    Args:
        records: 处理后的记录列表 / Processed record list.
        original: 原始输入数据（用于类型推断） / Original input data (for type inference).

    Returns:
        与原始输入格式匹配的结果 / Result matching original input format.
    """
    # pandas / SecretFlow DataFrame 统一返回 pandas DataFrame
    try:
        import pandas as pd

        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(records)
    except ImportError:
        pass

    if _is_secretflow_available():
        import pandas as pd
        from secretflow.data.horizontal import HDataFrame
        from secretflow.data.vertical import VDataFrame

        if isinstance(original, (HDataFrame, VDataFrame)) or hasattr(original, "partitions"):
            return pd.DataFrame(records)

    if isinstance(original, (list, tuple)):
        return records

    return records
