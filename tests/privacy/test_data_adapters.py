"""数据格式适配器单元测试 / Data Format Adapter Unit Tests.

中文说明：
覆盖 engine.privacy.data_adapters 中的纯函数与多格式提取逻辑，
包括 numpy/pandas/序列/稀疏矩阵/PyArrow IPC/duck-typing 等路径，以及
to_records/from_records 记录转换。SecretFlow 为可选依赖，未安装时相关分支跳过。

English Description:
Unit tests for data_adapters covering numpy/pandas/sequence/sparse/Arrow IPC/
duck-typing extraction paths and record conversions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.privacy import data_adapters as da


class TestToNumpyArray:
    """测试 _to_numpy_array：将各种输入转为 1-D numpy 数组。

    支持输入：list, tuple, ndarray, Series, 嵌套列表。
    数值型转为 float64，非数值型保持原样。
    """

    def test_numeric_list(self):
        """数值列表转为 float64 数组。"""
        arr = da._to_numpy_array([1, 2, 3])
        assert arr.dtype == np.float64
        assert arr.tolist() == [1.0, 2.0, 3.0]

    def test_2d_ravel(self):
        """二维输入通过 ravel 展平为 1-D。"""
        arr = da._to_numpy_array([[1, 2], [3, 4]])
        assert arr.ndim == 1
        assert arr.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_bool_dtype(self):
        """布尔值转为 0.0/1.0 浮点数。"""
        arr = da._to_numpy_array([True, False, True])
        assert arr.dtype == np.float64
        assert arr.tolist() == [1.0, 0.0, 1.0]

    def test_non_numeric_preserved(self):
        """非数值型保持原始类型。"""
        arr = da._to_numpy_array(["a", "b"])
        assert arr.tolist() == ["a", "b"]


class TestTo2DNumpyArray:
    """测试 _to_2d_numpy_array：将各种输入转为 2-D numpy 数组。

    用于 vector_sum 等需要矩阵输入的 DP 操作。
    稀疏矩阵直接透传（不转换为密集矩阵，避免内存爆炸）。
    """

    def test_dataframe(self):
        """DataFrame 转为 (n_rows, n_cols) 的 float64 矩阵。"""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        arr = da._to_2d_numpy_array(df)
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64

    def test_series(self):
        """Series 转为 (n, 1) 列向量。"""
        s = pd.Series([1, 2, 3])
        arr = da._to_2d_numpy_array(s)
        assert arr.shape == (3, 1)

    def test_1d_ndarray(self):
        """一维数组转为 (n, 1) 列向量。"""
        arr = da._to_2d_numpy_array(np.array([1, 2, 3]))
        assert arr.shape == (3, 1)

    def test_2d_ndarray(self):
        """二维数组保持形状，转为 float64。"""
        arr = da._to_2d_numpy_array(np.array([[1, 2], [3, 4]]))
        assert arr.shape == (2, 2)
        assert arr.dtype == np.float64

    def test_sparse_passthrough(self):
        """稀疏矩阵直接透传，不转为密集矩阵。"""
        sp = pytest.importorskip("scipy.sparse")
        mat = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        out = da._to_2d_numpy_array(mat)
        assert sp.issparse(out)

    def test_non_numeric(self):
        """非数值型保持原样。"""
        arr = da._to_2d_numpy_array([["x"], ["y"]])
        assert arr.tolist() == [["x"], ["y"]]


class TestExtractValues:
    """测试 extract_values：统一数据提取接口。

    支持多种输入格式：
    - list/tuple/ndarray: 直接转换
    - Series: 提取值
    - DataFrame: 需指定 column 参数
    - 稀疏矩阵: 透传
    - duck-typing: 尝试 to_numpy() / tolist()
    - Arrow IPC bytes: 解析后提取
    """

    def test_list(self):
        """列表输入转为 float64 数组。"""
        out = da.extract_values([1, 2, 3])
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_tuple(self):
        out = da.extract_values((4, 5))
        assert out.tolist() == [4.0, 5.0]

    def test_ndarray(self):
        out = da.extract_values(np.array([1.5, 2.5]))
        assert out.tolist() == [1.5, 2.5]

    def test_series(self):
        out = da.extract_values(pd.Series([1, 2]))
        assert out.tolist() == [1.0, 2.0]

    def test_dataframe_with_column(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = da.extract_values(df, column="b")
        assert out.tolist() == [3.0, 4.0]

    def test_dataframe_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="not found"):
            da.extract_values(df, column="zzz")

    def test_dataframe_no_column(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="must be specified"):
            da.extract_values(df)

    def test_sparse(self):
        sp = pytest.importorskip("scipy.sparse")
        mat = sp.csr_matrix(np.array([[1.0]]))
        out = da.extract_values(mat)
        assert sp.issparse(out)

    def test_duck_typed_to_numpy(self):
        class Fake:
            def to_numpy(self, zero_copy_only=False):
                return np.array([7, 8])

        out = da.extract_values(Fake())
        assert out.tolist() == [7.0, 8.0]

    def test_duck_typed_to_numpy_typeerror_fallback(self):
        class Fake:
            def to_numpy(self):
                return np.array([9])

        out = da.extract_values(Fake())
        assert out.tolist() == [9.0]

    def test_duck_typed_tolist(self):
        class Fake:
            def tolist(self):
                return [1, 2, 3]

        out = da.extract_values(Fake())
        assert out.tolist() == [1.0, 2.0, 3.0]

    def test_unsupported_type(self):
        with pytest.raises(TypeError, match="Unsupported"):
            da.extract_values(object())


class TestArrowIpc:
    """测试 PyArrow IPC Stream 序列化/反序列化。

    Arrow IPC 是跨语言零拷贝数据交换格式，
    用于 REST 端点的高性能二进制传输。
    """

    def test_roundtrip_with_column(self):
        """序列化→反序列化往返测试（指定列）。"""
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"x": [1, 2, 3], "y": [4, 5, 6]})
        blob = da.table_to_arrow_ipc_bytes(table)
        out = da.parse_arrow_ipc_bytes(blob, column="y")
        assert out.tolist() == [4.0, 5.0, 6.0]

    def test_roundtrip_default_first_column(self):
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"x": [10, 20]})
        blob = da.table_to_arrow_ipc_bytes(table)
        out = da.parse_arrow_ipc_bytes(blob)
        assert out.tolist() == [10.0, 20.0]

    def test_missing_column(self):
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"x": [1]})
        blob = da.table_to_arrow_ipc_bytes(table)
        with pytest.raises(ValueError, match="not found"):
            da.parse_arrow_ipc_bytes(blob, column="nope")

    def test_extract_values_bytes(self):
        pa = pytest.importorskip("pyarrow")
        table = pa.table({"x": [1, 2]})
        blob = da.table_to_arrow_ipc_bytes(table)
        out = da.extract_values(blob)
        assert out.tolist() == [1.0, 2.0]


class TestExtractChunks:
    """测试 extract_chunks：将分块数据转为 numpy 数组列表。"""

    def test_chunks(self):
        """每个分块转为独立的 float64 数组。"""
        out = da.extract_chunks([[1, 2], [3, 4]])
        assert len(out) == 2
        assert out[0].tolist() == [1.0, 2.0]
        assert out[1].tolist() == [3.0, 4.0]


class TestRecords:
    """测试 to_records：将各种数据格式转为 dict 记录列表。

    支持：list of dict, DataFrame, Polars, SecretFlow 等。
    """

    def test_to_records_dict_list(self):
        """dict 列表直接透传。"""
        recs = [{"a": 1}, {"a": 2}]
        assert da.to_records(recs) == recs

    def test_to_records_empty(self):
        assert da.to_records([]) == []

    def test_to_records_non_dict_list(self):
        with pytest.raises(TypeError, match="dict records"):
            da.to_records([1, 2, 3])

    def test_to_records_dataframe(self):
        df = pd.DataFrame({"a": [1, 2]})
        recs = da.to_records(df)
        assert recs == [{"a": 1}, {"a": 2}]

    def test_to_records_unsupported(self):
        with pytest.raises(TypeError, match="Unsupported"):
            da.to_records(object())

    def test_from_records_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        out = da.from_records([{"a": 9}], df)
        assert isinstance(out, pd.DataFrame)
        assert out.iloc[0]["a"] == 9

    def test_from_records_list(self):
        out = da.from_records([{"a": 1}], [{"a": 0}])
        assert out == [{"a": 1}]
