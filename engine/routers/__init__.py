"""按域拆分的 REST 子路由包。

每个模块对应一组语义相关的端点，导出一个 ``APIRouter`` 实例，
由 ``main.py`` 通过 ``include_router`` 统一挂载。
"""
