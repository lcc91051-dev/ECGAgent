import inspect
from typing import Callable, Dict, List, Optional, get_type_hints

# Map Python types to JSON Schema types
TYPE_MAP = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null"
}

class FunctionRegistry:
    _instance = None

    def __init__(self):
        # Store name -> function object mapping
        self.functions: Dict[str, Callable] = {}
        # Store metadata for each function, including parameters, return type, description, etc.
        self.metadata: Dict[str, dict] = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, name: Optional[str] = None, description: str = "", parameters: Optional[List[Dict]] = None, **kwargs):
        """
        装饰器：将函数注册到 FunctionRegistry
        """
        def decorator(func: Callable):
            # 1. 自动处理函数名
            actual_name = name if name else func.__name__
            actual_params = []
            if parameters is None:
                try:
                    sig = inspect.signature(func)
                    actual_params = list(sig.parameters.values())
                except Exception as e:
                    print(f"Warning: Could not parse signature for {func.__name__}: {e}")
            else:
                actual_params = parameters

            # 2. 构建 JSON Schema
            properties = {}
            required = []
            
            for param in actual_params:
                # 兼容手动提供的字典和自动提取的 inspect.Parameter
                if isinstance(param, dict):
                    p_name = param["name"]
                    p_type = param.get("type", "string")
                    p_desc = param.get("description", "")
                    p_req = param.get("required", True)
                else:
                    p_name = param.name
                    p_type = "number" if param.annotation in [int, float] else "string"
                    p_desc = f"Parameter {p_name}"
                    p_req = (param.default == inspect.Parameter.empty)

                properties[p_name] = {
                    "type": p_type,
                    "description": p_desc
                }
                if p_req:
                    required.append(p_name)

            schema = {
                "type": "object",
                "properties": properties,
                "required": required
            }

            # 3. 注册到实例
            self.functions[name] = func
            self.metadata[name] = {
                "name": name,
                "description": description,
                "parameters": schema
            }
            return func
        return decorator

    def get_function(self, name: str) -> Callable:
        """Get the function object by name"""
        return self.functions.get(name)

    def get_all_metadata(self) -> List[dict]:
        """Get metadata for all registered functions"""
        return list(self.metadata.values())

    def export_tool_schemas(self) -> List[dict]:
        """
        导出所有注册工具的 JSON Schema，适配 OpenAI/DashScope 格式。
        """
        return list(self.metadata.values())

# Create a global instance
registry = FunctionRegistry.get_instance()
register = registry.register
function_register = registry # Alias for backward compatibility with main.py
