import regex as re
import json
import inspect


def extract_tool_calls(response):
    """
    从 LLM 响应中提取工具调用。
    支持嵌套 JSON 和多种格式变体。
    """
    tool_calls = []
    
    # 使用递归匹配来处理嵌套 JSON
    # (?:{(?:[^{}]|(?:{[^{}]*}))*}) 可以匹配一层嵌套的 JSON
    # 使用 regex 库的递归模式可以处理更深的嵌套
    pattern = r"<FUNCTION>\s*(\w+)\s*<ARGS>\s*(\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})"
    
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        tool_name = match[0].strip()
        args_str = match[1].strip()
        
        try:
            # 清理可能的格式问题
            # 移除可能的 markdown 代码块标记
            args_str = re.sub(r'^```json?\s*', '', args_str)
            args_str = re.sub(r'\s*```$', '', args_str)
            
            # 尝试解析 JSON
            args = json.loads(args_str)
            tool_calls.append({
                "name": tool_name,
                "args": args
            })
        except json.JSONDecodeError as e:
            print(f"Parameter parsing failed for {tool_name}: {e}")
            print(f"  Raw args string: {args_str[:100]}...")
            
            # 尝试修复常见的 JSON 格式问题
            try:
                # 尝试移除尾部逗号
                fixed_str = re.sub(r',\s*}', '}', args_str)
                fixed_str = re.sub(r',\s*]', ']', fixed_str)
                args = json.loads(fixed_str)
                tool_calls.append({
                    "name": tool_name,
                    "args": args
                })
                print(f"  Successfully parsed after fixing trailing commas")
            except json.JSONDecodeError:
                # 如果仍然失败，跳过这个调用
                print(f"  Could not recover, skipping this tool call")
                continue
    
    return tool_calls


def has_config_parameter(func):
    """检查函数是否有 config 参数"""
    return 'config' in inspect.signature(func).parameters
