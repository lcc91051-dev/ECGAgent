import json


def messageMerge(function_return, messages):
    """
    将工具调用结果合并到消息列表中。
    
    Args:
        function_return: 工具调用结果列表
        messages: 对话消息列表
    """
    if not function_return:
        return
    
    tool_response_text = "\n\n"
    for item in function_return:
        tool_response_text += f"<FUNCTION>\n{item['name']}\n"
        tool_response_text += f"<ARGS>\n{json.dumps(item['args'], ensure_ascii=False, indent=2)}\n"
        tool_response_text += f"<RETURN>\n{json.dumps(item['return'], ensure_ascii=False, indent=2)}\n"
        tool_response_text += "</RESULT>\n"
    
    # 安全地查找最后一条 assistant 消息
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'assistant':
            messages[i]['content'] += tool_response_text
            return
    
    # 如果没有找到 assistant 消息，添加一条新的
    messages.append({
        'role': 'assistant',
        'content': tool_response_text.strip()
    })
