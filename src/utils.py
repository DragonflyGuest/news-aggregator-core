def extract_by_label(data, target_label):
    """
    从列表中提取具有特定标签的项。

    参数:
    - data: 一个包含字典的列表，每个字典包含 'text' 和 'label' 键。
    - target_label: 要提取的标签名（字符串）。

    返回:
    - 提取出的文本项列表。
    """
    return [item['text'] for item in data if item['label'] == target_label]
