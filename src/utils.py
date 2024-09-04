from collections import Counter

def count_and_sort_entities(entities):
    # 提取实体类型部分
    entity_types = [entity[0] for entity in entities if entity[1] == "PERSON"]

    # 使用 Counter 统计实体类型的出现次数
    count = Counter(entity_types)

    # 对统计结果进行排序，按照出现次数从高到低排序
    sorted_count = sorted(count.items(), key=lambda x: x[1], reverse=True)

    return sorted_count

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
