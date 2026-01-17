# -*- coding: utf-8 -*-
"""
测试 IFCap EF 模块集成
快速验证 EF 模块是否能正确处理检索到的描述语句
"""

import sys
from utils.ifcap_ef_integration import (
    retrieve_concepts_with_ef,
    retrieve_concepts_with_ef_hybrid,
    extract_entities_from_captions
)

def test_ef_module():
    """测试 EF 模块的基本功能"""
    print("=" * 60)
    print("测试 IFCap EF 模块集成")
    print("=" * 60)
    
    # 模拟检索到的描述语句
    test_captions = [
        "A dog is playing in the park with a ball.",
        "The dog runs across the green grass.",
        "A small dog chases a red ball in the park.",
        "The park has many trees and flowers.",
        "Children are playing in the park near the dog."
    ]
    
    print(f"\n输入描述语句 ({len(test_captions)} 条):")
    for i, cap in enumerate(test_captions, 1):
        print(f"  {i}. {cap}")
    
    # 测试 1: 提取实体
    print("\n" + "-" * 60)
    print("测试 1: 提取实体并统计频率")
    print("-" * 60)
    freq_entities = extract_entities_from_captions(test_captions)
    print(f"\n提取到的实体 (频率, 实体):")
    for freq, entity in freq_entities[:10]:
        print(f"  ({freq}, '{entity}')")
    
    # 测试 2: log_normal 过滤
    print("\n" + "-" * 60)
    print("测试 2: log_normal 过滤方法")
    print("-" * 60)
    concepts_log_normal = retrieve_concepts_with_ef(
        select_memory_captions=test_captions,
        filter_method='log_normal',
        alpha=1.0,
        max_concepts=10
    )
    print(f"\n提取的关键概念 ({len(concepts_log_normal)} 个):")
    for i, concept in enumerate(concepts_log_normal, 1):
        print(f"  {i}. {concept}")
    
    # 测试 3: normal 过滤
    print("\n" + "-" * 60)
    print("测试 3: normal 过滤方法")
    print("-" * 60)
    concepts_normal = retrieve_concepts_with_ef(
        select_memory_captions=test_captions,
        filter_method='normal',
        alpha=1.0,
        max_concepts=10
    )
    print(f"\n提取的关键概念 ({len(concepts_normal)} 个):")
    for i, concept in enumerate(concepts_normal, 1):
        print(f"  {i}. {concept}")
    
    # 测试 4: fixed 过滤
    print("\n" + "-" * 60)
    print("测试 4: fixed 过滤方法 (阈值=2)")
    print("-" * 60)
    concepts_fixed = retrieve_concepts_with_ef(
        select_memory_captions=test_captions,
        filter_method='fixed',
        fixed_threshold=2,
        max_concepts=10
    )
    print(f"\n提取的关键概念 ({len(concepts_fixed)} 个):")
    for i, concept in enumerate(concepts_fixed, 1):
        print(f"  {i}. {concept}")
    
    # 测试 5: 空输入处理
    print("\n" + "-" * 60)
    print("测试 5: 空输入处理")
    print("-" * 60)
    concepts_empty = retrieve_concepts_with_ef(
        select_memory_captions=[],
        filter_method='log_normal',
        max_concepts=10
    )
    print(f"\n空输入返回的默认概念: {concepts_empty}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        test_ef_module()
        print("\n✅ EF 模块集成测试通过！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

