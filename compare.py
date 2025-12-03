import json
import argparse
import sys
import os

def load_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            # 尝试作为整个 JSON 对象读取
            data = json.load(f)
            return data
        except json.JSONDecodeError:
            # 如果失败，尝试作为 JSONL (每一行一个 JSON 对象) 读取
            f.seek(0)
            try:
                lines = f.readlines()
                details = [json.loads(line) for line in lines if line.strip()]
                # 如果是 JSONL，通常没有外层的 details 包装，或者每行就是一个 item
                # 根据示例文件结构，如果每行是一个 item，我们构造成类似的结构
                return {"details": details}
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
                sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Find questions with different 'correct' status between two files.")
    parser.add_argument("file1", help="Path to the first file")
    parser.add_argument("file2", help="Path to the second file")
    
    args = parser.parse_args()
    
    data1 = load_file(args.file1)
    data2 = load_file(args.file2)
    
    # 获取 details 列表
    # 兼容直接是 list 的情况（虽然示例是有 wrapper 的）
    list1 = data1.get('details', {}) if isinstance(data1, dict) else data1
    list2 = data2.get('details', {}) if isinstance(data2, dict) else data2
    
    if not isinstance(list1, list) or not isinstance(list2, list):
        print("Error: Could not find a list of details/items in the files.")
        sys.exit(1)

    # 建立 file1 的查找表: example_abbr -> correct
    dict1 = {}
    for ind, value in list1.items():
        correct = value.get('correct')
        if ind is not None:
            dict1[ind] = correct
            
    diff_ids = []
    
    # 遍历 file2 进行对比
    for ind, value in list2.items():
        correct2 = value.get('correct')
        
        if ind is not None and ind in dict1:
            correct1 = dict1[ind]
            if correct1 != correct2:
                diff_ids.append(ind)
    
    # 打印结果
    if diff_ids:
        for q_id in diff_ids:
            print(q_id)
    else:
        # 如果没有差异，什么也不打印，或者可以打印一条消息（根据需求，通常脚本只输出结果）
        pass

if __name__ == "__main__":
    main()

