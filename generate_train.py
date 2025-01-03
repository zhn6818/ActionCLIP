import os

def generate_file_info(base_path):
    """
    遍历指定目录，生成文件信息列表。
    
    Args:
        base_path (str): 基础路径，包含子文件夹 person、parking、traffic。
    
    Returns:
        list: 包含每个文件夹的文件信息的字符串列表。
    """
    class_mapping = {
        "person": 0,
        "parking": 1,
        "traffic": 2
    }
    result = []

    for folder_name, class_label in class_mapping.items():
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} does not exist or is not a directory.")
            continue

        for root, _, files in os.walk(folder_path):
            file_count = len(files)
            if file_count > 0:
                result.append(f"{root} {file_count} {class_label}")

    return result

def save_to_file(output_path, file_info):
    """
    将生成的文件信息写入文本文件。
    
    Args:
        output_path (str): 输出文件路径。
        file_info (list): 文件信息列表。
    """
    with open(output_path, "w") as f:
        f.write("\n".join(file_info))
    print(f"File information saved to {output_path}")

if __name__ == "__main__":
    # 基础路径
    base_path = "/data1/zhn/dataset/ClipClassData/miniData/miniImages"
    # 输出文件路径
    output_file = "train.txt"

    # 生成文件信息
    file_info = generate_file_info(base_path)
    # 保存到文件
    save_to_file(output_file, file_info)