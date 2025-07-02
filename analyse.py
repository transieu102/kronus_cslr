import pandas as pd
from collections import defaultdict

# Đường dẫn tới 2 file CSV
file1 = './data/us/train.csv'  # Thay bằng đường dẫn thực tế
file2 = './data/us/dev.csv'  # Thay bằng đường dẫn thực tế

# Load dữ liệu

def load_gloss_lists(csv_path):
    df = pd.read_csv(csv_path)
    gloss_lists = df['gloss'].astype(str).apply(lambda x: x.split()).tolist()
    return gloss_lists

gloss_lists1 = load_gloss_lists(file1)
gloss_lists2 = load_gloss_lists(file2)

max_len = 12

def get_position_sets(gloss_lists, max_len):
    pos_sets = [set() for _ in range(max_len)]
    for gloss_list in gloss_lists:
        for i in range(min(len(gloss_list), max_len)):
            pos_sets[i].add(gloss_list[i])
    return pos_sets

pos_sets1 = get_position_sets(gloss_lists1, max_len)
pos_sets2 = get_position_sets(gloss_lists2, max_len)

print(f"Thống kê gloss set ở từng vị trí (max_len={max_len}):\n")
for i in range(max_len):
    set1 = pos_sets1[i]
    set2 = pos_sets2[i]
    overlap = set1 & set2
    print(f"Vị trí {i+1}:")
    print(f"  Tập 1: {len(set1)} glosses")
    print(f"  Tập 2: {len(set2)} glosses")
    print(f"  Overlap: {len(overlap)} glosses")
    print(f"  Có overlap: {'Có' if overlap else 'Không'}\n")
