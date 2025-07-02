import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def analyze_durations(csv_file):
    # Đọc file CSV
    df = pd.read_csv(csv_file)
    
    # Tạo thư mục để lưu các biểu đồ
    import os
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # Phân tích tổng quan
    print("\n=== TỔNG QUAN DỮ LIỆU ===")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số lượng gloss khác nhau: {df['gloss'].nunique()}")
    print("\nThống kê duration:")
    print(df['duration'].describe())
    
    # Phân tích theo từng gloss
    gloss_stats = df.groupby('gloss')['duration'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\n=== THỐNG KÊ THEO TỪNG GLOSS ===")
    print(gloss_stats)
    
    # Vẽ biểu đồ boxplot cho tất cả các gloss
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='gloss', y='duration', data=df)
    plt.xticks(rotation=90)
    plt.title('Phân phối Duration theo Gloss')
    plt.tight_layout()
    plt.savefig('analysis_results/duration_by_gloss.png')
    plt.close()
    
    # Vẽ histogram tổng quát của duration
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='duration', bins=50)
    plt.title('Phân phối Duration')
    plt.savefig('analysis_results/duration_histogram.png')
    plt.close()
    
    # Xác định outliers cho mỗi gloss sử dụng phương pháp IQR
    outliers = []
    for gloss in df['gloss'].unique():
        gloss_data = df[df['gloss'] == gloss]
        Q1 = gloss_data['duration'].quantile(0.25)
        Q3 = gloss_data['duration'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        gloss_outliers = gloss_data[
            (gloss_data['duration'] < lower_bound) | 
            (gloss_data['duration'] > upper_bound)
        ]
        
        if not gloss_outliers.empty:
            outliers.append(gloss_outliers)
    
    if outliers:
        outliers_df = pd.concat(outliers)
        print("\n=== PHÁT HIỆN OUTLIERS ===")
        print(f"Tổng số outliers: {len(outliers_df)}")
        print("\nChi tiết outliers:")
        print(outliers_df[['gloss', 'duration', 'video_id']])
        
        # Lưu outliers vào file CSV
        outliers_df.to_csv('analysis_results/outliers.csv', index=False)
        
        # Vẽ biểu đồ scatter plot để trực quan hóa outliers
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=df, x='gloss', y='duration', alpha=0.5)
        sns.scatterplot(data=outliers_df, x='gloss', y='duration', color='red', marker='x')
        plt.xticks(rotation=90)
        plt.title('Outliers trong Duration theo Gloss')
        plt.tight_layout()
        plt.savefig('analysis_results/outliers_visualization.png')
        plt.close()
    else:
        print("\nKhông phát hiện outliers nào trong dữ liệu.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python analyze_duration.py <csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_durations(csv_file) 