import pandas as pd
import numpy as np
from scipy import stats

def filter_common_durations(input_csv, output_csv, percentile_range=20):
    """
    Filter segments to keep only the most common durations for each gloss.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file
        percentile_range (int): Percentage of most common durations to keep (default: 20)
    """
    # Read input CSV
    df = pd.read_csv(input_csv)
    print(f"\n=== TỔNG QUAN DỮ LIỆU GỐC ===")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số lượng gloss khác nhau: {df['gloss'].nunique()}")
    
    # Initialize list to store filtered data
    filtered_data = []
    
    # Process each gloss
    for gloss in df['gloss'].unique():
        gloss_data = df[df['gloss'] == gloss]
        
        # Calculate the most common duration range
        # We'll use the mode (most common value) and keep values within percentile_range% of it
        mode_duration = gloss_data['duration'].mode().iloc[0]
        
        # Calculate range based on percentiles instead of standard deviation
        lower_percentile = (100 - percentile_range) / 2
        upper_percentile = 100 - lower_percentile
        
        lower_bound = gloss_data['duration'].quantile(lower_percentile / 100)
        upper_bound = gloss_data['duration'].quantile(upper_percentile / 100)
        
        # Filter data within the common range
        common_data = gloss_data[
            (gloss_data['duration'] >= lower_bound) &
            (gloss_data['duration'] <= upper_bound)
        ]
        
        filtered_data.append(common_data)
        
        print(f"\nGloss: {gloss}")
        print(f"  Tổng số mẫu: {len(gloss_data)}")
        print(f"  Số mẫu sau khi lọc: {len(common_data)}")
        print(f"  Duration phổ biến nhất: {mode_duration:.2f}")
        print(f"  Khoảng duration giữ lại: [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    # Combine all filtered data
    filtered_df = pd.concat(filtered_data, ignore_index=True)
    
    # Save to CSV
    filtered_df.to_csv(output_csv, index=False)
    
    print(f"\n=== KẾT QUẢ LỌC ===")
    print(f"Tổng số mẫu sau khi lọc: {len(filtered_df)}")
    print(f"Tỷ lệ giữ lại: {(len(filtered_df) / len(df) * 100):.2f}%")
    print(f"\nĐã lưu kết quả vào file: {output_csv}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python filter_common_durations.py <input_csv> <output_csv> [percentile_range]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    percentile_range = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    
    filter_common_durations(input_csv, output_csv, percentile_range) 