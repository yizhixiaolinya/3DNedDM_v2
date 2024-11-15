import pandas as pd

# 指定文件路径
file_path = 'C:/Users/Admin/Desktop/ukb669433.csv'
output_csv_path = 'C:/Users/Admin/Desktop/first_five_rows.csv'
output_txt_path = 'C:/Users/Admin/Desktop/first_five_rows.txt'

# 仅加载前五行数据
try:
    df = pd.read_csv(file_path, nrows=5)

    # 将前五行保存到CSV文件中
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    # 将前五行保存到TXT文件中，每行一个值，列与列之间用制表符分隔
    with open(output_txt_path, 'w', encoding='utf-8-sig') as txt_file:
        for row in df.values:
            txt_file.write("\t".join(map(str, row)) + "\n")

    print(f"前五行数据已保存到 {output_csv_path} 和 {output_txt_path}")
except Exception as e:
    print(f"读取CSV文件时出错: {e}")
