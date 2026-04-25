import csv

def expand_csv_to_7_columns(input_path, output_path):
    with open(input_path, "r", encoding="utf-8-sig") as fin, \
         open(output_path, "w", encoding="utf-8-sig", newline="") as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # 读取表头
        header = next(reader)
        print(f"原始列数: {len(header)}")
        print(f"原始列名: {header}")
        
        # 目标表头（7列）
        target_header = ["id", "time", "dummy1", "text", "dummy2", "dummy3", "label"]
        writer.writerow(target_header)
        
        # 处理每一行
        for i, row in enumerate(reader, start=2):
            # 原始列：微博id,微博发布时间,发布人账号,微博中文内容,微博图片,情感倾向
            # 映射到：id, time, dummy1, text, dummy2, dummy3, label
            new_row = [
                row[0],          # id (微博id)
                row[1],          # time (微博发布时间)
                "",              # dummy1
                row[3],          # text (微博中文内容)
                "",              # dummy2
                "",              # dummy3
                row[5] if len(row) > 5 else ""  # label (情感倾向)
            ]
            writer.writerow(new_row)
    
    print(f"已保存到: {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python fix_csv_columns.py <输入csv> <输出csv>")
        sys.exit(1)
    expand_csv_to_7_columns(sys.argv[1], sys.argv[2])
