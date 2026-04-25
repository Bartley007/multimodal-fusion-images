import csv
import zipfile
from pathlib import Path

# Configuration
SOURCE_CSV = Path("nCoV_100k_train.labled.csv")
OUTPUT_DIR = Path("dataset")
OUTPUT_CSV = OUTPUT_DIR / "dataset.csv"
OUTPUT_ZIP = OUTPUT_DIR / "images.zip"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)

print("Processing CSV...")

# Process CSV
processed_data = []
encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']

for encoding in encodings:
    try:
        with open(SOURCE_CSV, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            headers = next(reader)  # Skip header
            print(f"Using encoding: {encoding}")
            
            for idx, row in enumerate(reader):
                if len(row) >= 7:
                    row_id = str(row[0]) if row[0] else f"row_{idx}"
                    time_str = row[1] if len(row) > 1 else "2020-01-01 00:00"
                    publisher = row[2] if len(row) > 2 else ""
                    text_content = row[3] if len(row) > 3 else ""
                    image_urls = row[4] if len(row) > 4 else ""
                    video_urls = row[5] if len(row) > 5 else ""
                    label = row[6] if len(row) > 6 else "0"
                    
                    # Format time (convert Chinese date format)
                    if '月' in time_str and '日' in time_str:
                        time_str = time_str.replace('月', '-').replace('日', ' ')
                        time_str = f"2020-{time_str}"
                    
                    formatted_row = [
                        row_id, time_str, publisher, text_content, "", video_urls, label
                    ]
                    processed_data.append(formatted_row)
                    
                    if len(processed_data) % 5000 == 0:
                        print(f"Processed {len(processed_data)} rows...")
                
                if len(processed_data) >= 20000:  # Process 20k rows
                    print("Reached 20,000 rows")
                    break
        
        break
        
    except UnicodeDecodeError:
        continue
    except Exception as e:
        print(f"Error: {e}")
        continue

# Save CSV
headers = ['id', 'time', 'publisher', 'text', 'image_path', 'video_urls', 'label']
with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(processed_data)

# Create empty ZIP
with zipfile.ZipFile(OUTPUT_ZIP, 'w') as zf:
    pass

print(f"\n✅ Completed!")
print(f"📁 Dataset directory: {OUTPUT_DIR.absolute()}")
print(f"📊 CSV: {OUTPUT_CSV}")
print(f"🖼️  Images ZIP: {OUTPUT_ZIP}")
print(f"📝 Processed {len(processed_data)} rows")

# Create readme
with open(OUTPUT_DIR / "readme.txt", 'w', encoding='utf-8') as f:
    f.write(f"""
Dataset Processing Summary
========================
Processed rows: {len(processed_data)}
Files created:
- dataset.csv: Formatted CSV with 7 columns
- images.zip: Empty ZIP file (images would be downloaded separately)

Format: id, time, publisher, text, image_path, video_urls, label
""")

print("\n🎯 Ready for upload to Multimodal Fusion Prediction Platform!")
