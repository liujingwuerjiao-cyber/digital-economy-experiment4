
import docx
import sys

try:
    doc = docx.Document("实验大纲：二级密封拍卖仿真实验设计.docx")
    for para in doc.paragraphs:
        print(para.text)
except Exception as e:
    print(f"Error reading docx: {e}")
