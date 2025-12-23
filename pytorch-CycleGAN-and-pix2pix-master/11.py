import os
import shutil

# ① 源目录：所有病人/编号文件夹所在的目录
SOURCE_ROOT = "/data/HSI_RCC/新he"

# ② 目标目录：所有 bmp 要集中放到这里（不存在会自动创建）
DEST_ROOT = "/data/HSI_RCC/新he_bmp_all"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def copy_all_bmp(source_root, dest_root):
    ensure_dir(dest_root)
    count = 0

    for dirpath, dirnames, filenames in os.walk(source_root):
        for filename in filenames:
            if filename.lower().endswith(".bmp"):
                src_path = os.path.join(dirpath, filename)

                # 处理重名文件：在目标目录中加 _1, _2, ...
                base, ext = os.path.splitext(filename)
                dest_path = os.path.join(dest_root, filename)
                idx = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(dest_root, f"{base}_{idx}{ext}")
                    idx += 1

                shutil.copy2(src_path, dest_path)
                count += 1
                print(f"复制：{src_path}  ->  {dest_path}")

    print(f"\n完成！共复制 {count} 个 .bmp 文件到：{dest_root}")

if __name__ == "__main__":
    copy_all_bmp(SOURCE_ROOT, DEST_ROOT)