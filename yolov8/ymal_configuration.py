from pathlib import Path as path
content=f"""train:train/images
val:valid/images
test:test/images
names=['stone']
"""
with path(r"D:\graval detection project\resyris\resyris\test_data_realworld\dataset\data.yaml").open('w') as f:
    f.write(content)