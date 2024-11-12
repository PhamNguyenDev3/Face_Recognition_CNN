import pandas as pd

# Đọc hai tệp CSV vào dataframe
# df1 = pd.read_csv('Datasets.csv')
df2 = pd.read_csv('Nguyen.csv')
df3 = pd.read_csv('ThuDiem.csv')
df4 = pd.read_csv('ThuThao.csv')
df5 = pd.read_csv('ThuHien.csv')
df6 = pd.read_csv('HuynhDuc.csv')
# Nối hai dataframe lại với nhau
merged_df = pd.concat([df2,df3,df4,df5,df6], ignore_index=False)

# Lưu dataframe kết quả vào tệp CSV mới
merged_df.to_csv('Datasets.csv', index=False)
