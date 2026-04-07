# import csv

# # 打开CSV文件
# with open('critical_uti.csv', mode='r') as file:
#     reader = csv.DictReader(file)
    
#     # 用于存储符合条件的seed
#     seeds = []
    
#     # 遍历每一行
#     for row in reader:
#         # 将uti转换为浮点数
#         uti = float(row['Uti'])
        
#         # 检查uti是否小于等于3.3
#         if uti <= 3.2:
#             # 将seed添加到列表中
#             seeds.append(int(row['Seed']))
    
#     # 以[xxx, xxx, ...]的形式输出
#     print(seeds)
    
#     # 统计数量
#     count = len(seeds)
#     print(f"满足条件的seed数量: {count}")

import csv

# 打开CSV文件
with open('critical_uti.csv', mode='r') as file:
    reader = csv.DictReader(file)
    
    # 用于存储符合条件的seed
    seeds_leq_3_2 = []  # 存储uti <= 3.2的seed
    seeds_eq_3_3 = []   # 存储uti = 3.3的seed
    
    # 遍历每一行
    for row in reader:
        uti = float(row['Uti'])
        if uti <= 3.2:
            seeds_leq_3_2.append(int(row['Seed']))
        elif uti == 3.3:
            seeds_eq_3_3.append(int(row['Seed']))
    
    # 计算需要补充的uti = 3.3的seed数量
    total_needed = 100
    current_count = len(seeds_leq_3_2)
    additional_needed = total_needed - current_count
    
    # 如果uti <= 3.2的seed已经足够，直接输出
    if additional_needed <= 0:
        final_seeds = seeds_leq_3_2[:total_needed]
    else:
        # 补充uti = 3.3的seed
        final_seeds = seeds_leq_3_2 + seeds_eq_3_3[:additional_needed]
    
    # 输出最终的seed列表
    print(final_seeds)
    
    # 统计总数
    print(f"最终的seed数量: {len(final_seeds)}")