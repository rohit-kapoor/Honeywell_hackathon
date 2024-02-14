import pandas as pd
# l1=[]
# with open("/raid/sankalpm/assignment/position-data-2024-02-01.csv","r") as file1:
#     l1=file1.readlines()
#     l1=l1[2:]
# with open("/raid/sankalpm/assignment/position-data-2024-02-01_cleaned.csv","w") as file1:
#     file1.writelines(l1)

# df1 = pd.read_csv("/raid/sankalpm/assignment/position-data-2024-02-01_cleaned.csv")
# print(df1.columns)

# l1=[]


# l2=[]
# with open("/raid/sankalpm/assignment/position-data-2024-02-02.csv","r") as file2:
#     l2=file2.readlines()
#     l2=l2[2:]
# with open("/raid/sankalpm/assignment/position-data-2024-02-02_cleaned.csv","w") as file2:
#     file2.writelines(l2)

# df2 = pd.read_csv("/raid/sankalpm/assignment/position-data-2024-02-02_cleaned.csv")
# print(df2.columns)
# l2=[]

# l3=[]
# with open("/raid/sankalpm/assignment/position-data-2024-02-03.csv","r") as file3:
#     l3=file3.readlines()
#     l3=l3[2:]
# with open("/raid/sankalpm/assignment/position-data-2024-02-03_cleaned.csv","w") as file3:
#     file3.writelines(l3)

# df3 = pd.read_csv("/raid/sankalpm/assignment/position-data-2024-02-03_cleaned.csv")
# # print(df3.columns)
# # l3=[]

# df_mega=pd.concat([df1,df2,df3],axis=0)
# pd.to_csv(df_mega,"")
# df_mega.to_csv("/raid/sankalpm/assignment/position-data-mega.csv")
# print(df_mega.columns)
# import pickle
# import pandas as pd
# import numpy as np
# df = pd.read_csv("/raid/sankalpm/assignment/position-data-mega.csv")
# dic = {}
#     l = list(set(df["id"]))
#     for id in l:
#         df_x = df[df["id"] == id]
#         for n in ['la','lo','alt','hd','gs']:
#             m1 = df_x[n].mean()
#             df_x[n].fillna(m1, inplace = True)
#         df_x_sorted = df_x.sort_values(by='t')
#         df_x_sorted.fillna(0,inplace = True)
#         c1 = list(df_x_sorted["t"])
#         c2 = list(df_x_sorted["la"])
#         c3 = list(df_x_sorted["lo"])
#         c4 = list(df_x_sorted["alt"])
#         c5 = list(df_x_sorted["hd"])
#         c6 = list(df_x_sorted["gs"])
#         len1 =  len(c1)
#         dic_temp = {}
#         for i in range(len1):
#             l1 = [c2[i], c3[i], c4[i], c5[i], c6[i]]
#             if c1[i] not in dic_temp:
#                 dic_temp[c1[i]] = []
#             dic_temp[c1[i]].append(l1)
#         dic[id] = dic_temp

# with open('/raid/sankalpm/assignment/dict2.pkl', 'wb') as f:
#     pickle.dump(dic, f)

# for key in dic2:
#     print(key, dic[key])
        

import pickle
import numpy as np
import datetime
l = []
lbs = []
with open('/raid/sankalpm/assignment/dict.pkl', 'rb') as handle:
    dic1 = pickle.load(handle)

    for key1 in dic1:
    #     lt = []
        lbs.append(key1)
    #     for key2 in dic1[key1]:
    #         timestamp_integer = 0
    #         if(str(key2) != '0'):
    #             timestamp = datetime.datetime.strptime(str(key2).split('.')[0], '%Y-%m-%d %H:%M:%S')

    #             # Convert the datetime object to a Unix timestamp (integer)
    #             timestamp_integer = int(timestamp.timestamp() * 1)
    #             # print(timestamp_integer)
    #             # print(dic1[key1][key2][0])
    #         lt.append(np.array([timestamp_integer]+dic1[key1][key2][0]))
    #         # print(len(lt))
    #     l.append(np.array(lt))
    # # print(l.shape)
    # with open('/raid/sankalpm/assignment/X2.pkl', 'wb') as f:
    #     pickle.dump(l, f)
    with open('/raid/sankalpm/assignment/L1.pkl', 'wb') as f:
        pickle.dump(lbs, f)




    
    