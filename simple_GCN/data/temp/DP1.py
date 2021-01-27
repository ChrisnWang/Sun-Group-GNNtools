#%%

import pandas as pd
import numpy as np

#%%

#读
attributes = pd.read_csv("checkin.csv",header = None)
attributes = attributes.head(90000)

#%%

#提取
attributes_column=list(attributes.columns)
user_Point=attributes[[attributes_column[0],attributes_column[4]]]
user_Point

#%%

#去重
s = set() #创建空集合
user_Point = np.array(user_Point)
user_Point2=list(set([tuple(t) for t in user_Point]))
user_Point2=pd.DataFrame(user_Point2)
print(user_Point2)
user_Point2.to_csv("DP_checkin.csv",header=["user","Point"])

#%%

#提取所有地点
Point = user_Point2[1]
#地点去重
Point_only = list(set(Point))
print(Point_only)
#人员去重
User = user_Point2[0]
User_only = list(set(User))
user_num = len(User_only)
print(User_only)
print(user_num)


#%%
minn =200000
ind =-1
for i,j in enumerate(Point_only):
    right = 0
    false = 0
    for k in Point:
        if j == k:
            right = right + 1#有多少人去的
    judges = user_num - right - right #没人去的数量-有人去的数量
    print ("judges",judges)
    if abs(judges) < minn:
        minn = judges
        ind = i
    if judges==0:
        print("111111111111")
        print(Point_only[i])
    print("minn",minn)

#%%
print(ind)
print(Point_only[ind])
reu = Point_only[ind]

#ee8b1d0ea22411ddb074dbd65f1665cf
#相差12个人
#总共152个人

jud = []
inn = []
user_Point3 = np.array(user_Point2)
user_Point2_len = len(user_Point2)
for i in range(user_Point2_len):
    if user_Point3[i][1] == reu:
        jud.append(user_Point3[i][0])#是谁
for i in range(user_num):
    if User_only[i] in jud:
        inn.append(1)
    else:
        inn.append(0)
user = pd.DataFrame(User_only)
inn = pd.DataFrame(inn)
result=pd.concat([user,inn], axis=1)
result.to_csv("DP_attributes.csv")
