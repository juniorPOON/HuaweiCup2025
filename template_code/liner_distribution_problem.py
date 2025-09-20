'''

机场线性-调度-问题
资源分配

'''

import pulp

model = pulp.LpProblem("airport_problem",pulp.LpMinimize)

tasks = {"A": 25, "B": 35, "C": 15,"D":20 , "E" : 30 }
vehicles = {"V1": 60, "V2": 50,"V3":40}


x = pulp.LpVariable.dicts("assgin",[(i,j) for i in vehicles for j in tasks],
                          lowBound=0,upBound=1,cat='Integer')

model += pulp.lpSum(tasks[j] * x[(i,j)] for i in vehicles for j in tasks)

for j in tasks:
    model += pulp.lpSum(x[(i,j)] for i in vehicles ) == 1

for i in vehicles:
    model += pulp.lpSum(x[(i,j)] * tasks[j] for j in tasks ) <= vehicles[i]

model.solve()

# 打印具体的分配方案：遍历所有车辆和任务，当x[i,j]=1时表示该车辆负责对应任务
for i in vehicles:
    for j in tasks:
        if pulp.value(x[(i,j)]) == 1:  # pulp.value()用于获取变量的最优取值
            print(f"车辆 {i} 负责 飞机 {j}")

# 打印最小总工作时长（目标函数的最优值）
print("总工作时长 =", pulp.value(model.objective))