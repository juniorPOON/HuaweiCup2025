# -*- coding: utf-8 -*-
"""
整数规划 (IP) 调度优化模板
-----------------------------------
适用场景：
- 电力/能源调度：机组启停 + 储能充放电 + 电网购电
- 生产排产：车间设备开关 + 任务分配
- 投资决策：是否建设/选择某种设备
核心思想：
目标函数 (总成本最小) + 一系列约束 (供需平衡/机组出力/电池能量守恒)
"""

import pulp

# ===================== 参数区（可改） =====================
T = 24  # 时段数（比如一天 24 小时）
hours = range(T)

# 负荷需求曲线（举例：白天高，晚上低）
demand = [220,210,200,210,250,300,380,460,520,580,600,580,
          560,540,520,500,480,450,420,400,380,350,300,250]

# 电网电价（峰时贵，谷时便宜）
grid_price = [0.5]*8 + [0.8]*8 + [1.2]*6 + [0.5]*2 # 元/kWh

# 机组参数
Pmin, Pmax = 50.0, 200.0      # 出力上下限 (MW)
var_cost   = 0.5              # 边际成本 (元/kWh)
start_cost = 1000.0           # 启动一次的固定成本
ramp_up, ramp_down = 80.0, 80.0  # 爬坡限制
u_init, P_init = 0, 0.0       # 初始状态：关机，出力=0
BigM = Pmax                   # 足够大的数，用于逻辑约束

# 储能参数
SOC0    = 100.0   # 初始电量
E_max   = 200.0   # 电池容量上限
E_min   = 20.0    # 电池最小SOC
Pch_max = 100.0   # 最大充电功率
Pdis_max= 100.0   # 最大放电功率
eta_ch, eta_dis = 0.95, 0.95  # 充放电效率
cycle_back = True  # 是否要求SOC期末回到初始值

# ===================== 建模 =====================
m = pulp.LpProblem("UnitCommitment", pulp.LpMinimize)

# 决策变量
Pgen   = pulp.LpVariable.dicts("Pgen", hours, lowBound=0)               # 机组出力
u      = pulp.LpVariable.dicts("UnitOn", hours, cat="Binary")           # 机组开机状态 (0/1)
v_on   = pulp.LpVariable.dicts("StartUp", hours, cat="Binary")          # 启动指示 (0/1)

Pch    = pulp.LpVariable.dicts("Charge", hours, lowBound=0)             # 充电功率
Pdis   = pulp.LpVariable.dicts("Discharge", hours, lowBound=0)          # 放电功率
ych    = pulp.LpVariable.dicts("Ych", hours, cat="Binary")              # 充电模式 (0/1)
ydis   = pulp.LpVariable.dicts("Ydis", hours, cat="Binary")             # 放电模式 (0/1)
SOC    = pulp.LpVariable.dicts("SOC", hours, lowBound=E_min, upBound=E_max)  # 电池电量

Pgrid  = pulp.LpVariable.dicts("Grid", hours, lowBound=0)               # 电网购电功率

# ===================== 目标函数 =====================
# 机组燃料成本 + 启动成本 + 电网购电成本
m += pulp.lpSum([
    var_cost * Pgen[t] + start_cost * v_on[t] + grid_price[t] * Pgrid[t]
    for t in hours
])

# ===================== 约束条件 =====================

# 1. 供需平衡：发电 + 放电 + 购电 = 负荷 + 充电
for t in hours:
    m += Pgen[t] + Pdis[t] + Pgrid[t] == demand[t] + Pch[t]

# 2. 出力限制：受开机状态控制
for t in hours:
    m += Pgen[t] >= Pmin * u[t]
    m += Pgen[t] <= Pmax * u[t]

# 3. 启动判定：若从关机->开机，则 v_on=1
m += v_on[0] >= u[0] - u_init
for t in range(1, T):
    m += v_on[t] >= u[t] - u[t-1]

# 4. 爬坡约束：相邻时段出力变化不能太快
for t in range(1, T):
    m += Pgen[t] - Pgen[t-1] <= ramp_up + BigM*(2 - u[t] - u[t-1])
    m += Pgen[t-1] - Pgen[t] <= ramp_down + BigM*(2 - u[t] - u[t-1])

# 5. 电池充放电互斥 + 功率上限
for t in hours:
    m += ych[t] + ydis[t] <= 1
    m += Pch[t]  <= Pch_max  * ych[t]
    m += Pdis[t] <= Pdis_max * ydis[t]

# 6. 电池能量守恒 (SOC 动态)
m += SOC[0] == SOC0 + eta_ch*Pch[0] - (1/eta_dis)*Pdis[0]
for t in range(1, T):
    m += SOC[t] == SOC[t-1] + eta_ch*Pch[t] - (1/eta_dis)*Pdis[t]

# 7. SOC 末端回到初始 (防止“榨干电池”作弊)
if cycle_back:
    m += SOC[T-1] == SOC0

# ===================== 求解 =====================
m.solve(pulp.PULP_CBC_CMD(msg=False))

# ===================== 输出 =====================
print("状态:", pulp.LpStatus[m.status])
print("最优总成本 = {:.2f}".format(pulp.value(m.objective)))

for t in hours:
    print(f"t={t:02d} | 负荷={demand[t]} | 开机={int(u[t].value())} | "
          f"Pgen={Pgen[t].value():.1f} | 充电={Pch[t].value():.1f} | "
          f"放电={Pdis[t].value():.1f} | 购电={Pgrid[t].value():.1f} | "
          f"SOC={SOC[t].value():.1f}")
