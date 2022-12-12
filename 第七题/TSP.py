# -*- coding: utf-8 -*-
import random
import copy
import sys
import math
import tkinter
import threading
from functools import reduce

# 城市坐标
city_num = 8
distance_x = [
    360,482,666,597,209,201,492,294]
distance_y = [
    244,114,104,552,70,425,227,331]

# 城市距离
distance_graph = [ [0.0 for col in range(city_num)] for row in range(city_num)]
class TSP(object):

    def __init__(self, root, width = 800, height = 600, n = city_num):

        # 创建画布
        self.root = root                               
        self.width = width      
        self.height = height
        # 城市数目初始化为city_num
        self.n = n
        # Tkinter.Canvas
        self.canvas = tkinter.Canvas(
                root,
                width = self.width,
                height = self.height,
                bg = "#EBEBEB",             # 背景白色 
                xscrollincrement = 1,
                yscrollincrement = 1
            )
        self.canvas.pack(expand = tkinter.YES, fill = tkinter.BOTH)
        self.title("TSP神经网络(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5
        self.__lock = threading.RLock()     # 线程锁

        self.__bindEvents()
        self.new()

        # 计算城市之间的距离
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))

    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)        # 退出程序
        self.root.bind("n", self.new)          # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)         # 停止搜索

    # 更改标题
    def title(self, s):

        self.root.title(s)

    # 初始化
    def new(self, evt = None):

        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()     # 清除信息 
        self.nodes = []  # 节点坐标
        self.nodes2 = [] # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                    y - self.__r, x + self.__r, y + self.__r,
                    fill = "#ff0000",      # 填充红色
                    outline = "#000000",   # 轮廓白色
                    tags = "node",
                )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x,y-10,              # 使用create_text方法在坐标（302，77）处绘制文字  
                    text = '('+str(x)+','+str(y)+')',    # 所绘制文字的内容  
                    fill = 'black'                       # 所绘制文字的颜色为灰色
                )
            
        # 顺序连接城市
        self.path = range(city_num)
        self.line(self.path)

        self.A=1.5     # 变化率A, 和B取一样的值
        self.D=1.0     # 变化率D
        self.u0=0.02   # 初始u值
        self.step=0.01 # 步长
        self.iter = 1  # 迭代次数
        
        # 计算总长度
        self.DistanceCity = self.__cal_total_distance()
        # 初始化状态
        self.U=[ [0.5*self.u0*math.log(city_num-1) for col in range(city_num)] for row in range(city_num)]
        # 加上随机值[-1:1]
        for row in range(city_num):
            for col in range(city_num):
                self.U[row][col] += 2*random.random()-1
        # 阈值函数
        self.V=[ [0.0 for col in range(city_num)] for row in range(city_num)]
        for row in range(city_num):
            for col in range(city_num):
                self.V[row][col] = (1+math.tanh(self.U[row][col]/self.u0))/2
                
    # 计算路径总距离
    def __cal_total_distance(self):
        
        temp_distance = 0.0
        
        for i in range(1, city_num):
            start, end = self.path[i], self.path[i-1]
            temp_distance += distance_graph[start][end]

        # 回路
        end = self.path[0]
        temp_distance += distance_graph[start][end]
        return temp_distance
        
    # 将节点按order顺序连线
    def line(self, order):
        # 删除原线
        self.canvas.delete("line")
        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill = "#000000", tags = "line")
            return i2
        
        # order[-1]为初始值
        reduce(line2, order, order[-1])

    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print("\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    # 停止搜索
    def DeltaU(self):

        # 计算每一行的和
        rowsum = []
        for row in range(city_num):
            row_sum = 0
            for col in range(city_num):
                row_sum += self.V[row][col]
            rowsum.append(row_sum-1)
            
        # 计算每一列的和
        colsum = []
        for col in range(city_num):
            col_sum = 0
            for row in range(city_num):
                col_sum += self.V[row][col]
            colsum.append(col_sum-1)
            
        # 将第一列移向最后
        deltau = copy.deepcopy(self.V)
        for row in deltau:
            temp = row[0]
            del row[0]
            row.append(temp)
            
        # 计算deltau
        for row in range(city_num):
            for col in range(city_num):
                deltau[row][col] = -1*(self.A*rowsum[row]+self.A*colsum[col]+self.D*self.DistanceCity*deltau[row][col])
                
        return deltau

    # 计算能量
    def Energy(self):

        # 计算每一行的和的平方和
        rowsum = []
        for row in range(city_num):
            row_sum = 0
            for col in range(city_num):
                row_sum += self.V[row][col]
            rowsum.append(row_sum-1)
        rowsumsqr = 0
        for row in rowsum:
            rowsumsqr += row*row
        # 计算每一列的和的平方和
        colsum = []
        for col in range(city_num):
            col_sum = 0
            for row in range(city_num):
                col_sum += self.V[row][col]
            colsum.append(col_sum-1)
        colsumsqr = 0
        for col in colsum:
            colsumsqr += col*col

        # 将第一列移向最后
        PermitV = copy.deepcopy(self.V)
        for row in PermitV:
            temp = row[0]
            del row[0]
            row.append(temp)
            for item in row:
                item *= self.DistanceCity
        # 矩阵点乘和
        sumV = 0
        for row in range(city_num):
            for col in range(city_num):
                sumV += PermitV[row][col] * self.V[row][col]
        # 计算能量
        E = 0.5*(self.A*rowsumsqr+self.A*colsumsqr+self.D*sumV) # 采用简化形式的能量函数
        return E

    # 生成路径
    def Pathcheck(self):
        V1 = [ [0 for col in range(city_num)] for row in range(city_num)]
        # 寻找每一列的最大值
        for col in range(city_num):
            MAX = -1.0
            MAX_row = -1.0
            for row in range(city_num):
                if self.V[row][col] > MAX:
                    MAX = self.V[row][col]
                    MAX_row = row
            # 相应位置赋值为1
            V1[MAX_row][col] = 1
            
        # 计算每一行的和
        rowsum = []
        for row in range(city_num):
            row_sum = 0
            for col in range(city_num):
                row_sum += V1[row][col]
            rowsum.append(row_sum)
            
        # 计算每一列的和
        colsum = []
        for col in range(city_num):
            col_sum = 0
            for row in range(city_num):
                col_sum += V1[row][col]
            colsum.append(col_sum)
        # 计算差的平方和
        sumV1 = 0
        for item in range(city_num):
            sumV1 += (rowsum[item] - colsum[item])**2
        # 形成路径
        path = []
        if sumV1 != 0:
            path.append(-1)
        else:
            for col in range(city_num):
                for row in range(city_num):
                    if V1[row][col] == 1:
                        path.append(row)
        return path
        
    # 开始搜索
    def search_path(self, evt = None):

        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()
        
        while self.__running:
            
            # 计算deltau
            delta_u = self.DeltaU()
            # 更新 u 矩阵
            for row in range(city_num):
                for col in range(city_num):
                    self.U[row][col] += delta_u[row][col] * self.step
            # 计算阈值
            for row in range(city_num):
                for col in range(city_num):
                    self.V[row][col] = (1+math.tanh(self.U[row][col]/self.u0))/2

            # 计算能量
            E = self.Energy()
            # 生成路径
            path = self.Pathcheck()
            if path[0] != -1:
                self.path = path
                # 连线
                self.line(self.path)
                print("迭代次数：",self.iter,"最佳路径总距离：",int(self.__cal_total_distance()))
            else:
                print("迭代次数：",self.iter,"失败")
                
            # 设置标题
            self.title("TSP神经网络(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
            self.iter += 1

    # 主循环
    def mainloop(self):
        self.root.mainloop()
  
if __name__ == '__main__':
    TSP(tkinter.Tk()).mainloop()