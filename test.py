#import numpy as np
#import pandas as pd
#import time
#
#N_STATES = 6   # 1维世界的宽度
#ACTIONS = ['left', 'right']     # 探索者的可用动作
#EPSILON = 0.9   # 贪婪度 greedy
#ALPHA = 0.1     # 学习率
#GAMMA = 0.9    # 奖励递减值
#MAX_EPISODES = 13   # 最大回合数
#FRESH_TIME = 0.3    # 移动间隔时间
#
#def build_q_table(n_states, actions):
#    table = pd.DataFrame(
#        np.zeros((n_states, len(actions))),     # q_table 全 0 初始
#        columns=actions,    # columns 对应的是行为名称
#    )
#    return table
#
## 在某个 state 地点, 选择行为
#def choose_action(state, q_table):
#    state_actions = q_table.iloc[state, :]  # 选出这个 state 的所有 action 值
#    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0):  # 非贪婪 or 或者这个 state 还没有探索过
#        action_name = np.random.choice(ACTIONS)
#    else:
#        action_name = state_actions.idxmax()    # 贪婪模式
#    return action_name
#
#
#def get_env_feedback(S, A):
#    # This is how agent will interact with the environment
#    if A == 'right':    # move right
#        if S == N_STATES - 2:   # terminate
#            S_ = 'terminal'
#            R = 1
#        else:
#            S_ = S + 1
#            R = 0
#    else:   # move left
#        R = 0
#        if S == 0:
#            S_ = S  # reach the wall
#        else:
#            S_ = S - 1
#    return S_, R
#
#def update_env(S, episode, step_counter):
#    # This is how environment be updated
#    env_list = ['-']*(N_STATES-1) + ['T']   # '---------T' our environment
#    if S == 'terminal':
#        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
#        print('\r{}'.format(interaction), end='')
#        time.sleep(2)
#        print('\r                                ', end='')
#    else:
#        env_list[S] = 'o'
#        interaction = ''.join(env_list)
#        print('\r{}'.format(interaction), end='')
#        time.sleep(FRESH_TIME)
#        
#def rl():
#    q_table = build_q_table(N_STATES, ACTIONS)  # 初始 q table
#    for episode in range(MAX_EPISODES):     # 回合
#        step_counter = 0
#        S = 0   # 回合初始位置
#        is_terminated = False   # 是否回合结束
#        update_env(S, episode, step_counter)    # 环境更新
#        while not is_terminated:
#
#            A = choose_action(S, q_table)   # 选行为
#            S_, R = get_env_feedback(S, A)  # 实施行为并得到环境的反馈
#            q_predict = q_table.loc[S, A]    # 估算的(状态-行为)值
#            if S_ != 'terminal':
#                q_target = R + GAMMA * q_table.iloc[S_, :].max()   #  实际的(状态-行为)值 (回合没结束)
#            else:
#                q_target = R     #  实际的(状态-行为)值 (回合结束)
#                is_terminated = True    # terminate this episode
#
#            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  #  q_table 更新
#            S = S_  # 探索者移动到下一个 state
#
#            update_env(S, episode, step_counter+1)  # 环境更新
#
#            step_counter += 1
#    return q_table
#
#if __name__ == "__main__":
#    q_table = rl()
#    print('\r\nQ-table:\n')
#    print(q_table)

#from matplotlib import pyplot as plt
#
#img = plt.imread("D:/照片(本地)/20180522-卓雅小学六年级5班毕业照片/2018毕业照/学生+班主任单照\罗怡然.JPG")
#h, w, _ = img.shape
#xs, ys = [], []
#for i in range(100):
#    mean = w*np.random.rand(), h*np.random.rand()
#    a = 50 + np.random.randint(50, 200)
#    b = 50 + np.random.randint(50, 200)
#    c = (a + b)*np.random.normal()*0.2
#    cov = [[a, c], [c, b]]
#    count = 200
#    x, y = np.random.multivariate_normal(mean, cov, size=count).T
#    xs.append(x)
#    ys.append(y)
#x = np.concatenate(xs)
#y = np.concatenate(ys)
#hist, _, _ = np.histogram2d(x, y, bins=(np.arange(0, w), np.arange(0, h)))
#hist = hist.T
#plt.imshow(hist)
#
#from scipy.ndimage import filters
#heat = filters.gaussian_filter(hist, 10.0)
#plt.imshow(heat);
#


#import os
#import struct
#import numpy as np
#import cv2
#KSIZE = 3
#SIGMA = 3
#image = cv2.imread("0.JPG")
#print("image shape:",image.shape)
#dst = cv2.GaussianBlur(image, (KSIZE,KSIZE), SIGMA, KSIZE)
#cv2.imshow("img1",image)
#cv2.imshow("img2",dst)
#cv2.waitKey()
#cv2.destroyAllWindows()

#import visdom
import torch as t

#vis = visdom.Visdom(env=u'test1')
#
#vis.image(t.randn(64,64).numpy())
#
#vis.image(t.randn(3,64,64).numpy(), win='random2')
#
#vis.images(t.randn(36,3,64,64).numpy(), nrow=6, win='random3', opts={'title':'random_imgs'})
#
#tensor= t.Tensor(3,4)
#tensor.cuda(0)
#tensor.is_cuda
#
#critierion = t.nn.CrossEntropyLoss(weight=t.Tensor([1,3])).cuda()
#
#input = t.autograd.Variable(t.randn(4,2)).cuda()
#target = t.autograd.Variable(t.Tensor([1,0,0,1])).long().cuda()
#
#loss = critierion(input, target)
#critierion._buffers
#
#x = t.cuda.FloatTensor(2,3)
#print(x.get_device())
#
#y = t.FloatTensor(2,3).cuda()
#print(y.get_device())
#
#print(x.is_cuda, y.is_cuda)
#print(x.get_device()==y.get_device()==0)
#
#t.set_default_tensor_type('torch.cuda.FloatTensor')
#t.set_default_dtype('')
#
#a = t.ones(2,3)

#try:
#    import ipdb
#except:
#    import ipdb as ipdb
#    
#def sum(x):
#    r = 0
#    for i in x:
#        r += i
#    return r
#
#def mul(x):
#    r = 1
#    for i in x:
#        r *= i
#    return r
#
#ipdb.set_trace()
#
#x = [1,2,3,4,5]
#r = sum(x)
#r = mul(x)

'''
import numpy as np  #导入numpy模块
import operator as op  #导入operator模块
import operator 
from matplotlib import pyplot as plt

#生成训练数据函数
def createDataSet():
    dataSet = np.array([[2.1,1.2],[1.3,2.5],[1.4,2.3],[2.2,1.3],[2.3,1.5]])  #数据集的规模是四行两列的数组
    labels = ['A','B','B','A','A']  #数据集每一条记录所对应的标签
    return dataSet, labels  #返回数据集数据和它对应的标签


#K-NN预测分类函数

#inX:一条未知标签的数据记录
#dataSet:训练数据集
#labels:训练数据集每一条记录所对应的标签
#k:事先人为定义的数值
def classify0(inX, dataSet, labels, k):
    #获取数据集的行数或者记录数，.shape[0]:获取矩阵的行数，.shape[1]：获取数据集的列数
    dataSetSize = dataSet.shape[0]  
    #将修改规模后的未知标签的数据记录与训练数据集作差
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet  
    #再将对应得到的差值求平方
    sqDiffMat = diffMat**2  
    #横向求得上面差值平方的和，axis=1:表示矩阵的每一行分别进行相加
    sqDistances = sqDiffMat.sum(axis=1)
    #再对每条记录的平方和开方，得到每一条已知标签的记录与未知标签数据的距离
    distances = sqDistances**0.5
    #对求得的距离进行排序，返回的是排序之后的数值对应它原来所在位置的下标
    sortedDistIndicies = distances.argsort()  
    #创建一个空的字典，用来保存和统计离未知标签数据最近的已知标签与该标签的个数，标签作为字典的key（键），该标签的个数作为字典的value（值）
    classCount={}          
    for i in range(k):
        #sortedDistIndicies[i]排序之后得到的k个离未知标签数据最近的训练数据记录下标，比如是第二条记录，那它就等于1（下标从零开始）
        #voteIlabel就是：训练数据第sortedDistIndicies[i]条记录的标签
        voteIlabel = labels[sortedDistIndicies[i]]
        #把上面得到的那个标签作为字典的键，并且累计k个标签中，该标签有多少个，这个累计个数作为该标签的键值，也就是value
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #对上面循环得到的标签统计字典进行按值（标签的累计数）排序，并且是从大到小排序
    #下面的写法是固定的
    #classCount.items()：得到字典的键和值两个列表
    #key=operator.itemgetter(1)：指明该排序是按字典的值从小到大排序，与键无关
    #reverse=True：从大到小排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #统计数最大的那个标签，作为对未知标签的预测，也就是该返回标签就是未知标签数据inX的预测标签了
    return sortedClassCount[0][0]

#填写一个预测属性值，输出属性对应的标签
    
dataSet, labels = createDataSet()
aa = classify0([1.2, 2.3], dataSet, labels, 3)
print(aa)
'''


from math import log
import operator

def calcShannonEnt(dataSet):  # 计算数据的熵(entropy)
    numEntries=len(dataSet)  # 数据条数
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1] # 每行数据的最后一个字（类别）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1  # 统计有多少个类以及每个类的数量
    shannonEnt=0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries # 计算单个类的熵值
        shannonEnt-=prob*log(prob,2) # 累加每个类的熵值
    return shannonEnt

def createDataSet1():    # 创造示例数据
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    labels = ['头发','声音']  #两个特征
    return dataSet,labels

def splitDataSet(dataSet,axis,value): # 按某个特征分类后的数据
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            reducedFeatVec =featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):  # 选择最优的分类特征
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)  # 原始的熵
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob =len(subDataSet)/float(len(dataSet))
            newEntropy +=prob*calcShannonEnt(subDataSet)  # 按特征分类后的熵
        infoGain = baseEntropy - newEntropy  # 原始熵与按特征分类后的熵的差值
        if (infoGain>bestInfoGain):   # 若按某特征划分后，熵值减少的最大，则次特征为最优分类特征
            bestInfoGain=infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):    #按分类后类别数量排序，比如：最后分类为2男1女，则判定为男；
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]  # 类别：男或女
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel=labels[bestFeat]
    myTree={bestFeatLabel:{}} #分类结果以字典形式保存
    del(labels[bestFeat])
    featValues=[example[bestFeat] for example in dataSet]
    uniqueVals=set(featValues)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet\
                            (dataSet,bestFeat,value),subLabels)
    return myTree


if __name__=='__main__':
    dataSet, labels=createDataSet1()  # 创造示列数据
    myTree = createTree(dataSet, labels)  # 输出决策树模型结果
    print(myTree)

print('END')