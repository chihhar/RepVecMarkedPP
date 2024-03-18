# Copyright (c) Facebook, Inc. and its affiliates.

from functools import partial
import contextlib
import numpy as np
import pdb

def generate_hawkes_modes05():
    np.random.seed(seed=32)
    [T,LL,L_TRG1,class_his] = simulate_hawkes_modes05(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score,class_his]
def generate_hawkes_pin2():
    np.random.seed(seed=32)
    [T,LL,L_TRG1,class_his] = simulate_hawkes_modes05(50000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[40000:].mean()
    return [T,score,class_his]
def generate_hawkes_pin3(class_num):
    np.random.seed(seed=32)
    [T,LL,L_TRG1,class_his] = simulate_hawkes_pin3(30000,0.2,[0.8,0.0],[1.0,20.0],class_suu=class_num)
    score = - LL[24000:].mean()
    return [T,score,class_his]
def generate_hawkes_pin4(class_num):
    np.random.seed(seed=32)
    [T,LL,L_TRG1,class_his] = simulate_hawkes_pin3(100000,0.2,[0.8,0.0],[1.0,20.0],class_suu=class_num)
    score = - LL[80000:].mean()
    return [T,score,class_his]
def simulate_hawkes_pin3(n,mu,alpha,beta,short_thre=1,long_thre=5,class_suu=10):
    T = []
    LL = []
    L_TRG1 = []
    class_his=[]

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    class_num=0
    preaccept_num=0
    while 1:
        l = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/l #scakeは逆数。大きいと母数が小さいため、大きい値を取りやすい。
        else: # short mode
            step = np.random.exponential(scale=0.25)/l

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            
            if preaccept_num==1 and is_long_mode==0:
                class_num=(class_num+1)%class_suu # class change and reload class_num
            class_his.append(class_num)#accept前のclassで判断。例えば、10秒待ったものxを採用したら、mode1のまんまなので、classは変わっていない。次に変わる。
            preaccept_num=is_long_mode #modeが反映されるのは、いベントが発生した瞬間。長いモードから短いに切り替わると違う場所。0連続したい->0連続したい。 同じ、1長い->1長い　長くならなかったから同じ、0->1 短い長い次は変えたい。1->0間隔が空いたから連続。これは切り替わる。
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            L_TRG1.append(l_trg1)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
        
        if count == n:
            break
    if l_trg1 > long_thre:
            is_long_mode = 1

    if l_trg1 < short_thre:
        is_long_mode = 0
    #pdb.set_trace()
    if preaccept_num==1 and is_long_mode==0:
        class_num=(class_num+1)%class_suu # class change and reload class_num
    
    class_his.append(class_num)#accept前のclassで判断。例えば、10秒待ったものxを採用したら、mode1のまんまなので、classは変わっていない。次に変わる。
    class_his=class_his[1:]
    return [np.array(T),np.array(LL),np.array(L_TRG1),class_his]


def simulate_hawkes_modes05(n,mu,alpha,beta,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    class_his=[]

    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    class_num=0
    preaccept_num=0
    while 1:
        l = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/l #scakeは逆数。大きいと母数が小さいため、大きい値を取りやすい。
        else: # short mode
            step = np.random.exponential(scale=0.25)/l

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            
            if preaccept_num==1 and is_long_mode==0:
                class_num=(class_num+1)%10 # class change and reload class_num
            class_his.append(class_num)#accept前のclassで判断。例えば、10秒待ったものxを採用したら、mode1のまんまなので、classは変わっていない。次に変わる。
            preaccept_num=is_long_mode #modeが反映されるのは、いベントが発生した瞬間。長いモードから短いに切り替わると違う場所。0連続したい->0連続したい。 同じ、1長い->1長い　長くならなかったから同じ、0->1 短い長い次は変えたい。1->0間隔が空いたから連続。これは切り替わる。
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            L_TRG1.append(l_trg1)
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
        
        if count == n:
            break
    if l_trg1 > long_thre:
            is_long_mode = 1

    if l_trg1 < short_thre:
        is_long_mode = 0
    if preaccept_num==1 and is_long_mode==0:
        class_num=(class_num+1)%10 # class change and reload class_num
    class_his.append(class_num)#accept前のclassで判断。例えば、10秒待ったものxを採用したら、mode1のまんまなので、classは変わっていない。次に変わる。
    class_his=class_his[1:]
    return [np.array(T),np.array(LL),np.array(L_TRG1),class_his]


def pinwheel(num_samples, num_classes):
    radial_std = 0.3# radius
    tangential_std = 0.1#
    num_per_class = num_samples
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.einsum("ti,tij->tj", features, rotations)

def pin_wheel_generate():
    ndim=2
    num_classes=10
    event_time, score, class_his = generate_hawkes_modes05()
    class_his=np.array(class_his)
    
    n = 100000#len(event_times)#周期数->イベント数なのでは？
    data_fn = partial(pinwheel, num_classes=num_classes)
    data = data_fn(n)#pinwheelを周期数*クラス数num_classesで生成
    seq = np.zeros((n, ndim + 1))# [時刻,x,y]
    seq[:, 0] = event_time
    
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):#dataを前から等分で wakeru iがクラスind、data_iが位置
        seq[:, 1:] = seq[:, 1:] + data_i * (i == class_his)[:, None]#適当に作ったものをインデックス番号で取ってくる。dataが適当にインデックス版に作られるから...
    return seq

def pin_wheel_generate2():
    ndim=2
    num_classes=10
    event_time, score, class_his = generate_hawkes_pin2()
    class_his=np.array(class_his)
    # import pickle
    # with open(f"./pickled/data/pin2_class",mode="wb") as file:
    #     pickle.dump(class_his,file)
    # with open(f"./pickled/data/pin2_time",mode="wb") as file:
    #     pickle.dump(event_time,file)
    
    n = 50000#len(event_times)#周期数->イベント数なのでは？
    data_fn = partial(pinwheel2, num_classes=num_classes)
    data = data_fn(n)#pinwheelを周期数*クラス数num_classesで生成
    seq = np.zeros((n, ndim + 1))# [時刻,x,y]
    seq[:, 0] = event_time
    
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):#dataを前から等分で wakeru iがクラスind、data_iが位置
        seq[:, 1:] = seq[:, 1:] + data_i * (i == class_his)[:, None]#適当に作ったものをインデックス番号で取ってくる。dataが適当にインデックス版に作られるから...
    return seq

def pin_wheel_generate3():
    ndim=2
    num_classes=3
    #event_times, classes = zip(*mhp.data)#event_times[0~T]までに発生したイベント時刻 classes:どのクラスなのか
    event_time, score, class_his = generate_hawkes_pin3(num_classes)
    class_his=np.array(class_his)
    # import pickle
    # with open(f"./pickled/data/pin3_class",mode="wb") as file:
    #     pickle.dump(class_his,file)
    # with open(f"./pickled/data/pin3_time",mode="wb") as file:
    #     pickle.dump(event_time,file)
    
    n = 30000#len(event_times)#周期数->イベント数なのでは？
    data_fn = partial(pinwheel2, num_classes=num_classes)
    data = data_fn(n)#pinwheelを周期数*クラス数num_classesで生成
    seq = np.zeros((n, ndim + 1))# [時刻,x,y]
    seq[:, 0] = event_time
    
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):#dataを前から等分で wakeru iがクラスind、data_iが位置
        seq[:, 1:] = seq[:, 1:] + data_i * (i == class_his)[:, None]#適当に作ったものをインデックス番号で取ってくる。dataが適当にインデックス版に作られるから...
    return seq
def pin_wheel_generate4():
    ndim=2
    num_classes=3
    #event_times, classes = zip(*mhp.data)#event_times[0~T]までに発生したイベント時刻 classes:どのクラスなのか
    event_time, score, class_his = generate_hawkes_pin4(num_classes)
    class_his=np.array(class_his)
    # import pickle
    # with open(f"./pickled/data/pin3_class",mode="wb") as file:
    #     pickle.dump(class_his,file)
    # with open(f"./pickled/data/pin3_time",mode="wb") as file:
    #     pickle.dump(event_time,file)
    
    n = 30000#len(event_times)#周期数->イベント数なのでは？
    data_fn = partial(pinwheel2, num_classes=num_classes)
    data = data_fn(n)#pinwheelを周期数*クラス数num_classesで生成
    seq = np.zeros((n, ndim + 1))# [時刻,x,y]
    seq[:, 0] = event_time
    
    for i, data_i in enumerate(np.split(data, num_classes, axis=0)):#dataを前から等分で wakeru iがクラスind、data_iが位置
        seq[:, 1:] = seq[:, 1:] + data_i * (i == class_his)[:, None]#適当に作ったものをインデックス番号で取ってくる。dataが適当にインデックス版に作られるから...
    return seq

def pinwheel2(num_samples, num_classes):
    radial_std = 0.3# radius
    tangential_std = 0.1#
    num_per_class = num_samples
    rate = 0.25
    rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

    features = np.random.randn(num_classes * num_per_class, 2) \
        * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)

    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    #pdb.set_trace()
    rotations = np.reshape(rotations.T, (-1, 2, 2))

    return 2 * np.einsum("ti,tij->tj", features, rotations)

# seq=pin_wheel_generate()
# import matplotlib.pyplot as plt
# for i in range(seq.shape[0]):
#     plt.scatter(seq[i][0],seq[i][1],c="C1")
# plt.savefig("test.png")
# pdb.set_trace()
# @contextlib.contextmanager
# def temporary_seed(seed):
#     state = np.random.get_state()
#     np.random.seed(seed)
#     try:
#         yield
#     finally:
#         np.random.set_state(state)


# if __name__ == "__main__":

#     import matplotlib
#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     num_classes = 10

#     rng = np.random.RandomState(13579)
#     data = pinwheel(num_classes, 1000, rng)

#     for i, data_i in enumerate(np.split(data, num_classes, axis=0)):
#         plt.scatter(data_i[:, 0], data_i[:, 1], c=f"C{i}", s=2)

#     plt.xlim([-4, 4])
#     plt.ylim([-4, 4])
#     plt.savefig(f"pinwheel{num_classes}.png")