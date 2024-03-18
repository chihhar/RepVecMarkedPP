import numpy as np
from scipy.stats import lognorm,gamma
from scipy.optimize import brentq
######################################################
### Data Generator
######################################################
def generate_stationary_poisson():
    np.random.seed(seed=32)
    tau = np.random.exponential(size=100000)
    T = tau.cumsum()
    score = 1
    return [T,score]

######################################################
### non-stationary poisson process
######################################################
def generate_nonstationary_poisson():
    np.random.seed(seed=32)
    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2*np.pi*t/L)*amp + 1
    l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos(2*np.pi*t1/L) )*amp   + (t2-t1)
    
    while 1:
        T = np.random.exponential(size=210000).cumsum()*0.5
        r = np.random.rand(210000)
        index = r < l_t(T)/2.0
        
        if index.sum() > 100000:
            T = T[index][:100000]
            score = - ( np.log(l_t(T[80000:])).sum() - l_int(T[80000-1],T[-1]) )/20000
            break
       
    return [T,score]

######################################################
### stationary renewal process
######################################################
def generate_stationary_renewal():
    np.random.seed(seed=32)
    s = np.sqrt(np.log(6*6+1))
    mu = -s*s/2
    tau = lognorm.rvs(s=s,scale=np.exp(mu),size=100000)
    lpdf = lognorm.logpdf(tau,s=s,scale=np.exp(mu))
    T = tau.cumsum()
    score = - np.mean(lpdf[80000:])
    
    return [T,score]

######################################################
### non-stationary renewal process
######################################################
def generate_nonstationary_renewal():
    np.random.seed(seed=32)
    L = 20000
    amp = 0.99
    l_t = lambda t: np.sin(2*np.pi*t/L)*amp + 1
    l_int = lambda t1,t2: - L/(2*np.pi)*( np.cos(2*np.pi*t2/L) - np.cos(2*np.pi*t1/L) )*amp   + (t2-t1)

    T = []
    lpdf = []
    x = 0

    k = 4
    rs = gamma.rvs(k,size=100000)
    lpdfs = gamma.logpdf(rs,k)
    rs = rs/k
    lpdfs = lpdfs + np.log(k)

    for i in np.arange(100000):
        x_next = brentq(lambda t: l_int(x,t) - rs[i],x,x+1000)
        like = l_t(x_next)
        T.append(x_next)
        lpdf.append(  lpdfs[i] + np.log(like) )  
        x = x_next

    T = np.array(T)
    lpdf = np.array(lpdf)
    score = - lpdf[80000:].mean()
    
    return [T,score]

######################################################
### self-correcting process
######################################################
def generate_self_correcting():
    np.random.seed(seed=32)
    
    def self_correcting_process(mu,alpha,n):
    
        t = 0
        x = 0
        T = []
        log_l = []
        Int_l = []
    
        for i in np.arange(n):
            e = np.random.exponential()
            tau = np.log( e*mu/np.exp(x) + 1 )/mu
            # e = ( np.exp(mu*tau)- 1 )*np.exp(x) /mu
            t = t+tau
            T.append(t)
            x = x + mu*tau
            log_l.append(x)
            Int_l.append(e)
            x = x -alpha

        return [np.array(T),np.array(log_l),np.array(Int_l)]
    
    [T,log_l,Int_l] = self_correcting_process(1,1,100000)
    score = - ( log_l[80000:] - Int_l[80000:] ).sum() / 20000
    
    return [T,score]

######################################################
### Hawkes process
######################################################
def generate_hawkes1():
    np.random.seed(seed=32)
    [T,LL] = simulate_hawkes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def generate_hawkes2():
    np.random.seed(seed=32)
    [T,LL] = simulate_hawkes(100000,0.2,[0.4,0.4],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def generate_eahawkes():
    np.random.seed(seed=32)
    [T,LL] = easyend_simulate_hawkes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        like = mu + l_trg1 + l_trg2
        step = np.random.exponential()/like
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/like: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)]

def easyend_simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    threshold=2.1
    while 1:
        like = mu + l_trg1 + l_trg2
        if like > threshold:#lの大きさでステップの大きさが切り替わる。　ランダム性はrandom.rand
            
            step = 3.0#い経過時間
        else:
            step = 0.5#
            #step = np.random.exponential()/l

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/like: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return [np.array(T),np.array(LL)]


def generate_hawkes_modes05():
    np.random.seed(seed=32)
    [T,LL,L_TRG1] = simulate_hawkes_modes05(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def generate_hawkes_pin3(class_num):
    np.random.seed(seed=32)
    [T,LL,L_TRG1,class_his] = simulate_hawkes_pin3(10000,0.2,[0.8,0.0],[1.0,20.0],class_suu=class_num)
    score = - LL[8000:].mean()
    return [T,score,class_his]

def simulate_hawkes_modes05(n,mu,alpha,beta,class_suu=10,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    
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
        like = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/like
        else: # short mode
            step = np.random.exponential(scale=0.25)/like

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/like: #accept
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
    
def generate_hawkes_modes():
    np.random.seed(seed=32)
    [T,LL,L_TRG1] = simulate_hawkes_modes(100000,0.2,[0.8,0.0],[1.0,20.0])
    score = - LL[80000:].mean()
    return [T,score]

def simulate_hawkes_modes(n,mu,alpha,beta,short_thre=1,long_thre=5):
    T = []
    LL = []
    L_TRG1 = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    is_long_mode = 0
    
    while 1:
        like = mu + l_trg1 + l_trg2
        #step = np.random.exponential(scale=1)/l

        if l_trg1 > long_thre:
            is_long_mode = 1

        if l_trg1 < short_thre:
            is_long_mode = 0

        if is_long_mode: # long mode
            step = step = np.random.exponential(scale=2)/like
        else: # short mode
            step = np.random.exponential(scale=0.5)/like

        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/like: #accept
            T.append(x)
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
        
    return [np.array(T),np.array(LL),np.array(L_TRG1)]

def main():
    import pdb
    [T,score_ref] = generate_hawkes_modes05()
    
    train=T[:,int(len(T)*0.80)]
    valid=T[int(len(T)*0.8):int(len(T)*0.9)]
    test=T[int(len(T)*0.9):]
    
    dif_train=np.ediff1d(train)
    dif_valid=np.ediff1d(valid)
    dif_test=np.ediff1d(test)
    pdb.set_trace()
    diff_T=np.ediff1d(T)
    print(T)
    
if __name__=="__main__":
    main()