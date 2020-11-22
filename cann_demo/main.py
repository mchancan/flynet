import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def main():
    data = np.loadtxt('Iapp_n100.csv', delimiter=',')
    
    Nunits = data.shape[0]+1
    Iapp_nn = np.zeros((Nunits,Nunits))
    Iapp_nn[:data.shape[0],:data.shape[0]] = data
    tmax = Nunits+1
    Ithresh = np.ones((Nunits,))
    rinit = np.zeros((1,Nunits)) # VELOCITY
    rinit[0] = 0

    #initialize a normal distribution with frozen in mean=-1, std. dev.= 1
    gs = norm(loc = 0., scale = 1)
    x = np.arange(-50, 51, 1)
    y = 3.5*gs.pdf(x)

    W = 1.2*np.ones((Nunits,Nunits))
    for i in range(Nunits-1):
        for j in range(Nunits-1):
            k = (j+50-i) % (Nunits-1)
            W[i,j] = y[k]
    W[Nunits-1,Nunits-1] = 0
    W[0:Nunits-1,Nunits-1] = -1.5

    dt = 0.001
    tvec = np.arange(0,tmax+dt,dt)
    Nt = tvec.size

    tau = 4
    r = np.zeros((Nunits,Nt))
    rmax = 100
    r[:,0] = rinit

    Iapp = np.zeros((Nunits,Nt))
    Idur = 0.4

    for i in range(Nunits):
        Iappi = Iapp_nn[int(i),:].reshape(Nunits,1)
        noni = np.round((i+1)/dt)
        noffi = np.round((i+1+Idur)/dt)
        Iapp[:,int(noni):int(noffi)] = Iappi*np.ones((1,int(noffi-noni)))

    out = np.zeros((Nunits+1,Nunits))

    j = 0
    for i in range(1,Nt):
        I = np.matmul(W,r[:,i-1]) + Iapp[:,i-1]
        newr = r[:,i-1] + dt/tau*(I-Ithresh-r[:,i-1])
        newr = np.clip(newr, 0, rmax)
        r[:,i] = newr
        if np.mod(i,np.round(Nt/Nunits)) == 0:
            idx = int(np.round(i*dt))
            out[j,:]= r[:,i]
            j+=1

    # FlyNet+CANN results: out.argmax(axis=0)
    
if __name__ == '__main__':
    main()