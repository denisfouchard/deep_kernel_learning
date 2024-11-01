# A demonstration script for regression with deep RKHM:
# "Deep Learning with Kernels through RKHM and the Perron-Frobenius Operator".

import numpy.random as nr
import tensorflow as tf
import tensorflow.keras
import numpy as np
import time

d=10 # We focus on the C*-algebra of d by d matrices and its C*-subalgebras
datanum=1000 # number of training samples
tdatanum=1000 # number of test samples
fnum=1000 # number of samples used to construct the represntation space
epochs=1000 # number of epochs
c=0.001 # parameter in the Laplacian kernel
L=2 # number of the layers
dim=np.array([10,1]) # dimension of blocks for each layer
lam1=100 # Perron-Frobenius regularlization parameter
lam2=0.01 # regularlization parameter for ||f_L||

ind=np.arange(0,fnum,1,dtype=np.int32)

def tf_kron(a,b):
    a_shape = [a.shape[0],a.shape[1]]
    b_shape = [b.shape[0],b.shape[1]]
    return tf.reshape(tf.reshape(a,[a_shape[0],1,a_shape[1],1])*tf.reshape(b,[1,b_shape[0],1,b_shape[1]]),[a_shape[0]*b_shape[0],a_shape[1]*b_shape[1]])

def matchange(x,m,datanum,d):
    return tf.reshape(x,[1,datanum,m*d])
    
@tf.function
def opti(G,Gtmp,label,c1,opt,Gtest,testlabel):
    with tf.GradientTape(persistent=True) as tape :
        tape.watch(c1)
        ydata=tf.matmul(G,c1[0])
        ytestdata=tf.matmul(Gtest,c1[0])
        Gpre=[]
        GGtmp=tf.gather(Gtmp,indices=ind)
        Gpre.append(GGtmp)
        
        features=tf.reshape(ydata[ind[0]*dim[0]:(ind[0]+1)*dim[0],:],[dim[0],d])
        for i in range(fnum-1):
            features=tf.concat([features,tf.reshape(ydata[ind[i+1]*dim[0]:(ind[i+1]+1)*dim[0],:],[dim[0],d])],axis=0)


        for j in range(0,L-1,1):
            if j>=0:
                mdim=dim[j]
            tmp1=matchange(ydata,dim[j],datanum,d)
            tmp1test=matchange(ytestdata,dim[j],tdatanum,d)
            tmpf=matchange(features,dim[j],fnum,d)
            for i in range(fnum-1):
                tmp1=tf.concat([tmp1,matchange(ydata,dim[j],datanum,d)],axis=0)
            for i in range(fnum-1):
                tmp1test=tf.concat([tmp1test,matchange(ytestdata,dim[j],tdatanum,d)],axis=0)
            for i in range(tdatanum-1):
                tmpf=tf.concat([tmpf,matchange(features,dim[j],fnum,d)],axis=0)
        
            GG=tf.math.exp(-c*tf.reduce_sum(abs(tf.transpose(tmp1,(1,0,2))-tmpf[0:datanum,:,:]),axis=2))
            GGtmp=tf.gather(GG,indices=ind)
            GG=tf_kron(GG,tf.eye(dim[j+1]))
            
            Gpre.append(GGtmp)
            GGtest=tf.math.exp(-c*tf.reduce_sum(abs(tf.transpose(tmp1test,(1,0,2))-tmpf),axis=2))
            GGtest=tf_kron(GGtest,tf.eye(dim[j+1]))
            
            ydata=tf.matmul(GG,c1[j+1])
            ytestdata=tf.matmul(GGtest,c1[j+1])
            features=tf.reshape(ydata[ind[0]*dim[j+1]:(ind[0]+1)*dim[j+1],:],[dim[j+1],d])
            for i in range(fnum-1):
                features=tf.concat([features,tf.reshape(ydata[ind[i+1]*dim[j+1]:(ind[i+1]+1)*dim[j+1],:],[dim[j+1],d])],axis=0)

        

        reg1=lam1*(tf.norm(Gpre[L-1],2)+tf.norm(tf.linalg.solve(0.01*tf.eye(fnum)+Gpre[L-1],tf.eye(fnum)),2))
        reg2=lam2*tf.norm(tf.linalg.diag_part(tf.matmul(tf.matmul(tf.transpose(c1[L-1],(1,0)),GGtmp),c1[L-1])),np.inf)

        reg=reg1+reg2

        loss=tf.norm(tf.linalg.diag_part(tf.matmul(tf.transpose(ydata-tf.math.real(label),(1,0)),ydata-tf.math.real(label))),np.inf)/datanum
        lossreg=loss+reg
        testloss=tf.norm(tf.linalg.diag_part(tf.matmul(tf.transpose(ytestdata-tf.math.real(labeltest),(1,0)),ytestdata-tf.math.real(labeltest))),np.inf)/tdatanum
        grad=tape.gradient(lossreg,c1)
        opt.apply_gradients(zip(grad, c1))
        return abs(testloss-loss)

if __name__ == '__main__':
    nr.seed(0)
    a=nr.randn(d,d)
    xdata=np.zeros((datanum,d))
    ydata=np.zeros((datanum,d))
    for i in range(datanum):
        xdata[i,:]=0.1*nr.randn(d)
        ydata[i,:]=xdata[i,:]**2+0.001*nr.randn(d)
    xdata=tf.constant(xdata,dtype=tf.float32)
    label=tf.constant(ydata,dtype=tf.float32)

    xtestdata=np.zeros((tdatanum,d))
    ytestdata=np.zeros((tdatanum,d))
    for i in range(tdatanum):
        xtestdata[i,:]=0.1*nr.randn(d)
        ytestdata[i,:]=xtestdata[i,:]**2
    xtestdata=tf.constant(xtestdata,dtype=tf.float32)
    labeltest=tf.constant(ytestdata,dtype=tf.float32)


    ydata=xdata
    ytestdata=xtestdata
    features=xdata[ind[0],:]
    for i in range(fnum-1):
        features=tf.concat([features,xdata[ind[i+1],:]],axis=0)

    tmp1=tf.reshape(ydata,[1,datanum,d])
    tmp1test=tf.reshape(ytestdata,[1,tdatanum,d])
    tmpf=tf.reshape(features,[1,fnum,d])
    for i in range(fnum-1):
        tmp1=tf.concat([tmp1,tf.reshape(ydata,[1,datanum,d])],axis=0)
    for i in range(fnum-1):
        tmp1test=tf.concat([tmp1test,tf.reshape(ytestdata,[1,tdatanum,d])],axis=0)
    for i in range(tdatanum-1):
        tmpf=tf.concat([tmpf,tf.reshape(features,[1,fnum,d])],axis=0)

    G=tf.math.exp(-c*tf.reduce_sum(abs(tf.transpose(tmp1,(1,0,2))-tmpf[0:datanum,:,:]),axis=2))
    Gtmp=tf.gather(G,indices=ind)
    G=tf_kron(G,tf.eye(dim[0]))
    Gtest=tf.math.exp(-c*tf.reduce_sum(abs(tf.transpose(tmp1test,(1,0,2))-tmpf),axis=2))
    Gtest=tf_kron(Gtest,tf.eye(dim[0]))
    
    nr.seed(int(time.time()))

    c1=[]
    for j in range(L):
        c1.append(np.zeros((dim[j]*fnum,d)))

    for k in range(L):
        for i in range(fnum):
            for j in range(int(d/dim[k])):
                c1[k][i*dim[k]:(i+1)*dim[k],j*dim[k]:(j+1)*dim[k]]=0.1*np.ones((dim[k],dim[k]))+0.05*nr.randn(dim[k],dim[k])

    for j in range(L):   
        c1[j]=tf.Variable(c1[j],dtype=tf.float32)

    opt = tf.keras.optimizers.SGD(0.03*1e-2)
    
    for epoch in range(1,epochs+1,1):
        error=opti(G,Gtmp,label,c1,opt,Gtest,labeltest)
        print(epoch,"Generalization Error: ", error.numpy(),flush=True)
