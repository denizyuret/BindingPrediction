using Knet
include("readdata.jl")

# Try a convolutional model, similar to CharCNN in http://arxiv.org/pdf/1508.06615.pdf
# The input is a sequence of characters of length T.
# We map this to an embedding matrix DxT (alternative is to use one-hot with D=4).
# We apply a 1-D convolution bank (width and number of filters to be decided).
# In the simplest case with one conv layer, we apply max pooling of window T.
# We apply an affine transform and a softmax to the output.
# B shows the minibatch size, and everything is in 4D for CUDNN ops.

# x is a (1,T,D,N) matrix of inputs where T is the sequence length, D is the dimension of each char embedding, N is the minibatch size
# we convolve with a (1,W,D,C) filter bank with W/2 padding to avoid dimension change
# this gives a (1,T,C,N) output which gets a bias added and f applied, neither of which effects the dimension
# we apply max pooling with pwindow=T which gives an output of (1,1,C,N)
# we have a fully connected layer ending with a softmax at the end
# cbfp73 has the following default options each of which can be modified in model01:
# cwindow=3, cpadding=div(cwindow,2), cstride=1, cmode=Knet.CUDNN_CONVOLUTION,
# pwindow=2, ppadding=0, pstride=pwindow, pmode=Knet.CUDNN_POOLING_MAX,
# cinit=Xavier(), binit=Constant(0), f=:relu

@knet function model01(x; T=51, D=4, C=512, W=3, o...)
    y = cbfp73(x; o..., pwindow=T, out=C)
    return wbf73(y; o..., out=2, f=:soft)
end

function train01(; N=2^13, lr=0.001, adam=true, nbatch=128, o...)
    global f = compile(:model01; o...)
    setp(f, lr=lr, adam=adam)
    global d = GulayseData(nbatch)
    sloss = zloss = -1; n = nextn = 1
    for (x,ygold) in d
        ypred = forw(f, x)
        sl = softloss(ypred,ygold); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        back(f, ygold, softloss)
        update!(f)
        (n += 1) > N && break
    end
end

train01()
