using Knet
using Knet: stack_isempty
include("readdata.jl")

# Trying an rnn model.

@knet function model05(x; hidden=0, o...)
    h = lstm(x; o..., out=hidden)
    if predict
        return wbf(h; o..., out=2, f=:soft)
    end
end

function rnnforw(f, x)
    # x is a (1,T,D,N) one-hot tensor
    # T:sequence length, D:input dimension, N:minibatch size
    global xdbg = x
    (X,T,D,N) = size(x)
    reset!(f)
    ypred = nothing
    for t=1:T
        xt = reshape(x[1,t,:,:], (D, N))
        ypred = sforw(f, xt; predict=(t==T))
    end
    return ypred
end

function train05(; N=2^10, lr=0.001, adam=true, nbatch=128, gclip=0, hidden=256, seed=0, o...)
    seed > 0 && setseed(seed)
    global f = compile(:model05; hidden=hidden, o...)
    setp(f, lr=lr, adam=adam)
    global d = GulayseData(nbatch)
    sloss = zloss = -1; n = nextn = 1
    for (x,ygold) in d
        ypred = rnnforw(f, x)
        sback(f, ygold, softloss)
        while !stack_isempty(f); sback(f); end
        gscale = (gclip > 0 ? min(1.0, gclip/gnorm(f)) : 1.0)
        update!(f; gscale=gscale)
        sl = softloss(ypred,ygold); sloss = (n==1 ? sl : 0.99 * sloss + 0.01 * sl)
        zl = zeroone(ypred,ygold);  zloss = (n==1 ? zl : 0.99 * zloss + 0.01 * zl)
        n==nextn && (println((n,sloss,1-zloss)); nextn*=2)
        (n += 1) > N && break
    end
end

train05()
