using Knet
include("readdata.jl")

# Similar to model02, trying more fully connected layers

@knet function model03(x; T=51, D=4, W=3, o...)
    c1 = cbf73(x; o..., out=64)
    c2 = cbf73(c1; o..., out=128)
    c3 = cbf73(c2; o..., out=256)
    c = pool(c3; o..., window=T)
    f1 = wbf73(c; o..., out=128)
    f2 = wbf73(f1; o..., out=128)
    return wbf73(f2; o..., out=2, f=:soft)
end

function train03(; N=2^13, lr=0.001, adam=true, nbatch=128, o...)
    global f = compile(:model03; o...)
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

train03()
