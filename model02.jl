using Knet
include("readdata.jl")

# Similar to model01, trying more convolutional layers

@knet function model02(x; T=51, D=4, W=3, o...)
    a = cbf73(x; o..., out=64)
    b = cbf73(a; o..., out=128)
    c = cbf73(b; o..., out=256)
    y = pool(c; o..., window=T)
    return wbf73(y; o..., out=2, f=:soft)
end

function train02(; N=2^13, lr=0.001, adam=true, nbatch=128, o...)
    global f = compile(:model02; o...)
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

train02()
