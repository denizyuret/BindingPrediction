using Knet
include("readdata.jl")

# Copy of VGG model A: http://arxiv.org/pdf/1409.1556v6.pdf

@knet function model04(x; T=51, D=4, W=3, o...)
    c1 = cbfp73(x;  o..., out=64, pwindow=T)
    c2 = cbfp73(c1; o..., out=128, pwindow=T)
    c3 = cbf73(c2; o..., out=256)
    c4 = cbfp73(c3; o..., out=256, pwindow=T)
    c5 = cbf73(c4; o..., out=512)
    c6 = cbfp73(c5; o..., out=512, pwindow=T)
    c7 = cbf73(c6; o..., out=512)
    c8 = cbfp73(c7; o..., out=512, pwindow=T)
    f1 = wbf73(c8; o..., out=4096)
    f2 = wbf73(f1; o..., out=4096)
    return wbf73(f2; o..., out=2, f=:soft)
end

function train04(; N=2^13, lr=0.001, adam=true, nbatch=128, o...)
    global f = compile(:model04; o...)
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

train04()
