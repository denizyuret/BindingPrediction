using GZip
const file0 = "GFP ChIP-SEQ-antibody1.fastq.gz"
const file1 = "NeuroD2 ChIP-SEQ-antibody1.fastq.gz"

# The files are too big, we'll just write an iterator that reads from
# the file as needed.

import Base: start, done, next

type GulayseData
    nbatch::Int
    flip::Bool
    f0::GZipStream
    f1::GZipStream
    x::Array{Float32,4}
    y::Array{Float32,2}
    GulayseData(n=128,f=true)=new(n,f)
end

function start(d::GulayseData)
    isdefined(d,:f0) && close(f0)
    isdefined(d,:f1) && close(f1)
    d.f0 = GZip.open(file0)
    d.f1 = GZip.open(file1)
    d.x = zeros(Float32, 1, 51, 4, d.nbatch)
    d.y = zeros(Float32, 2, d.nbatch)
    return nothing
end

function done(d::GulayseData, s)
    eof(d.f0) || eof(d.f1)
end

function next(d::GulayseData, s)
    d.x[:] = d.y[:] = 0
    for n=1:d.nbatch      # how do we handle train vs test?
        d.flip ? d.y[1,n]=1 : d.y[2,n]=1 # do not have a loss function for 1-D output
        xread(d.flip ? d.f1 : d.f0, d.x, n) # alternate reading from positive and negative examples
        d.flip = !d.flip
    end
    return ((d.x, d.y), nothing)
end

function xread(fp::GZipStream, x::Array{Float32,4}, n::Int, s::ASCIIString="")
    for i=1:4
        eof(fp) && return
        tmp = readline(fp)
        i==2 && (s = tmp)
    end
    length(s)==52 || error("length(s)==$(length(s))")
    for i=1:51
        s[i] == 'A' ? x[1,i,1,n] = 1 :
        s[i] == 'C' ? x[1,i,2,n] = 1 :
        s[i] == 'G' ? x[1,i,3,n] = 1 :
        s[i] == 'T' ? x[1,i,4,n] = 1 :
        s[i] == 'N' ? nothing :
        error("s[$i]==$(s[i])")
    end
end
