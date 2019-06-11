using CUDAdrv, CuArrays, CUDAnative, GPUArrays
CuArrays.allowscalar(false)
import Base.Broadcast: Broadcasted, ArrayStyle
using LazyArrays

function mycopyto_knl!(dest, bc)
  I = CuArrays.@cuindex dest
  @inbounds dest[I...] = bc[I...]
  nothing
end

function mycopyto!(dest, bc::Broadcasted{ArrayStyle{CuArray}})
  axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
  bc′ = Broadcast.preprocess(dest, bc)
  dev = CUDAdrv.device()
  thr = attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
  blk = length(dest) ÷ thr + 1
  @cuda blocks=blk threads=thr mycopyto_knl!(dest, bc)
  return dest
end

a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (1, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)

mycopyto!(d_c, @~ d_a .* d_b)

c = a .* b
c ≈ d_c
