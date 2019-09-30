using CUDAnative, CuArrays, CUDAdrv
# kernel doing 3m + 1 fused multiply adds and 4 loads
# arithmetic intensity = (3m + 1)  / (4 * sizeof(T)))
function fma_kernel!(out, a, b, c, ::Val{m}) where m
  i = (blockIdx().x-1) * blockDim().x + threadIdx().x
  @inbounds a_val = a[i]
  @inbounds b_val = b[i]
  @inbounds c_val = c[i]

  for j in 1:m
      a_val = CUDAnative.fma(a_val, b_val, c_val)
      b_val = CUDAnative.fma(a_val, b_val, c_val)
      c_val = CUDAnative.fma(a_val, b_val, c_val)
  end

  @inbounds out[i] = CUDAnative.fma(a_val, b_val, c_val)
  nothing
end

function main()
  nbytes = 1024 ^ 3
  nrep = 10
  for T in (Float32, Float64)
    n = div(nbytes, 4 * sizeof(T))
    threads = 1024
    blocks = div(n + threads - 1, threads)
    a, b, c, d = CuArray.(zeros.(T, ntuple(_ -> n, 4)))
    for arithmetic_intensity in 1:20
      m = div(arithmetic_intensity * 4 * sizeof(T), 3) - 1
      @cuda threads = threads blocks = blocks fma_kernel!(a, b, c, d, Val(m))
      @CUDAdrv.profile for r = 1:nrep
        @cuda threads = threads blocks = blocks fma_kernel!(a, b, c, d, Val(m))
      end
    end
    CuArrays.unsafe_free!(a)
    CuArrays.unsafe_free!(b)
    CuArrays.unsafe_free!(c)
    CuArrays.unsafe_free!(d)
  end
end

main()
