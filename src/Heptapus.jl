module Heptapus

using TypedTables, CSV, CUDAdrv, CUDAnative, CuArrays

export empiricalbandwidth, Roofline

"""
    empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, numtests=10)

Compute the emperical bandwidth in GB/s of a CUDA device using `nbytes` of
memory.  The device to test can be selected with `devicenumber` and the
bandwidth is an average of `ntests`.
"""
function empiricalbandwidth(nbytes=2*1024^3; devicenumber=0, ntests=10,
                            use_memcpy=false)
  if use_memcpy
    return empiricalbandwidth_memcpy(nbytes, devicenumber, ntests)
  else
    return empiricalbandwidth_kernel(nbytes, devicenumber, ntests)
  end
end

function empiricalbandwidth_memcpy(nbytes, devicenumber, ntests)
    dev = CuDevice(devicenumber)
    ctx = CuContext(dev)

    a = Mem.alloc(Mem.Device, nbytes÷2)
    b = Mem.alloc(Mem.Device, nbytes÷2)

    stream = CuStream()
    Mem.copy!(a, b, nbytes÷2, async=true, stream=stream)
    Mem.copy!(b, a, nbytes÷2, async=true, stream=stream)

    t = CUDAdrv.@elapsed for n = 1:ntests
        Mem.copy!(a, b, nbytes÷2, async=true, stream=stream)
        Mem.copy!(b, a, nbytes÷2, async=true, stream=stream)
    end

    Mem.free(a)
    Mem.free(b)

    bandwidth = 2*nbytes*ntests/(t*1e9)
end

function bandwidth_kernel!(a, b, c, d)
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  if i <= length(a)
    @inbounds a[i] = CUDAnative.fma(b[i], c[i], d[i])
  end
  nothing
end

function empiricalbandwidth_kernel(nbytes, devicenumber, ntests)
    T = Float32

    quarter_nbytes = nbytes ÷ 4
    nelems = quarter_nbytes ÷ sizeof(T)
    a = ntuple(_ -> CuArray{T}(undef, nelems), 4)

    threads = 1024
    blocks = (nelems + threads - 1) ÷ threads

    @cuda threads=threads blocks=blocks bandwidth_kernel!(a[1], a[2], a[3], a[4])
    @cuda threads=threads blocks=blocks bandwidth_kernel!(a[2], a[3], a[4], a[1])
    @cuda threads=threads blocks=blocks bandwidth_kernel!(a[3], a[4], a[1], a[2])
    @cuda threads=threads blocks=blocks bandwidth_kernel!(a[4], a[1], a[2], a[3])

    t = CUDAdrv.@elapsed for n = 1:ntests
      @cuda threads=threads blocks=blocks bandwidth_kernel!(a[1], a[2], a[3], a[4])
      @cuda threads=threads blocks=blocks bandwidth_kernel!(a[2], a[3], a[4], a[1])
      @cuda threads=threads blocks=blocks bandwidth_kernel!(a[3], a[4], a[1], a[2])
      @cuda threads=threads blocks=blocks bandwidth_kernel!(a[4], a[1], a[2], a[3])
    end

    CuArrays.unsafe_free!.(a)

    bandwidth = 4nbytes*ntests/(t*1e9)
end

"""
    Roofline(command::Cmd)

Use `nvprof` to profile `command` and compute for each kernel executed:

  - arithmetic intensity;
  - performance (GFLOP/s);
  - kernel max empirical bandwidth (GB/s);
  - max GFLOP/s estimate (GFLOP/s);
  - max empirical bandwidth (GB/s);
  - local loads or stores (`Bool`);
  - floaing point type used; and
  - mixed floating point operations (`Bool`).
"""
struct Roofline
    t::Table

    function Roofline(command::Cmd; use_memcpy_bandwidth=false)
        s = mktemp() do f, _
            metrics = [:dram_write_bytes,
                       :dram_read_bytes,
                       :local_load_transactions,
                       :local_store_transactions,
                       :flop_hp_efficiency,
                       :flop_sp_efficiency,
                       :flop_dp_efficiency,
                       :flop_count_hp,
                       :flop_count_sp,
                       :flop_count_dp]
            cmd = `nvprof -u ms --csv --metrics $(join(metrics,",")) --log-file $f $command`
            @info "Getting metrics" cmd
            run(cmd)
            Table(CSV.File(f, comment="=", allowmissing=:none))
        end

        t = mktemp() do f, _
            cmd = `nvprof --print-gpu-summary -u ms --csv --log-file $f $command`
            @info "Getting timings" cmd
            run(cmd)
            Table(CSV.File(f, comment="=", allowmissing=:none, datarow=3))
        end

        kernels, matching_kernel = let
          kernels_t = unique(t.Name)
          kernels_s = unique(s.Kernel)

          @info "Kernels found in timings" kernels_t
          @info "Kernels found in measurements" kernels_s

          # remove CUDA memcpy/memset from timings
          filter!(s -> !occursin("[CUDA", s), kernels_t)

          # match kernels found in timings and measurements based on
          # lexicographical sorting
          matching_kernel = Dict(zip(sort(kernels_s), sort(kernels_t)))
          kernels_s, matching_kernel
        end

        getmetric(T, k, m; trim=0) =
            parse(T, s[map(row -> row.Kernel == k &&
                           getproperty(row, Symbol("Metric Name")) == String(m),
                           s)][1].Avg[1:end-trim])

        # get average kernel execution time in seconds
        gettime(k) = t[map(row -> row.Name == matching_kernel[k], t)][1].Avg/1e3

        eb = empiricalbandwidth(use_memcpy=use_memcpy_bandwidth)
        @info "Maximum empirical bandwidth $eb (GB/s)"

        maxempiricalbandwidth = fill(eb, length(kernels))
        arithmeticintensity = zeros(length(kernels))
        performance = similar(arithmeticintensity)
        kernelmaxempiricalbandwidth = similar(arithmeticintensity)
        maxgflopsestimate = similar(arithmeticintensity)
        haslocal = zeros(Bool, length(kernels))
        hasmixedflops = zeros(Bool, length(kernels))
        floptype = Array{Type}(undef, length(kernels))

        @info "Computing results"
        for (i, k) in enumerate(kernels)
            dram_write_bytes = getmetric(Int64, k, :dram_write_bytes)
            dram_read_bytes = getmetric(Int64, k, :dram_read_bytes)

            local_load_transactions = getmetric(Int64, k, :local_load_transactions)
            local_store_transactions = getmetric(Int64, k, :local_store_transactions)

            flop_efficiency = (hp=getmetric(Float64, k, :flop_hp_efficiency, trim=1),
                               sp=getmetric(Float64, k, :flop_sp_efficiency, trim=1),
                               dp=getmetric(Float64, k, :flop_dp_efficiency, trim=1))

            flop_count = (hp=getmetric(Int64, k, :flop_count_hp),
                          sp=getmetric(Int64, k, :flop_count_sp),
                          dp=getmetric(Int64, k, :flop_count_dp))

            elapsedtime = gettime(k)

            if local_load_transactions > 0 || local_store_transactions > 0
                haslocal[i] = true
                @warn """Kernel $k has nonzero load or store transactions.

                    This could be evidece of register spilling and the returned
                    roofline numbers will not be accurate for this kernel.

                """ local_load_transactions local_store_transactions
            end

            if sum(flop_count) != maximum(flop_count)
                hasmixedflops[i] = true
                @warn """Kernel $k has floating point operations for multiple types.

                    The performance results will use the floating point type with
                    the maximum flop count.  The may or may not be what is desired.

                    Note: `flop_count` contains half, single, and double precision.
                """ flop_count
            end

            p = argmax(flop_count)
            floptype[i] = (hp=Float16, sp=Float32, dp=Float64)[p]

            bytes = dram_write_bytes + dram_read_bytes
            flops = flop_count[p]

            arithmeticintensity[i] = flops/bytes

            # in GFLOP/s
            performance[i] = (flops/elapsedtime)/1e9
            maxgflopsestimate[i] = 100performance[i]/flop_efficiency[p]

            kernelmaxempiricalbandwidth[i] =
              empiricalbandwidth(bytes, use_memcpy=use_memcpy_bandwidth)
        end

        new(Table(kernels = kernels,
                  arithmeticintensity = arithmeticintensity,
                  performance = performance,
                  kernelmaxempiricalbandwidth = kernelmaxempiricalbandwidth,
                  maxgflopsestimate = maxgflopsestimate,
                  maxempiricalbandwidth = maxempiricalbandwidth,
                  haslocal = haslocal,
                  floptype = floptype,
                  hasmixedflops = hasmixedflops))
    end
end

function Base.show(io::IO, r::Roofline)
    print(io, "Roofline containing ")
    show(io, MIME"text/plain"(), r.t)
end

end # module
