tic()
 @simd for i = 1:Num_Cal #in range(1, Num_Cal)
    @inbounds E = Hertz.energy(V, Config)
      end
toc()

tic()
 @simd for i = 1:Num_Cal
  E = Fast_Hertz.energy(V1, Config1)
 end
toc()


# tic()
#  @simd for i = 1:Num_Cal #in range(1, Num_Cal)
#     @inbounds E1 = MultiHZ.energy(V1, Config1)
#       end
# toc()

# tic()
# for i in range(1, Num_Cal)
#     E1 = energy(V1, Config1)
# end
# time2 = toc()

# at = deepcopy(Config)

# tic()
# Minimise_Energy.min_Energy(at)
# toc()