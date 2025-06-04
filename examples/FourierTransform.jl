using CairoMakie
using TensorTrainNumerics
using FFTW

N = 16  
d = Int(log2(N))

F_qtt = TensorTrainNumerics.qtt_fft1(ComplexF64, d)

x_qtt_c = qtt_sin(8,λ=π)

y_qtt = F_qtt * x_qtt_c

# Convert result back to full vector
y_qtt_full = ttv_to_tensor(y_qtt)

# Compute standard FFT
x_c = ComplexF64.(x)
y_fft = fft(x_c)

# Plot comparison
fig = Figure()
ax1 = Axis(fig[1,1], title="Real part: QTT FFT vs FFTW", xlabel="k", ylabel="Re")
lines!(ax1, 0:N-1, real(y_fft), label="FFTW")
lines!(ax1, 0:N-1, real(y_qtt_full), label="QTT FFT")
axislegend(ax1)

ax2 = Axis(fig[2,1], title="Imag part: QTT FFT vs FFTW", xlabel="k", ylabel="Im")
lines!(ax2, 0:N-1, imag(y_fft), label="FFTW")
lines!(ax2, 0:N-1, imag(y_qtt_full), label="QTT FFT")
axislegend(ax2)

fig

