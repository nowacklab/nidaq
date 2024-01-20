# Hardware-limited NI DAQ measurements

See:

- `nidaq` for the hardware-limited sweep code.
- `examples` for different ways to interact with the NI DAQ.

## Calculate voltages from samples

Output:
```julia
outinvcoeff = [-outcoeff[1], 1.0] ./ outcoeff[2]
Vx = outinvcoeff[1] .+ outinvcoeff[2] .* x
```

Input:
```julia
inpoly(incoeff) = s -> incoeff[1] + incoeff[2]*s + incoeff[3]*s^2 + incoeff[4]*s^3
inp = inpoly(incoeff)
Vy = inp.(y)
```

## TODO

- Documentation
- Extract common functionality to enable creation of new scripts
- Session setup (set parameters like commit author?)
- Similarly automatic plotting
- Selective module loading (speed)

