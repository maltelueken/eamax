import eamax

# Race log-densities are precision-sensitive and every consumer repo runs in float64;
# the reference tolerances here assume it.
eamax.enable_x64()
