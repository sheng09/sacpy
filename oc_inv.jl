#!/usr/bin/env julia

using PyCall
println(PyCall.python)
oc_inv = pyimport("oc_inv")



r  = [3500., 3000, 2800, 2900, 2500, 2500, 500]
vr = [  5.0, 5.1,  6.0,  13, 13.5,   12,  14]
z, vz = oc_inv.flatten(r, vr, 6371.0)

theta_step_deg = 0.1
niter = 100
critical_dist_err = 1e-10


dist_deg = collect(range(5, 90, length=20)) # where the data are
dist_km = dist_deg * (pi/180.0) * 6371.0  # in km
# first time run cost a bit longer due to compilation
_, _, _, par_trvt_v = oc_inv.many_dist2trvt_jac(dist_km, z, vz, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
@time _, _, _, par_trvt_v = oc_inv.many_dist2trvt_jac(dist_km, z, vz, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)

par_trvt_v2 = zeros(Float64, size(par_trvt_v) )
for iz in 1:length(z)
    vz1 = copy(vz)
    vz2 = copy(vz)
    vz1[iz] *= (1-1e-6)
    vz2[iz] *= (1+1e-6)
    #_, _, trvt1 = many_dist2trvt(dist, z,  vz1, theta_step_deg=theta_step_deg)
    #_, _, trvt2 = many_dist2trvt(dist, z,  vz2, theta_step_deg=theta_step_deg)
    _, _, trvt1, _ = oc_inv.many_dist2trvt_jac(dist_km, z, vz1, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
    _, _, trvt2, _ = oc_inv.many_dist2trvt_jac(dist_km, z, vz2, theta_step_deg=theta_step_deg, critical_dist_err=critical_dist_err, niter=niter)
    junk = (trvt2 - trvt1) / (2e-6*vz[iz])
    par_trvt_v2[:, iz] = junk
end
display(par_trvt_v)
display(par_trvt_v2)
display("Difference in percent:")
display((par_trvt_v2 - par_trvt_v) ./ par_trvt_v * 100)
