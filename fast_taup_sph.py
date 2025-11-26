#!/usr/bin/env python3

from pprint import pp
import numpy as np
from sacpy import oc_inv, fast_taup
from numba import jit
import inspect
import matplotlib.pyplot as plt
import pylab as pl


__zero_k_tol__           = 1e-9
__zero_dr_tol__          = 1e-5
__zero_p_sph_tol__       = 1e-2    # Empirical value: p=1e-2 correspond to theta = 0.0005 degree from surface with v=6km/s
__zero_k_mul_p_sph_tol__ = 1e-5    # Zero for c=p*k. Used in F2(...) function.
__next_after_zero_p_sph__= 1e-10  # The next float after zero for p in spherical model.
__ENABLE_NUMBA__ = False
def jit_wrapper(*args, **kwargs):
    """
    A decorator that returns numba.jit if __ENABLE_NUMBA__ is True,
    otherwise returns a simple identity function.
    """
    
    # 1. Check the global flag
    if __ENABLE_NUMBA__:
        # If Numba is enabled, return the actual @jit decorator
        # @jit can be called with or without arguments, so we pass them through
        return jit(*args, **kwargs)

    else:
        # 2. If Numba is disabled, return a function that acts as an identity decorator.
        # This function takes the original function 'func' and returns it unchanged.
        def identity_decorator(func):
            # We use inspect to ensure the returned function has the same metadata 
            # (like __name__) as the original, though often not strictly necessary 
            # for basic functionality.
            if not inspect.isfunction(func):
                # Handles cases where @maybe_jit is used without parentheses on a class method
                return func
            return func
        
        # If @maybe_jit was called with arguments (like @maybe_jit(nopython=True)), 
        # *args or **kwargs will be populated. In the disabled state, these args 
        # should be ignored, and we should return the identity function itself.
        if args or kwargs:
            return identity_decorator
        else:
            # If @maybe_jit was called without arguments (like @maybe_jit), 
            # the Python decorator mechanism calls the decorator with the function
            # as the first argument. We need to handle this case explicitly.
            if len(args) == 1 and inspect.isfunction(args[0]):
                return identity_decorator(args[0])
            return identity_decorator

######################################################################################
# Fundamental numerical components for spherical taup calculations
@jit_wrapper(nopython=True, nogil=True)
def G1(sin_v, cos_v):
    return np.log(sin_v/(1.+cos_v) )
@jit_wrapper(nopython=True, nogil=True)
def G2(sin_v, cos_v, c):
    if np.abs(c)<__zero_k_mul_p_sph_tol__:          # c=0.0
        return np.log(sin_v/(1.+cos_v))
    elif np.abs(c + 1.) < __zero_k_mul_p_sph_tol__: # c=-1.0
        return -cos_v/(1.+sin_v)
    elif np.abs(c - 1.) < __zero_k_mul_p_sph_tol__: # c=1.0
        return (-1.-sin_v)/cos_v
    elif np.abs(c) < 1.:                            # |c|<1.0
        v0 = c * sin_v/(1.+cos_v) - 1.
        v1 = np.sqrt(1. - c*c)
        return 1./v1 * np.log( np.abs((v0+v1) / (v0-v1)) )
    elif np.abs(c) > 1.:                            # |c|>1.0
        v0 = 1.-c*sin_v/(1.+cos_v)
        v1 = np.sqrt(c*c - 1.)
        return 2. /v1 * np.arctan( v0 / v1 )
@jit_wrapper(nopython=True, nogil=True)
def G3(sin_v, cos_v, c):
    if np.abs(c)<__zero_k_mul_p_sph_tol__:          # c=0.0
        return -cos_v/sin_v
    elif np.abs(c + 1.) < __zero_k_mul_p_sph_tol__: # c=-1.0
        v0 = (1.+sin_v)
        return -cos_v*(sin_v+2.)/(3.*v0*v0)
    elif np.abs(c - 1.) < __zero_k_mul_p_sph_tol__: # c=1.0
        return -2./(3.0*cos_v*(sin_v-1.0)) + sin_v/(3.*cos_v)
    else:
        v0 = c*c-1.
        return cos_v/((sin_v-c)*v0) - c/v0 *G2(sin_v, cos_v, c)
@jit_wrapper(nopython=True, nogil=True)
def G4(sin_v, cos_v, c):
    if np.abs(c)>__zero_k_mul_p_sph_tol__:          # c!=0.0
        v0 = -1./c
        return v0*np.log(sin_v/(1.+cos_v) )  - v0*G2(sin_v, cos_v, c) #v0*G1(sin_v, cos_v) - v0*G2(sin_v, cos_v, c)
    else:
        return -cos_v/sin_v
@jit_wrapper(nopython=True, nogil=True)
def G5(sin_v, cos_v, c):
    if np.abs(c)>__zero_k_mul_p_sph_tol__:          # c!=0.0
        return G2(sin_v, cos_v, c) + c*G3(sin_v, cos_v, c)
    else:
        return np.log(sin_v/(1.+cos_v) ) # G1(sin_v, cos_v)
@jit_wrapper(nopython=True, nogil=True)
def G6(sin_v, cos_v, c):
    if np.abs(c)>__zero_k_mul_p_sph_tol__:          # c!=0.0
        v0 = 1./c
        v1 = v0*v0
        return v1*np.log(sin_v/(1.+cos_v) ) - v1*G2(sin_v, cos_v, c) + v0*G3(sin_v, cos_v, c)  #v1*G1(sin_v, cos_v) - v1*G2(sin_v, cos_v, c) + v0*G3(sin_v, cos_v, c)
    else:
        return 0.5* ( np.log(sin_v/(1.+cos_v) ) - cos_v/(sin_v*sin_v) ) #0.5* ( G1(sin_v, cos_v) - cos_v/(sin_v*sin_v) )

#def F1_v2(sin_v, cos_v): # np.log(sin_v/(1.0+cos_v))
#    return np.log(sin_v/(1.0+cos_v))
#def par_F1_par_theta_v2(sin_v, cos_v): # 1.0/sin_v
#    return 1.0/sin_v
@jit_wrapper(nopython=True, nogil=True)
def F2(sin_v, cos_v, c):
    """
    For solving the 2nd integral in spherical distance-traveltime derivations.
    #
    sin_v: sin(theta) at an interface.
    cos_v: cos(theta) at an interface.
    c:     p*k, where p is the ray parameter (s/rad) and k is the velocity gradient in the layer.
           The definition of k is k = (vbot-vtop)/(rbot-rtop).
    #
    Return: the value of the integral.
    #
    Don't worry for theta=pi/2 case, as F2 works for it.
    """
    if np.abs(c)<__zero_k_mul_p_sph_tol__:           # c=0.0
        return np.log(sin_v/(1.0+cos_v)) # return F1_v2(sin_v, cos_v)
    elif np.abs(c + 1.0) < __zero_k_mul_p_sph_tol__: # c=-1.0
        return -cos_v/(1.0+sin_v) # the same as (-1.0 +sin_v)/cos_v but safer for theta=pi/2
    elif np.abs(c - 1.0) < __zero_k_mul_p_sph_tol__: # c=1.0 (this cannot happen if theta=pi/2)
        #################################################################################
        # seems nan if theta=pi/2
        # can this happen?
        #################################################################################
        # This require c=p*k=1.0. So k=1/p>0.0
        # Let's check if a ray can penetrate rtop and turn at rbot at the same time.
        # vbot = vtop + k*(rbot-rtop) = vtop + (rbot-rtop)/p
        # (1) The ray penetrates rtop. This requires:
        #     p = rtop * sin(theta_top)/vtop < rtop/vtop
        # (2) The ray turns at rbot. This requires:
        #     p = rbot/vbot
        # So we have:
        #    rbot/vbot = p < rtop/vtop
        #   <==> rbot*vtop < rtop*vbot
        #   <==> rbot*vtop < rtop*[vtop + (rbot-rtop)/p]
        #   <==> rbot*vtop < rtop*vtop + rtop*(rbot-rtop)/p
        #   <==> (rbot-rtop)*vtop < rtop*(rbot-rtop)/p    !!! Note: rbot<rtop
        #   <==>             vtop > rtop/p
        #   <==>             p    > rtop/vtop
        # This contradicts (1). So this case cannot happen.
        #################################################################################
        return (-1.0 -sin_v)/cos_v
    elif np.abs(c) < 1.0:        # |c|<1.0
        v0 = sin_v/(1.0+cos_v)
        v1 = np.sqrt(1.0 - c*c)
        v2 = c*v0-1.0
        return 1.0/v1 * np.log( np.abs((v2+v1) / (v2-v1)) ) # = 1.0/v1 * log((c-1+v1)/(c-1-v1))
    elif np.abs(c) > 1.0:        # |c|>1.0
        v0 = sin_v/(1.0+cos_v)
        v1 = np.sqrt(c*c - 1.0)
        return 2.0 / v1 * np.arctan( (1.0-c*v0) / v1 ) # = 2.0/v1 * arctan((1.0-c)/v1)
@jit_wrapper(nopython=True, nogil=True)
def F2_jac(sin_v, cos_v, c, F2_value):
    """
    Return pf2_ptheta, pf2_pc
    For theta=pi/2 case, the `pf2_ptheta` is meaningless, but `pf2_pc` is still valid.
    """
    pf2_ptheta = 1.0/(sin_v-c) # don't worry for c=sin_v case, as it can never happen
    if (np.abs(c)<__zero_k_mul_p_sph_tol__) or (np.abs(c + 1.0) < __zero_k_mul_p_sph_tol__) or (np.abs(c - 1.0) < __zero_k_mul_p_sph_tol__):
        pf2_pc = 0.0
        # here could be wrong!
        # Fixed with G? functions. Now F2 are not used anymore
    else:
        tan_half_v = sin_v/(1.0+cos_v)
        v1 = 1.0-c*c
        pf2_pc = c/v1 * F2_value + 2.0*(c-tan_half_v) / (v1 * c * (c - 2*tan_half_v + c*tan_half_v*tan_half_v) )
    return pf2_ptheta, pf2_pc

# dx,dt for p != 0
@jit_wrapper(nopython=True, nogil=True)
def lyr_not_const_vel(k, c=1e-9):
    return np.abs(k) > c
@jit_wrapper(nopython=True, nogil=True) # dx,dt for a layer penetrated with non-zero ray parameter
def deprecated_dxdt4lyr(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
             rbot, vbot, sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
             p, dr, k, inv_k, not_const_vel, debug=False):
    """
    Return dx, dt.
    #
    `dr` equals to rbot-rtop for penetrated layer, and rc-rtop for the critical layer.
    """
    # non-zero thickness layer. already checked while calling this function
    if not_const_vel:
        c      = p*k
        f2_top = F2(sin_top, cos_top, c)
        f2_bot = F2(sin_bot, cos_bot, c)
        dx     = c*(f2_bot-f2_top) + (theta_bot-theta_top)
        dt     = (inv_k)*(f1_top-f1_bot - f2_top+f2_bot)
        #if debug:
        #    dt2     = p*(cot_top-cot_bot)
        #    print('non-constant velocity layer in dxdt4lyr', k, dt, dt2)
        #    print('f1_top-f1_bot=', f1_top-f1_bot, '-f2_top+f2_bot=', - f2_top+f2_bot)
    else: # constant velocity layer
        dx     = (theta_bot-theta_top)
        dt     = p*(cot_top-cot_bot)
        #if debug:
        #    print('constant velocity layer in dxdt4lyr', k, dt)
    return dx, dt
@jit_wrapper(nopython=True, nogil=True) # Jac   for a layer penetrated with non-zero ray parameter
def deprecated_dxdt4plyr_jac(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
                  rbot, vbot, sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                  p, dr, k, inv_k, not_const_vel):
    """
    Return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot
    """
    ###
    # non-zero thickness layer. already checked while calling this function
    #####
    # There are only three independent variables: p, vtop, vbot
    # We will calculate the partial derivatives to these three variables.
    # Intermediate variables: k, c, theta_top, theta_bot, f1_top, f1_bot, f2_top, f2_bot
    ##### theta_top = theta_top(p, vtop)   rtop sin(theta_top) / vtop  = p
    tmp = rtop*cos_top
    pthetatop_pp    = vtop/tmp # if cos_top != 0.0 else 0.0 # pi/2 cannot happen for a penetrating layer
    pthetatop_pvtop =    p/tmp # if cos_top != 0.0 else 0.0
    ##### theta_bot = theta_bot(p, vbot)
    tmp = rbot*cos_bot
    pthetabot_pp    = vbot/tmp # if cos_bot != 0.0 else 0.0
    pthetabot_pvbot =    p/tmp # if cos_bot != 0.0 else 0.0
    if not_const_vel:
        c = p*k
        f2_top = F2(sin_top, cos_top, c)
        f2_bot = F2(sin_bot, cos_bot, c)
        #print(f'k={k} in dxdt4plyr_jac, dx={dx}', c*(f2_bot-f2_top), (theta_bot-theta_top) )
        ##### k = k(vtop, vbot) = (vbot-vtop)/(rbot-rtop) = (vbot-vtop)/dr
        pk_pvtop = -1.0/dr
        pk_pvbot = -pk_pvtop
        ##### c  = p * k = p * k(vtop, vbot)
        pc_pk    = p
        pc_pp    = k
        pc_pvtop = pc_pk * pk_pvtop
        pc_pvbot = pc_pk * pk_pvbot
        ##### f1_top = f1_top(theta_top) = f1_top( theta_top(p, vtop) )
        pf1top_pthetatop = 1.0/sin_top
        pf1top_pp        = pf1top_pthetatop * pthetatop_pp
        pf1top_pvtop     = pf1top_pthetatop * pthetatop_pvtop
        ##### f1_bot = f1_bot(theta_bot) = f1_bot( theta_bot(p, vbot) )
        pf1bot_pthetabot = 1.0/sin_bot
        pf1bot_pp        = pf1bot_pthetabot * pthetabot_pp
        pf1bot_pvbot     = pf1bot_pthetabot * pthetabot_pvbot
        ##### f2_top = f2_top(theta_top, c) = f2_top(theta_top(p, vtop), c) 
        pf2top_pthetatop, pf2top_pc= F2_jac(sin_top, cos_top, c, f2_top)
        pf2top_pp    = pf2top_pthetatop * pthetatop_pp    + pf2top_pc * pc_pp
        pf2top_pvtop = pf2top_pthetatop * pthetatop_pvtop + pf2top_pc * pc_pvtop
        pf2top_pvbot =                                      pf2top_pc * pc_pvbot
        ##### f2_bot = f2_bot(theta_bot, c) = f2_bot(theta_bot(p, vbot), c)
        pf2bot_pthetabot, pf2bot_pc= F2_jac(sin_bot, cos_bot, c, f2_bot)
        pf2bot_pp    = pf2bot_pthetabot * pthetabot_pp    + pf2bot_pc * pc_pp
        pf2bot_pvtop =                                      pf2bot_pc * pc_pvtop
        pf2bot_pvbot = pf2bot_pthetabot * pthetabot_pvbot + pf2bot_pc * pc_pvbot
        #####
        dx        = c*(f2_bot-f2_top) + (theta_bot-theta_top)
        pdx_pp    = pc_pp    * (f2_bot-f2_top) + c*(pf2bot_pp    - pf2top_pp)    + (pthetabot_pp    - pthetatop_pp)
        pdx_pvtop = pc_pvtop * (f2_bot-f2_top) + c*(pf2bot_pvtop - pf2top_pvtop) + (                - pthetatop_pvtop)
        pdx_pvbot = pc_pvbot * (f2_bot-f2_top) + c*(pf2bot_pvbot - pf2top_pvbot) + (pthetabot_pvbot                  )
        #####
        dt        = (inv_k)*(f1_top-f1_bot - f2_top+f2_bot)
        pdt_pp    = inv_k*( pf1top_pp    - pf1bot_pp    - pf2top_pp    + pf2bot_pp    )
        pdt_pvtop = inv_k*( pf1top_pvtop                - pf2top_pvtop + pf2bot_pvtop ) - pk_pvtop*inv_k * dt
        pdt_pvbot = inv_k*(              - pf1bot_pvbot - pf2top_pvbot + pf2bot_pvbot ) - pk_pvbot*inv_k * dt
    else:
        #####
        dx        = (theta_bot-theta_top)
        #pdx_pp    = -vtop/(rtop*cos_top) + vbot/(rbot*cos_bot)
        #pdx_pvbot =  p/(rbot*cos_bot) - p/dr*( np.log(sin_top/(1.+cos_top)) - np.log(sin_bot/(1.+cos_bot)) )
        #pdx_pvtop = -p/(rtop*cos_top) + p/dr*( np.log(sin_top/(1.+cos_top)) - np.log(sin_bot/(1.+cos_bot)) )
        tmp        =                     p/dr*( np.log(sin_top/(1.+cos_top)) - np.log(sin_bot/(1.+cos_bot)) )
        pdx_pp     =  pthetabot_pp - pthetatop_pp
        pdx_pvbot  =  pthetabot_pvbot - tmp
        pdx_pvtop  = -pthetatop_pvtop + tmp
        #####
        dt        = p*(cot_top-cot_bot)
        #pdt_pp    = -p*vtop/(sin_top*sin_top*rtop*cos_top) + p*vbot/(sin_bot*sin_bot*rbot*cos_bot) + cot_top - cot_bot
        #pdt_pvbot =  p*p/(sin_bot*sin_bot*rbot*cos_bot)    - p*p/dr * (  0.5*np.log(sin_top/(1.+cos_top)) - cos_top/(2.*sin_top*sin_top) 
        #                                                                -0.5*np.log(sin_bot/(1.+cos_bot)) + cos_bot/(2.*sin_bot*sin_bot) )
        #pdt_pvtop = -p*p/(sin_top*sin_top*rtop*cos_top)    + p*p/dr * (  0.5*np.log(sin_top/(1.+cos_top)) - cos_top/(2.*sin_top*sin_top) 
        #                                                                -0.5*np.log(sin_bot/(1.+cos_bot)) + cos_bot/(2.*sin_bot*sin_bot) )
        tmp0      = sin_top*sin_top
        tmp1      = sin_bot*sin_bot
        tmp2      = -p/tmp0
        tmp3      = p/tmp1
        pdt_pp    =  tmp2 * pthetatop_pp + tmp3 * pthetabot_pp + cot_top - cot_bot
        tmp4      = 0.5*p*p/dr * (  np.log(sin_top/(1.+cos_top)) - cos_top/tmp0
                                   -np.log(sin_bot/(1.+cos_bot)) + cos_bot/tmp1 )
        pdt_pvbot =  tmp3 * pthetabot_pvbot - tmp4
        pdt_pvtop =  tmp2 * pthetatop_pvtop + tmp4
        #####
    return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot
@jit_wrapper(nopython=True, nogil=True) # for critical layer where the ray turns (ray parameter p also non-zero)
def deprecated_dxdt4clyr_jac(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
                  p, dr, k, inv_k, not_const_vel): # dr is rbot-rtop
    ###
    # non-zero thickness layer. already checked while calling this function
    #####
    # There are only three independent variables: p, vtop, vbot
    # We will calculate the partial derivatives to these three variables.
    ##### theta_top = theta_top(p, vtop)   rtop sin(theta_top) / vtop  = p
    tmp = rtop*cos_top
    pthetatop_pp    = vtop/tmp # if cos_top != 0.0 else 0.0 # pi/2 cannot happen at the top interface
    pthetatop_pvtop =    p/tmp # if cos_top != 0.0 else 0.0
    ##### theta_bot = pi/2, a constant
    if not_const_vel:
        c = p*k
        f2_top = F2(sin_top, cos_top, c)
        f2_bot = F2(1.0,     0.0,     c)
        ##### k = k(vtop, vbot) = (vbot-vtop)/(rbot-rtop) = (vbot-vtop)/dr
        pk_pvtop = -1.0/dr
        pk_pvbot = -pk_pvtop
        ##### c  = p * k = p * k(vtop, vbot)
        pc_pk = p
        pc_pp = k 
        pc_pvtop = pc_pk * pk_pvtop
        pc_pvbot = pc_pk * pk_pvbot
        ##### f1_top = f1_top(theta_top) = f1_top( theta_top(p, vtop) )
        pf1top_pthetatop = 1.0/sin_top
        pf1top_pp        = pf1top_pthetatop * pthetatop_pp
        pf1top_pvtop     = pf1top_pthetatop * pthetatop_pvtop
        ##### f1_bot = f2_bot(pi/2) = 0.0, a constant
        ##### f2_top = f2_top(theta_top, c) = f2_top(theta_top(p, vtop), c) 
        pf2top_pthetatop, pf2top_pc= F2_jac(sin_top, cos_top, c, f2_top)
        pf2top_pp    = pf2top_pthetatop * pthetatop_pp    + pf2top_pc * pc_pp
        pf2top_pvtop = pf2top_pthetatop * pthetatop_pvtop + pf2top_pc * pc_pvtop
        pf2top_pvbot =                                      pf2top_pc * pc_pvbot
        ##### f2_bot = f2_bot(pi/2, c) = f2_bot(c)
        _,                pf2bot_pc= F2_jac(1.0,     0.0,     c, f2_bot) # F2 = F2(c)
        pf2bot_pp    =                                      pf2bot_pc * pc_pp
        pf2bot_pvtop =                                      pf2bot_pc * pc_pvtop
        pf2bot_pvbot =                                      pf2bot_pc * pc_pvbot
        #####
        dx        = c*(f2_bot-f2_top) + (0.5*np.pi-theta_top)
        pdx_pp    = pc_pp    * (f2_bot-f2_top) + c*(pf2bot_pp    - pf2top_pp)     - pthetatop_pp
        pdx_pvtop = pc_pvtop * (f2_bot-f2_top) + c*(pf2bot_pvtop - pf2top_pvtop)  - pthetatop_pvtop
        pdx_pvbot = pc_pvbot * (f2_bot-f2_top) + c*(pf2bot_pvbot - pf2top_pvbot)
        #####
        dt        = (inv_k)*(f1_top - f2_top+f2_bot)
        pdt_pp    = inv_k*( pf1top_pp      - pf2top_pp    + pf2bot_pp    )
        pdt_pvtop = inv_k*( pf1top_pvtop   - pf2top_pvtop + pf2bot_pvtop ) - inv_k*pk_pvtop* dt
        pdt_pvbot = inv_k*(                - pf2top_pvbot + pf2bot_pvbot ) - inv_k*pk_pvbot* dt
    else:
        #####
        dx        = (0.5*np.pi-theta_top)
        #pdx_pp    = -vtop/(rtop*cos_top)
        #pdx_pvbot =                   - p/dr* np.log(sin_top/(1.+cos_top)) 
        #pdx_pvtop = -p/(rtop*cos_top) + p/dr* np.log(sin_top/(1.+cos_top))
        tmp        =                     p/dr* np.log(sin_top/(1.+cos_top))
        pdx_pp     = -pthetatop_pp
        pdx_pvbot  =                   - tmp
        pdx_pvtop  = -pthetatop_pvtop  + tmp
        #####
        dt        = p*(cot_top)
        #pdt_pp    = -p*vtop/(sin_top*sin_top*rtop*cos_top) + cot_top
        #pdt_pvbot =                                     - p*p/dr * (  0.5*np.log(sin_top/(1.+cos_top)) - cos_top/(2.*sin_top*sin_top) )
        #pdt_pvtop = -p*p/(sin_top*sin_top*rtop*cos_top) + p*p/dr * (  0.5*np.log(sin_top/(1.+cos_top)) - cos_top/(2.*sin_top*sin_top) )
        tmp0       = sin_top*sin_top
        tmp2       = -p/tmp0
        tmp4       = p*p/dr * (  0.5*np.log(sin_top/(1.+cos_top)) - cos_top/(2.*tmp0) )
        pdt_pp     = tmp2 * pthetatop_pp  + cot_top
        pdt_pvbot  =                        - tmp4
        pdt_pvtop  = tmp2 * pthetatop_pvtop + tmp4
    return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot

@jit_wrapper(nopython=True, nogil=True) # dx,dt for a layer penetrated with non-zero ray parameter
def dxdt4lyr(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
             rbot, vbot, sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
             p, dr, k, inv_k, not_const_vel, debug=False):
    """
    Return dx, dt.
    #
    `dr` equals to rbot-rtop for penetrated layer, and rc-rtop for the critical layer.
    """
    # non-zero thickness layer. already checked while calling this function
    if not_const_vel:
        c      = p*k
        neg_G2dif = G2(sin_bot, cos_bot, c) - G2(sin_top, cos_top, c)
        dx     = c*neg_G2dif + (theta_bot-theta_top)
        dt     = (inv_k)*(f1_top-f1_bot + neg_G2dif)
    else: # constant velocity layer
        dx     = (theta_bot-theta_top)
        dt     = p*(cot_top-cot_bot)
    return dx, dt
@jit_wrapper(nopython=True, nogil=True) # Jac   for a layer penetrated with non-zero ray parameter
def dxdt4plyr_jac(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
                  rbot, vbot, sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                  p, dr, k, inv_k, not_const_vel):
    ##### theta_top = theta_top(p, vtop)   rtop sin(theta_top) / vtop  = p
    tmp = rtop*cos_top
    pthetatop_pp    = vtop/tmp # if cos_top != 0.0 else 0.0 # pi/2 cannot happen for a penetrating layer
    pthetatop_pvtop =    p/tmp # if cos_top != 0.0 else 0.0
    ##### theta_bot = theta_bot(p, vbot)
    tmp = rbot*cos_bot
    pthetabot_pp    = vbot/tmp # if cos_bot != 0.0 else 0.0
    pthetabot_pvbot =    p/tmp # if cos_bot != 0.0 else 0.0
    #####
    c = p*k
    dx = c*( G2(sin_bot, cos_bot, c) - G2(sin_top, cos_top, c) ) + theta_bot - theta_top
    dt = p*( G4(sin_bot, cos_bot, c) - G4(sin_top, cos_top, c) )
    #####
    p_dr = p/dr
    #####
    minus_sin_sin_minus_c_top = -sin_top/(sin_top - c )
    minus_sin_sin_minus_c_bot = -sin_bot/(sin_bot - c )
    G5_dif = G5(sin_top, cos_top, c) - G5(sin_bot, cos_bot, c)
    pdx_pp    = minus_sin_sin_minus_c_top * pthetatop_pp    - minus_sin_sin_minus_c_bot * pthetabot_pp        -k    * G5_dif
    pdx_pvbot =                                             - minus_sin_sin_minus_c_bot * pthetabot_pvbot     -p_dr * G5_dif
    pdx_pvtop = minus_sin_sin_minus_c_top * pthetatop_pvtop                                                   +p_dr * G5_dif
    #####
    minus_p_sin_sin_minus_c_top = -p/(sin_top*(sin_top-c))
    minus_p_sin_sin_minus_c_bot = -p/(sin_bot*(sin_bot-c))
    G3_dif = G3(sin_top, cos_top, c) - G3(sin_bot, cos_bot, c)
    G6_dif = G6(sin_top, cos_top, c) - G6(sin_bot, cos_bot, c)
    pdt_pp    = minus_p_sin_sin_minus_c_top * pthetatop_pp   - minus_p_sin_sin_minus_c_bot * pthetabot_pp      - G3_dif
    pdt_pvbot =                                              - minus_p_sin_sin_minus_c_bot * pthetabot_pvbot   - p*p_dr * G6_dif
    pdt_pvtop = minus_p_sin_sin_minus_c_top * pthetatop_pvtop                                                  + p*p_dr * G6_dif
    return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot
@jit_wrapper(nopython=True, nogil=True) # for critical layer where the ray turns (ray parameter p also non-zero)
def dxdt4clyr_jac(rtop, vtop, sin_top, cos_top, cot_top, theta_top, f1_top,
                  p, dr, k, inv_k, not_const_vel): # dr is rbot-rtop
    ##### theta_top = theta_top(p, vtop)   rtop sin(theta_top) / vtop  = p
    tmp = rtop*cos_top
    pthetatop_pp    = vtop/tmp # if cos_top != 0.0 else 0.0 # pi/2 cannot happen for a penetrating layer
    pthetatop_pvtop =    p/tmp # if cos_top != 0.0 else 0.0
    ##### theta_bot = pi/2 const
    sin_bot = 1.0
    cos_bot = 0.0
    theta_bot = np.pi*0.5
    #####
    c = p*k
    dx = c*( G2(sin_bot, cos_bot, c) - G2(sin_top, cos_top, c) ) + theta_bot - theta_top
    dt = p*( G4(sin_bot, cos_bot, c) - G4(sin_top, cos_top, c) )
    #####
    p_dr = p/dr
    #####
    minus_sin_sin_minus_c_top = -sin_top/(sin_top - c )
    G5_dif = G5(sin_top, cos_top, c) - G5(sin_bot, cos_bot, c)
    pdx_pp    = minus_sin_sin_minus_c_top * pthetatop_pp      -k    * G5_dif
    pdx_pvbot =                                               -p_dr * G5_dif
    pdx_pvtop = minus_sin_sin_minus_c_top * pthetatop_pvtop   +p_dr * G5_dif
    #####
    minus_p_sin_sin_minus_c_top = -p/(sin_top*(sin_top-c))
    G3_dif = G3(sin_top, cos_top, c) - G3(sin_bot, cos_bot, c)
    G6_dif = G6(sin_top, cos_top, c) - G6(sin_bot, cos_bot, c)
    pdt_pp    = minus_p_sin_sin_minus_c_top * pthetatop_pp    - G3_dif
    pdt_pvbot =                                               - p*p_dr * G6_dif
    pdt_pvtop = minus_p_sin_sin_minus_c_top * pthetatop_pvtop + p*p_dr * G6_dif
    return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot

# dt for p == 0
@jit_wrapper(nopython=True, nogil=True)
def dxdt4lyr_zeroP(rtop, vtop, rbot, vbot, dr, k, inv_k, not_const_vel):
    """
    Return dt.
    # Note, dx is always zero for p==0 case.
    """
    # non-zero thickness layer. already checked while calling this function
    if not_const_vel:
        dt = inv_k * np.log(vtop/vbot)
    else: # constant velocity layer
        dt = (rtop-rbot) / vtop
    return dt
@jit_wrapper(nopython=True, nogil=True)
def dxdt4lyr_zeroP_jac(rtop, vtop, rbot, vbot, dr, k, inv_k, not_const_vel):
    """
    Return dt, pdt_pp, pdt_pvtop, pdt_pvbot
    # Note, dx is always zero for p=0 case, so that pdx_pu are all zero for any variable u.
    """
    # non-zero thickness layer. already checked while calling this function
    if not_const_vel:
        #####
        dx = 0.
        pdx_pp    = (rbot-rtop)*(rtop*vbot*vbot - rbot*vtop*vtop)/(rbot*rtop*(rbot*vtop-rtop*vbot)) + k*(np.log((vbot*rtop)/(vtop*rbot)))
        pdx_pvbot = 0.
        pdx_pvtop = 0.
        #####
        dt = inv_k * np.log(vtop/vbot)
        ##### k = k(vtop, vbot) = (vbot-vtop)/(rbot-rtop) = (vbot-vtop)/dr
        pk_pvtop = -1.0/dr
        pk_pvbot = -pk_pvtop
        ##### dt = inv_k * np.log(vtop/vbot)
        pdt_pp    = 0.0
        pdt_pvtop = -pk_pvtop*inv_k * dt + inv_k / vtop
        pdt_pvbot = -pk_pvbot*inv_k * dt - inv_k / vbot
    else:
        #####
        dx = 0.
        pdx_pp = (1./rbot-1./rtop) / vbot
        pdx_pvbot = 0.
        pdx_pvtop = 0.
        #####
        dt = (rtop-rbot) / vtop 
        #####
        pdt_pp    = 0.0
        pdt_pvtop = 0.5*dr/(vtop*vtop)
        pdt_pvbot = pdt_pvtop
    return dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot

@jit_wrapper(nopython=True, nogil=True)
def theta_at_interface(p, r, v):
    sin_v = p*v/r
    cos_v = np.sqrt(1.0 - sin_v*sin_v)
    cot_v = cos_v/sin_v
    #f1_v  = F1_v2(sin_v, cos_v)
    f1_v  = np.log(sin_v/(1.0+cos_v))
    theta_v = np.arcsin(sin_v)
    return sin_v, cos_v, cot_v, f1_v, theta_v

@jit_wrapper(nopython=True, nogil=True)
def cut_and_interp_line(x, y, xstart, xend): # x must be increasing float, and y not-decreasing int
    """
    Given a line defined by (x, y), cut the line to the interval [xstart, xend], and
    the new line must start at xstart and end at xend, with linear interpolation if needed.
    """
    if xstart > xend or xstart > x[-1] or xend < x[0]:
        return x[:0], y[:0] # empty arrays
    xstart = max(xstart, x[0])
    xend   = min(xend,   x[-1])
    #####
    #if xstart == xend: # not needed
    #    y_value = np.interp(xstart, x, y)
    #    return np.array([xstart]), np.array([y_value])
    #####
    istart = np.searchsorted(x, xstart, side='left') # x[istart-1] <  xstart <= x[istart]
    iend   = np.searchsorted(x, xend, side='right')  # x[iend-1]   <= xend   <  x[iend]
    #####
    if xstart != x[istart]: # xstart < x[istart]
        #ystart = np.interp(xstart, x, y)
        ystart = y[istart] # the y is int, so use the value just after xstart
        if x[iend-1] != xend: #  x[iend-1] < xend < x[iend]
            #yend = np.interp(xend, x, y)
            yend = y[iend] # the y is int, so use the value just after xend
            new_x = x[istart-1:iend+1].copy()
            new_y = y[istart-1:iend+1].copy()
            new_x[0]   = xstart
            new_y[0]   = ystart
            new_x[-1]  = xend
            new_y[-1]  = yend
        else:
            new_x = x[istart-1:iend].copy()
            new_y = y[istart-1:iend].copy()
            new_x[0]   = xstart
            new_y[0]   = ystart
    else:
        if x[iend-1] != xend: #  x[iend-1] < xend < x[iend]
            #yend = np.interp(xend, x, y)
            yend = y[iend] # the y is int, so use the value just after xend
            new_x = x[istart:iend+1].copy()
            new_y = y[istart:iend+1].copy()
            new_x[-1]  = xend
            new_y[-1]  = yend
        else:
            new_x = x[istart:iend].copy()
            new_y = y[istart:iend].copy()
    return new_x, new_y
@jit_wrapper(nopython=True, nogil=True)
def common_x_two_lines(x1, y1, x2, y2): # x must be increasing float, and y not-decreasing int
    """
    Get two new lines as subsets of the input two lines, and also the two new lines
    have the common x intervals as the intersection of x1 and x2.
    """
    x_start = max(x1[0], x2[0])
    x_end   = min(x1[-1], x2[-1])
    if x_start <= x_end:
        new_x1, new_y1 = cut_and_interp_line(x1, y1, x_start, x_end)
        new_x2, new_y2 = cut_and_interp_line(x2, y2, x_start, x_end)
        ######
        common_x  = np.union1d(new_x1, new_x2)
        ind1 = np.searchsorted(new_x1, common_x, side='left')
        common_y1 = new_y1[ind1]
        ind2 = np.searchsorted(new_x2, common_x, side='left')
        common_y2 = new_y2[ind2]
        return common_x, common_y1, common_y2
    else:
        return x1[:0], y1[:0], y2[:0] # empty arrays
@jit_wrapper(nopython=True, nogil=True)
def common_x_two_segs(seg_x1, seg_y1, ibreaks1, seg_x2, seg_y2, ibreaks2):
    nmax = seg_x1.size + seg_x2.size
    common_seg_x    = np.zeros(nmax, dtype=seg_x1.dtype)
    common_seg_y1   = np.zeros(nmax, dtype=seg_y1.dtype)
    common_seg_y2   = np.zeros(nmax, dtype=seg_y2.dtype)
    common_ibreaks  = np.zeros(nmax+1, dtype=ibreaks1.dtype)
    ####
    nleg1 = ibreaks1.size-1
    nleg2 = ibreaks2.size-1
    nx = 0
    nseg = 1
    for ileg1 in range(nleg1):
        x1 = seg_x1[ibreaks1[ileg1]: ibreaks1[ileg1+1] ]
        y1 = seg_y1[ibreaks1[ileg1]: ibreaks1[ileg1+1] ]
        for ileg2 in range(nleg2):
            x2 = seg_x2[ibreaks2[ileg2]: ibreaks2[ileg2+1] ]
            y2 = seg_y2[ibreaks2[ileg2]: ibreaks2[ileg2+1] ]
            ####
            common_x, common_y1, common_y2 = common_x_two_lines(x1, y1, x2, y2)
            if common_x.size > 0:
                common_seg_x[nx: nx + common_x.size] = common_x
                common_seg_y1[nx: nx + common_x.size] = common_y1
                common_seg_y2[nx: nx + common_x.size] = common_y2
                nx += common_x.size
                #######
                common_ibreaks[nseg] = nx
                nseg += 1
    common_seg_x  = common_seg_x[:nx]
    common_seg_y1 = common_seg_y1[:nx]
    common_seg_y2 = common_seg_y2[:nx]
    common_ibreaks= common_ibreaks[:nseg]
    return common_seg_x, common_seg_y1, common_seg_y2, common_ibreaks

######################################################################################
# Low level functions for spherical taup calculations
@jit_wrapper(nopython=True, nogil=True)
def p2ib(p, r, vr):
    """
    Return the layer index in which the ray turns.
    #
    Note: the ith layer corresponds to (r[i], r[i+1]]. So each layer excludes its top interface
    but includes its bottom interface. Also, the surface belongs to the -1th layer.
    #
    p: ray parameter in s/rad
    r: radius array in km
    vr: velocity array in km/s
    """
    r_vr = r/vr
    if p >= r_vr[0]:
        return -1 # cannot penetrate any layer
    ib = r.size-1 # default is penetrating all layers
    for i in range(r.size-1):
        if r_vr[i] > p and p >= r_vr[i+1]:
            ib = i
            break
    return ib
@jit_wrapper(nopython=True, nogil=True)
def p2xt_all(p, r, vr, jac=True): # fundemental: compute x, t, ray path, and jacobian
    dr    = r[1:] - r[:-1]   #np.diff(r)
    dv    = vr[1:] - vr[:-1] #np.diff(vr)
    k     = np.zeros_like(dv)
    inv_k = np.zeros_like(dv)
    #np.divide(dv, dr, out=k,     where=(dr!=0))
    #np.divide(dr, dv, out=inv_k, where=(dv!=0))
    for idx in range(dr.size):
        if dr[idx]!=0.0:
            k[idx] = dv[idx]/dr[idx]
        if dv[idx]!=0.0:
            inv_k[idx] = dr[idx]/dv[idx]
    not_const_vel      = np.where( np.abs(k) > __zero_k_tol__ ,  True, False ) # boolean array for non-constant velocity layer
    not_zero_thickness = np.where( np.abs(dr)> __zero_dr_tol__,  True, False ) # boolean array for non-zero thickness layer
    ####################################################################################################################################################################
    ib = p2ib(p, r, vr)
    ####################################################################################################################################################################
    # The ray turns in layer ib, so that it penerate the layers 0,1,2,...,ib-1, and turn in the ibth layer, where the jth layer means (r[j], r[j+1]], ...
    # Specifically, the ray will penetrate layers (r0, r1], (r1, r2], ..., (r[ib-1], r[ib]], and turn in layer (r[ib], r[ib+1]].
    # There are three cases:
    # (1) ib = -1.            The ray turn at r0, which means x=0 and t=0.
    # (2) ib = r.size-1.      The ray penetrates all layers (r0, r1], (r1, r2], ..., (r[ib-1], r[ib]]
    # (3) 0 <= ib < r.size-1. The ray penetrates     layers (r0, r1], (r1, r2], ..., (r[ib-1], r[ib]], and turn in layer (r[ib], r[ib+1]].
    #                         In the turning layer, we can find out the critical radius and velocity, the the last layer:(r[ib], rc].
    ####################################################################################################################################################################
    # We also need to care about the special case p=0 (vertical ray going straightly down)
    # Let's say that we also find the ib for the ray. It will follow the same logic as (2) or (3) above.  The only difference is the way of calculating in each layer.
    ####################################################################################################################################################################
    # So, an alogrithm is:
    # if ib == -1:
    #     return 0.0, 0.0
    # if p != 0:
    #     for ilayer in 0,1,...,ib-1:
    #         dx, dt = penetrating layer(r[ilayer], r[ilayer+1])
    #     if ib != r.size-1:
    #         find rc
    #         dx, dt = penetrating layer(r[ilayer], rc)
    # else:
    #     for ilayer in 0,1,...,ib-1:
    #         dx, dt = special penetrating layer(r[ilayer], r[ilayer+1])
    #     if ib != r.size-1:
    #         find rc
    #         dx, dt = special penetrating layer(r[ilayer], rc)
    ####################################################################################################################################################################
    if ib == -1: # the ray cannot penetrate any layer
        zero_array = np.zeros(r.size, dtype=np.float64)
        return 0., 0., -1,  zero_array, zero_array, zero_array,    zero_array, 0., zero_array, 0., zero_array
    #############################################################################################
    x = 0.0
    t = 0.0
    ##### for seismic ray path
    npts = ib+1 if (ib==r.size-1) else ib+2 # number of ray path points
    # nlayers == npts-1
    path_dxs  = np.zeros(npts-1, dtype=np.float64)
    path_drs  = np.zeros(npts-1, dtype=np.float64)
    path_dts  = np.zeros(npts-1, dtype=np.float64)
    ##### for jacobian
    pdx_pp    = np.zeros(npts-1, dtype=np.float64) # par dx / par p at each layer/ray segment
    pdx_pvtop = np.zeros(npts-1, dtype=np.float64) # par dx / par vtop at each layer/ray segment
    pdx_pvbot = np.zeros(npts-1, dtype=np.float64) # par dx / par vbot at each layer/ray segment
    pdt_pp    = np.zeros(npts-1, dtype=np.float64) # par dt / par p at each layer/ray segment
    pdt_pvtop = np.zeros(npts-1, dtype=np.float64) # par dt / par vtop at each layer/ray segment
    pdt_pvbot = np.zeros(npts-1, dtype=np.float64) # par dt / par vbot at each layer/ray segment
    #############################################################################################
    if p > __zero_p_sph_tol__: # p != 0 # Note __zero_p_sph_tol__ is selected for Earth!!! Don't change it.
        # p=__zero_p_sph_tol__ correspond to theta = 0.0005 degree from surface with v=6km/s
        sin_top, cos_top, cot_top, f1_top, theta_top = theta_at_interface(p, r[0], vr[0])
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, r[ilyr+1], vr[ilyr+1])
            if not_zero_thickness[ilyr]:
                tmp = dxdt4plyr_jac(r[ilyr],   vr[ilyr],   sin_top, cos_top, cot_top, theta_top, f1_top,
                                    r[ilyr+1], vr[ilyr+1], sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                    p, dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                dx, dt, pdx_pp[ilyr], pdx_pvtop[ilyr], pdx_pvbot[ilyr], pdt_pp[ilyr], pdt_pvtop[ilyr], pdt_pvbot[ilyr] = tmp
                x += dx
                t += dt
                path_dxs[ilyr] = dx
                path_drs[ilyr] = dr[ilyr]
                path_dts[ilyr] = dt
            ######
            sin_top = sin_bot
            cos_top = cos_bot
            cot_top = cot_bot
            f1_top  = f1_bot
            theta_top = theta_bot
        #########################################################################################
        if ib != (r.size-1): # ilyar=ib. The layer of (r[ib], rc]
            if not_zero_thickness[ib]:
                rtop = r[ib]
                vtop = vr[ib]
                rbot = r[ib+1]
                # p = rc/vc = rc/(k*rc+b) <==> rc = b/(1/p-k) & vc=k*rc+b
                b = vtop - k[ib]*rtop
                rc = b/(1.0/p - k[ib])
                #vc = k[ib]*rc + b
                #sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, rc, vc)
                sin_bot = 1.0
                cos_bot = 0.0
                cot_bot = 0.0
                f1_bot  = 0.0 # np.log(sin_v/(1.0+cos_v))
                theta_bot = np.pi/2.0
                tmp = dxdt4clyr_jac(r[ib], vr[ib], sin_top, cos_top, cot_top, theta_top, f1_top,
                                    p, dr[ib], k[ib], inv_k[ib], not_const_vel[ib])
                dx, dt, pdx_pp[ib], pdx_pvtop[ib], pdx_pvbot[ib], pdt_pp[ib], pdt_pvtop[ib], pdt_pvbot[ib] = tmp
                x += dx
                t += dt
                ######
                # compute pdt_pvtop, pdt_pvbot, pdx_pvtop, pdx_pvbot, pdt_pp, pdx_pp here for each layer if needed
                ######
                path_dxs[ib] = dx
                path_drs[ib] = rc-rtop
                path_dts[ib] = dt
        #########################################################################################
    else: # p == 0
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            if not_zero_thickness[ilyr]:
                tmp = dxdt4lyr_zeroP_jac(r[ilyr], vr[ilyr], r[ilyr+1], vr[ilyr+1], dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                dx, dt, pdx_pp[ilyr], pdx_pvtop[ilyr], pdx_pvbot[ilyr], pdt_pp[ilyr], pdt_pvtop[ilyr], pdt_pvbot[ilyr] = tmp
                t += dt
                path_drs[ilyr] = dr[ilyr]
                path_dts[ilyr] = dt
        if ib != (r.size-1):
            if not_zero_thickness[ib]:
                rc = r[ib+1]  # Do not need to interpolate here.
                vc = vr[ib+1] # Just use the bottom interface of the ibth layer.
                tmp = dxdt4lyr_zeroP_jac(r[ib], vr[ib], rc, vc, dr[ib], k[ib], inv_k[ib], not_const_vel[ib])
                dx, dt, pdx_pp[ib], pdx_pvtop[ib], pdx_pvbot[ib], pdt_pp[ib], pdt_pvtop[ib], pdt_pvbot[ib] = tmp
                t += dt
                path_drs[ib] = dr[ib]
                path_dts[ib] = dt
    #############################################################################################
    if jac:
        pxpp = pdx_pp.sum()
        ptpp = pdt_pp.sum()
        pxpv = np.zeros(r.size, dtype=np.float64) # par x / par p
        pxpv[:pdx_pvtop.size]    += pdx_pvtop
        pxpv[1:pdx_pvbot.size+1] += pdx_pvbot
        ptpv = np.zeros(r.size, dtype=np.float64) # par t / par p
        ptpv[:pdt_pvtop.size]    += pdt_pvtop
        ptpv[1:pdt_pvbot.size+1] += pdt_pvbot
        #############
        # dt = ptpp * dp + ptpv * dv
        # dx = pxpp * dp + pxpv * dv
        # If we want dt/dv with dx==0, then have dx=0=pxpp*dp + pxpv*dv
        # ==> dp = -pxpv/pxpp * dv
        # ==> dt = ptpp*(-pxpv/pxpp) * dv + ptpv * dv
        #
        dtdv = ptpv.copy() # par t / par v
        if np.abs(pxpp) > 1e-5:
            dtdv -= ptpp*(pxpv/pxpp)
        #print()
        #print('pxpp=', pxpp)
        #print('pxpv=', pxpv)
        #print('ptpp=', ptpp)
        #print('ptpv=', ptpv)
        # x in rad
        # t in s
        # p in s/rad
        # v in km/s
        # pxpv is rad / (km/s)   = rad*s/km
        # pxpp is rad / (s/rad)  = rad*rad/s
        # ptpv is   s / (km/s)   = s*s/km
        # ptpp is   s / (s/rad)  = rad
        # pxpv/pxpp is   rad*s/km * s/(rad*rad) = s*s/(rad*km)
        # ptpp*(pxpv/pxpp) is rad*s*s/(rad*km) = s*s/km. This is the same as ptpv. So no problem!
    return x, t, ib, path_dxs, path_drs, path_dts, pxpv, pxpp, ptpv, ptpp, dtdv
@jit_wrapper(nopython=True, nogil=True)
def p2xt(p, r, vr, dr, dv, k, inv_k, not_const_vel, not_zero_thickness, ib=-2, debug=False):
    """
    ib: must be valid: 0<=ib<=z.size-1
    """
    if ib <0 or ib >= r.size-1:
        ib = p2ib(p, r, vr)
    if ib == -1: # the ray cannot penetrate any layer
        return 0., 0., -1
    #############################################################################################
    x = 0.0
    t = 0.0
    #############################################################################################
    if p > __zero_p_sph_tol__: # p != 0 # Note __zero_p_sph_tol__ is selected for Earth!!! Don't change it.
        # p=__zero_p_sph_tol__ correspond to theta = 0.0005 degree from surface with v=6km/s
        sin_top, cos_top, cot_top, f1_top, theta_top = theta_at_interface(p, r[0], vr[0])
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, r[ilyr+1], vr[ilyr+1])
            if not_zero_thickness[ilyr]:
                dx, dt = dxdt4lyr(r[ilyr],   vr[ilyr],   sin_top, cos_top, cot_top, theta_top, f1_top,
                                r[ilyr+1], vr[ilyr+1], sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                p, dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr], debug=debug)
                x += dx
                t += dt
            ######
            sin_top = sin_bot
            cos_top = cos_bot
            cot_top = cot_bot
            f1_top  = f1_bot
            theta_top = theta_bot
        #########################################################################################
        if ib != (r.size-1): # ilyar=ib. The layer of (r[ib], rc]
            if not_zero_thickness[ib]:
                rtop = r[ib]
                vtop = vr[ib]
                rbot = r[ib+1]
                #vbot = vr[ib+1]
                # p = rc/vc = rc/(k*rc+b) <==> rc = b/(1/p-k) & vc=k*rc+b
                b = vtop - k[ib]*rtop
                rc = b/(1.0/p - k[ib])
                vc = k[ib]*rc + b
                #sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, rc, vc)
                sin_bot = 1.0
                cos_bot = 0.0
                cot_bot = 0.0
                f1_bot  = 0.0 # np.log(sin_v/(1.0+cos_v))
                theta_bot = np.pi/2.0
                dx, dt = dxdt4lyr(r[ib], vr[ib], sin_top, cos_top, cot_top, theta_top, f1_top,
                                  rc,    vc,     sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                  p, rc-r[ib], k[ib], inv_k[ib], not_const_vel[ib], debug=debug)
                x += dx
                t += dt
        #########################################################################################
    else: # p == 0
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            if not_zero_thickness[ilyr]:
                dt = dxdt4lyr_zeroP(r[ilyr], vr[ilyr], r[ilyr+1], vr[ilyr+1], dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                t += dt
        if ib != (r.size-1):
            if not_zero_thickness[ib]:
                rc = r[ib+1]  # Do not need to interpolate here.
                vc = vr[ib+1] # Just use the bottom interface of the ibth layer.
                dt = dxdt4lyr_zeroP(r[ib], vr[ib], rc, vc, rc-r[ib], k[ib], inv_k[ib], not_const_vel[ib])
                t += dt
    #############################################################################################
    return x, t, ib
@jit_wrapper(nopython=True, nogil=True)
def p2xt_ray_path(p, r, vr, dr, dv, k, inv_k, not_const_vel, not_zero_thickness, ib=-2, debug=False):
    """
    ib: must be valid: 0<=ib<=z.size-1
    """
    if ib <0 or ib >= r.size-1:
        ib = p2ib(p, r, vr)
    if ib == -1: # the ray cannot penetrate any layer
        return 0., 0., -1
    #############################################################################################
    x = 0.0
    t = 0.0
    ##### for seismic ray path
    npts = ib+1 if (ib==r.size-1) else ib+2 # number of ray path points
    # nlayers == npts-1
    path_dxs  = np.zeros(npts-1, dtype=np.float64)
    path_drs  = np.zeros(npts-1, dtype=np.float64)
    path_dts  = np.zeros(npts-1, dtype=np.float64)
    #############################################################################################
    if p > __zero_p_sph_tol__: # p != 0 # Note __zero_p_sph_tol__ is selected for Earth!!! Don't change it.
        # p=__zero_p_sph_tol__ correspond to theta = 0.0005 degree from surface with v=6km/s
        sin_top, cos_top, cot_top, f1_top, theta_top = theta_at_interface(p, r[0], vr[0])
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, r[ilyr+1], vr[ilyr+1])
            if not_zero_thickness[ilyr]:
                dx, dt = dxdt4lyr(r[ilyr],   vr[ilyr],   sin_top, cos_top, cot_top, theta_top, f1_top,
                                  r[ilyr+1], vr[ilyr+1], sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                  p, dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr], debug=debug)
                x += dx
                t += dt
                path_dxs[ilyr] = dx
                path_drs[ilyr] = dr[ilyr]
                path_dts[ilyr] = dt
            ######
            sin_top = sin_bot
            cos_top = cos_bot
            cot_top = cot_bot
            f1_top  = f1_bot
            theta_top = theta_bot
        #########################################################################################
        if ib != (r.size-1): # ilyar=ib. The layer of (r[ib], rc]
            if not_zero_thickness[ib]:
                rtop = r[ib]
                vtop = vr[ib]
                rbot = r[ib+1]
                #vbot = vr[ib+1]
                # p = rc/vc = rc/(k*rc+b) <==> rc = b/(1/p-k) & vc=k*rc+b
                b = vtop - k[ib]*rtop
                rc = b/(1.0/p - k[ib])
                vc = k[ib]*rc + b
                #sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, rc, vc)
                sin_bot = 1.0
                cos_bot = 0.0
                cot_bot = 0.0
                f1_bot  = 0.0 # np.log(sin_v/(1.0+cos_v))
                theta_bot = np.pi/2.0
                dx, dt = dxdt4lyr(r[ib], vr[ib], sin_top, cos_top, cot_top, theta_top, f1_top,
                                  rc,    vc,     sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                  p, rc-r[ib], k[ib], inv_k[ib], not_const_vel[ib], debug=debug)
                x += dx
                t += dt
                path_dxs[ib] = dx
                path_drs[ib] = rc-rtop
                path_dts[ib] = dt
        #########################################################################################
    else: # p == 0
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            if not_zero_thickness[ilyr]:
                dt = dxdt4lyr_zeroP(r[ilyr], vr[ilyr], r[ilyr+1], vr[ilyr+1], dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                t += dt
                path_drs[ilyr] = dr[ilyr]
                path_dts[ilyr] = dt
        if ib != (r.size-1):
            if not_zero_thickness[ib]:
                rc = r[ib+1]  # Do not need to interpolate here.
                vc = vr[ib+1] # Just use the bottom interface of the ibth layer.
                dt = dxdt4lyr_zeroP(r[ib], vr[ib], rc, vc, rc-r[ib], k[ib], inv_k[ib], not_const_vel[ib])
                t += dt
                path_drs[ib] = dr[ib]
                path_dts[ib] = dt
    #############################################################################################
    return x, t, ib, path_dxs, path_drs, path_dts
@jit_wrapper(nopython=True, nogil=True)
def p2xt_jac(p, r, vr, dr, dv, k, inv_k, not_const_vel, not_zero_thickness, ib=-2): # fundemental: compute x, t, ray path, and jacobian
    ####################################################################################################################################################################
    if ib <0 or ib >= r.size-1:
        ib = p2ib(p, r, vr)
    ####################################################################################################################################################################
    if ib == -1: # the ray cannot penetrate any layer
        zero_array = np.zeros(r.size, dtype=np.float64)
        return 0., 0., -1,  zero_array, zero_array, zero_array,    zero_array, 0., zero_array, 0., zero_array
    #############################################################################################
    x = 0.0
    t = 0.0
    ##### for jacobian
    pxpp = 0.0
    ptpp = 0.0
    pxpv = np.zeros(r.size, dtype=np.float64) # par x / par p
    ptpv = np.zeros(r.size, dtype=np.float64) # par t / par p
    #############################################################################################
    if p > __zero_p_sph_tol__: # p != 0 # Note __zero_p_sph_tol__ is selected for Earth!!! Don't change it.
        # p=__zero_p_sph_tol__ correspond to theta = 0.0005 degree from surface with v=6km/s
        sin_top, cos_top, cot_top, f1_top, theta_top = theta_at_interface(p, r[0], vr[0])
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, r[ilyr+1], vr[ilyr+1])
            if not_zero_thickness[ilyr]:
                tmp = dxdt4plyr_jac(r[ilyr],   vr[ilyr],   sin_top, cos_top, cot_top, theta_top, f1_top,
                                    r[ilyr+1], vr[ilyr+1], sin_bot, cos_bot, cot_bot, theta_bot, f1_bot,
                                    p, dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                #dx, dt, pdx_pp[ilyr], pdx_pvtop[ilyr], pdx_pvbot[ilyr], pdt_pp[ilyr], pdt_pvtop[ilyr], pdt_pvbot[ilyr] = tmp
                dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot = tmp
                x += dx
                t += dt
                pxpp += pdx_pp
                ptpp += pdt_pp
                pxpv[ilyr]     += pdx_pvtop
                pxpv[ilyr+1]   += pdx_pvbot
                ptpv[ilyr]     += pdt_pvtop
                ptpv[ilyr+1]   += pdt_pvbot
            ######
            sin_top = sin_bot
            cos_top = cos_bot
            cot_top = cot_bot
            f1_top  = f1_bot
            theta_top = theta_bot
        #########################################################################################
        if ib != (r.size-1): # ilyar=ib. The layer of (r[ib], rc]
            if not_zero_thickness[ib]:
                rtop = r[ib]
                vtop = vr[ib]
                rbot = r[ib+1]
                # p = rc/vc = rc/(k*rc+b) <==> rc = b/(1/p-k) & vc=k*rc+b
                b = vtop - k[ib]*rtop
                rc = b/(1.0/p - k[ib])
                #vc = k[ib]*rc + b
                #sin_bot, cos_bot, cot_bot, f1_bot, theta_bot = theta_at_interface(p, rc, vc)
                sin_bot = 1.0
                cos_bot = 0.0
                cot_bot = 0.0
                f1_bot  = 0.0 # np.log(sin_v/(1.0+cos_v))
                theta_bot = np.pi/2.0
                tmp = dxdt4clyr_jac(r[ib], vr[ib], sin_top, cos_top, cot_top, theta_top, f1_top,
                                    p, dr[ib], k[ib], inv_k[ib], not_const_vel[ib])
                #dx, dt, pdx_pp[ib], pdx_pvtop[ib], pdx_pvbot[ib], pdt_pp[ib], pdt_pvtop[ib], pdt_pvbot[ib] = tmp
                dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot = tmp
                x += dx
                t += dt
                pxpp += pdx_pp
                ptpp += pdt_pp
                pxpv[ib]     += pdx_pvtop
                pxpv[ib+1]   += pdx_pvbot
                ptpv[ib]     += pdt_pvtop
                ptpv[ib+1]   += pdt_pvbot
                ######
                # compute pdt_pvtop, pdt_pvbot, pdx_pvtop, pdx_pvbot, pdt_pp, pdx_pp here for each layer if needed
                ######
        #########################################################################################
    else: # p == 0
        for ilyr in range(ib): # ilyr=0,1,2,...,ib-1
            if not_zero_thickness[ilyr]:
                tmp = dxdt4lyr_zeroP_jac(r[ilyr], vr[ilyr], r[ilyr+1], vr[ilyr+1], dr[ilyr], k[ilyr], inv_k[ilyr], not_const_vel[ilyr])
                dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot = tmp
                pxpp += pdx_pp
                pxpv[ilyr]     += pdx_pvtop
                pxpv[ilyr+1]   += pdx_pvbot
                ptpp += pdt_pp
                ptpv[ilyr]     += pdt_pvtop
                ptpv[ilyr+1]   += pdt_pvbot
                t += dt
        if ib != (r.size-1):
            if not_zero_thickness[ib]:
                rc = r[ib+1]  # Do not need to interpolate here.
                vc = vr[ib+1] # Just use the bottom interface of the ibth layer.
                tmp = dxdt4lyr_zeroP_jac(r[ib], vr[ib], rc, vc, dr[ib], k[ib], inv_k[ib], not_const_vel[ib])
                dx, dt, pdx_pp, pdx_pvtop, pdx_pvbot, pdt_pp, pdt_pvtop, pdt_pvbot = tmp
                pxpp += pdx_pp
                pxpv[ib]     += pdx_pvtop
                pxpv[ib+1]   += pdx_pvbot
                ptpp += pdt_pp
                ptpv[ib]     += pdt_pvtop
                ptpv[ib+1]   += pdt_pvbot
                t += dt
    #############################################################################################
    # dt = ptpp * dp + ptpv * dv
    # dx = pxpp * dp + pxpv * dv
    # If we want dt/dv with dx==0, then have dx=0=pxpp*dp + pxpv*dv
    # ==> dp = -pxpv/pxpp * dv
    # ==> dt = ptpp*(-pxpv/pxpp) * dv + ptpv * dv
    #
    dtdv = ptpv.copy() # par t / par v
    if np.abs(pxpp) > 1e-5:
        dtdv -= ptpp*(pxpv/pxpp)
    return x, t, ib, pxpv, pxpp, ptpv, ptpp, dtdv

def benchmark_p2xt():
    r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
    dr = np.diff(r)
    dv = np.diff(vp)
    k     = dv/dr 
    inv_k = 1.0/k
    not_const_vel      = np.where( np.abs(k) > __zero_k_tol__ , True, False )
    not_zero_thickness = np.where( np.abs(dr)> __zero_dr_tol__, True, False )
    x, t, ib = p2xt(392, r, vp, dr, dv, k, inv_k, not_const_vel, not_zero_thickness)
    print('x,t=', x, t)
    plt.plot(x, t, 'o-')
    plt.show()

# Low level functions for getting feasible ray parameters given a r-vr profile
@jit_wrapper(nopython=True, nogil=True)
def r2p_turn_back(vr_r):
    vz = vr_r
    ################################################################################
    # Use the code from z2inv_rp_turn_back(...)
    # step 1:
    vmax = vz[0]
    ind_mono_vzs = np.zeros(vz.size, dtype=np.int64)
    n = 0
    for iz in range(1, vz.size):
        if vmax < vz[iz]:
            ind_mono_vzs[n] = iz
            n += 1
            vmax = vz[iz]
    ind_mono_vzs = ind_mono_vzs[:n]
    ##if debug:
    ##    print('ind_mono_vzs=', ind_mono_vzs)
    # step 2:
    ind_zs = np.zeros(vz.size*2, dtype=np.int64)
    nzs = 1 # the Zero inserted at the start
    ibreaks = np.zeros(ind_zs.size+1, dtype=np.int64)
    nbreaks = 1
    for im in range(ind_mono_vzs.size-1):
        if (ind_mono_vzs[im+1]-ind_mono_vzs[im]) == 1:
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
        else:
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
            #
            ibreaks[nbreaks] = nzs
            nbreaks += 1
            #
            ind_zs[nzs] = ind_mono_vzs[im]
            nzs += 1
    ind_zs[nzs] = ind_mono_vzs[-1]
    nzs += 1
    ibreaks[nbreaks] = nzs
    nbreaks += 1
    ind_zs = ind_zs[:nzs]
    ibreaks = ibreaks[:nbreaks]
    # step 3
    inv_rps = vz[ind_zs] #np.zeros(ind_zs.size, dtype=np.float64)
    ibs     = ind_zs-1   #np.zeros(ind_zs.size, dtype=np.float64)
    for iseg in range(ibreaks.size-1):
        istart = ibreaks[iseg]
        #inv_rps[istart] = np.nextafter(inv_rps[istart], inv_rps[istart+1]) # inv_rps is mono increasing!
        inv_rps[istart] += __next_after_zero_p_sph__
        ibs[istart] = ibs[istart+1]
    ################################################################################
    ps = 1.0/inv_rps
    return ps, ibs, ibreaks
@jit_wrapper(nopython=True, nogil=True)
def r2p_penetrate(vr_r):
    qs = vr_r
    idx_max = np.argmax(qs)
    ps      = np.array((1.0/qs[idx_max], __zero_p_sph_tol__), dtype=np.float64) # __zero_p_sph_tol__ means vertical ray here
    ibs     = np.array((vr_r.size-2, vr_r.size-1), dtype=np.int64)
    ibreaks = np.array((0,2), dtype=np.int64)
    if idx_max != vr_r.size-1:
        #If the qmax is not at the bottom-most interface, need to increase the starting q a little bit.
        #qmax = np.nextafter(qmax, qmax+2.0)
        ps[0]  -= __next_after_zero_p_sph__
        ibs[0]  = vr_r.size-1
    return ps, ibs, ibreaks
@jit_wrapper(nopython=True, nogil=True)
def r2p_both(vr_r): # for both turn-back and penetrating rays
    ps1, ibs1, ibreaks1 = r2p_turn_back(vr_r)
    ps2, ibs2, ibreaks2 = r2p_penetrate(vr_r) # only two points and one segment
    ps      = np.zeros(ps1.size  + ps2.size, dtype=np.float64)
    ibs     = np.zeros(ibs1.size + ibs2.size,     dtype=np.int64)
    ibreaks = np.zeros(ibreaks1.size + 1, dtype=np.int64)
    ps[:-2]      = ps1
    ps[-2:]      = ps2
    ibs[:-2]     = ibs1
    ibs[-2:]     = ibs2
    ibreaks[:-1] = ibreaks1
    ibreaks[-1]  = ps.size
    ####
    return ps, ibs, ibreaks
@jit_wrapper(nopython=True, nogil=True)
def denser_p_interval(pmax, pmin, ib_min, ib_max, v0, R0, size, buf_ps, buf_ibs):
    ang0 = np.arcsin(pmax*v0/R0)
    ang1 = np.arcsin(pmin*v0/R0)
    buf_ps[:] = (R0/v0)*np.sin( np.linspace(ang0, ang1, size) )
    buf_ps[0] = pmax
    buf_ps[-1]= pmin
    buf_ibs[:] = ib_max
    buf_ibs[0] = ib_min
@jit_wrapper(nopython=True, nogil=True)
def denser_ps(ps, ibs, ibreaks, v0, R0, max_theta_step_rad=0.0017, max_denser_coef=-1):
    """
    Return a new ps, ibs, ibreaks with denser sampling of incidence angles.
    """
    nsegs   = ibreaks.size - 1
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    #####
    theta = np.arcsin((v0/R0)*ps)
    n_per_layer = np.zeros(theta.size-1, dtype=np.int64)
    nrp = 0
    for iseg in range(nsegs):
        i0 = istarts[iseg]
        i1 = iends[iseg]
        # the valid is inv_rps[i0:i1], or the index, i0,i0+1,...,i1-1
        for ilayer in range(i0, i1-1):
            a0 = theta[ilayer]
            a1 = theta[ilayer+1]
            local_n = int( np.abs(a1 - a0)/max_theta_step_rad ) + 2
            n_per_layer[ilayer] = local_n
            nrp += local_n-1
        nrp += 1 # the last point of this segment
    #####
    res_ps      = np.zeros(nrp, dtype=np.float64)
    res_ibs     = np.zeros(nrp, dtype=np.int64)
    res_ibreaks = np.zeros(ibreaks.size, dtype=np.int64) # same size as ibreaks
    global_idx = 0
    for iseg in range(nsegs):
        i0 = istarts[iseg]
        i1 = iends[iseg]
        # the valid is ps[i0:i1], or the index, i0,i0+1,...,i1-1
        for irp in range(i0, i1-1):
            pmax = ps[irp]
            pmin = ps[irp+1]
            ib_min     = ibs[irp]
            ib_max     = ibs[irp+1]
            local_n    = n_per_layer[irp]
            buf_ps     = res_ps[ global_idx:global_idx+local_n]
            buf_ibs    = res_ibs[global_idx:global_idx+local_n]
            denser_p_interval(pmax, pmin, ib_min, ib_max, v0, R0, local_n, buf_ps, buf_ibs)
            global_idx += local_n - 1 # well, the last point will be removed here, except for the last one of the segment
        ## add the last point of this segment
        res_ps[ global_idx] = ps[ i1-1]
        res_ibs[global_idx] = ibs[i1-1]
        global_idx += 1
        res_ibreaks[iseg+1] = global_idx
    return res_ps, res_ibs, res_ibreaks

def benchmark_r2p():
    r, v = oc_inv.rd_ak135_OC_model()
    v[10] -= 3.0
    #r = r[20:40].copy()
    #v = v[20:40].copy()
    ps, ibs, ibreaks = r2p_both(r, v)
    print(np.diff(ps))
    ##########
    r_v = r/v
    ##########
    fig, (ax1) = plt.subplots(1,1, figsize=(6,8))
    ax1.plot(r_v, r, color='k')
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    nseg = istarts.size
    for iseg, (i0, i1) in enumerate(zip(istarts, iends)):
        ps_seg = ps[i0:i1]
        ibs_seg= ibs[i0:i1]
        if iseg!=nseg-1:
            r_seg  = r[ibs_seg]
            r_seg  = r[ibs_seg+1]
            ax1.plot(ps_seg, r_seg, marker='s', label=f'seg{iseg}')
    ##########
    ps, ibs, ibreaks = denser_ps(ps, ibs, ibreaks, v0=v[0], R0=r[0])
    print(np.diff(ps))
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    nseg = istarts.size
    for iseg, (i0, i1) in enumerate(zip(istarts, iends)):
        ps_seg = ps[i0:i1]
        ibs_seg= ibs[i0:i1]
        if iseg!=nseg-1:
            r_seg  = r[ibs_seg]
            r_seg  = r[ibs_seg+1]
            ax1.plot(ps_seg, r_seg, marker='.', label=f'seg{iseg}')
    plt.show()

######################################################################################
# Middle level functions for spherical taup calculations
def assemble_path(phase_name, phase_type, PSK0IJ_path_dxs, PSK0IJ_path_drs, PSK0IJ_path_dts, R0):
    ##################################################################
    P_dxs, S_dxs, K_dxs, _, I_dxs, J_dxs = PSK0IJ_path_dxs
    P_drs, S_drs, K_drs, _, I_drs, J_drs = PSK0IJ_path_drs
    P_dts, S_dts, K_dts, _, I_dts, J_dts = PSK0IJ_path_dts
    ##################################################################
    # Note, the drs are all negative!
    path_dxs, path_drs, path_dts = [0.], [R0,], [0.]
    if phase_type == 0: # e.g., PKIKP,  PKJKP,  PKiKP
        p_downgoing = True
        s_downgoing = True
        k_downgoing = True
        for c in phase_name:
            if c == 'P':
                if p_downgoing:
                    path_dxs.extend(P_dxs)
                    path_drs.extend(P_drs)
                    path_dts.extend(P_dts)
                    p_downgoing = False
                else:
                    path_dxs.extend(P_dxs[::-1])
                    path_drs.extend(P_drs[::-1]*-1.)
                    path_dts.extend(P_dts[::-1])
                    p_downgoing = True
            elif c == 'S':
                if s_downgoing:
                    path_dxs.extend(S_dxs)
                    path_drs.extend(S_drs)
                    path_dts.extend(S_dts)
                    s_downgoing = False
                else:
                    path_dxs.extend(S_dxs[::-1])
                    path_drs.extend(S_drs[::-1]*-1.)
                    path_dts.extend(S_dts[::-1])
                    s_downgoing = True
            elif c == 'K':
                if k_downgoing:
                    path_dxs.extend(K_dxs)
                    path_drs.extend(K_drs)
                    path_dts.extend(K_dts)
                    k_downgoing = False
                else:
                    path_dxs.extend(K_dxs[::-1])
                    path_drs.extend(K_drs[::-1]*-1.)
                    path_dts.extend(K_dts[::-1])
                    k_downgoing = True
            elif c == 'I':
                path_dxs.extend(I_dxs)
                path_drs.extend(I_drs)
                path_dts.extend(I_dts)
                #
                path_dxs.extend(I_dxs[::-1])
                path_drs.extend(I_drs[::-1]*-1.)
                path_dts.extend(I_dts[::-1])
            elif c == 'J':
                path_dxs.extend(J_dxs)
                path_drs.extend(J_drs)
                path_dts.extend(J_dts)
                #
                path_dxs.extend(J_dxs[::-1])
                path_drs.extend(J_drs[::-1]*-1.)
                path_dts.extend(J_dts[::-1])
            elif c in 'ic':
                pass
            else:
                # wrong phase name
                pass
    elif phase_type == 1: # e.g., PKP, PcP
        p_downgoing = True
        s_downgoing = True
        for c in phase_name:
            if c == 'P':
                if p_downgoing:
                    path_dxs.extend(P_dxs)
                    path_drs.extend(P_drs)
                    path_dts.extend(P_dts)
                    p_downgoing = False
                else:
                    path_dxs.extend(P_dxs[::-1])
                    path_drs.extend(P_drs[::-1]*-1.)
                    path_dts.extend(P_dts[::-1])
                    p_downgoing = True
            elif c == 'S':
                if s_downgoing:
                    path_dxs.extend(S_dxs)
                    path_drs.extend(S_drs)
                    path_dts.extend(S_dts)
                    s_downgoing = False
                else:
                    path_dxs.extend(S_dxs[::-1])
                    path_drs.extend(S_drs[::-1]*-1.)
                    path_dts.extend(S_dts[::-1])
                    s_downgoing = True
            elif c == 'K':
                path_dxs.extend(K_dxs)
                path_drs.extend(K_drs)
                path_dts.extend(K_dts)
                #
                path_dxs.extend(K_dxs[::-1])
                path_drs.extend(K_drs[::-1]*-1.)
                path_dts.extend(K_dts[::-1])
            elif c in 'c':
                pass
            else:
                # wrong phase name
                pass    
    elif phase_type == 2: # e.g., P, S
        for c in phase_name:
            if c == 'P':
                path_dxs.extend(P_dxs)
                path_drs.extend(P_drs)
                path_dts.extend(P_dts)
                #
                path_dxs.extend(P_dxs[::-1])
                path_drs.extend(P_drs[::-1]*-1.)
                path_dts.extend(P_dts[::-1])
            elif c == 'S':
                path_dxs.extend(S_dxs)
                path_drs.extend(S_drs)
                path_dts.extend(S_dts)
                #
                path_dxs.extend(S_dxs[::-1])
                path_drs.extend(S_drs[::-1]*-1.)
                path_dts.extend(S_dts[::-1])
            else:
                # wrong phase name
                pass    
    ###################################################################
    path_xs = np.cumsum(path_dxs)
    path_rs = np.cumsum(path_drs)
    path_ts = np.cumsum(path_dts)
    return path_xs, path_rs, path_ts
@jit_wrapper(nopython=True, nogil=True)
def phase_ps2xt(ps, nP, nS, nK, nI, nJ,  
                mt_r, mt_vp, mt_vs, mt_dr, mt_dvp, mt_dvs, mt_kp, mt_inv_kp, mt_ks, mt_inv_ks, mt_ncvp, mt_ncvs, mt_nzt,
                oc_r, oc_vp, oc_vs, oc_dr, oc_dvp, oc_dvs, oc_kp, oc_inv_kp, oc_ks, oc_inv_ks, oc_ncvp, oc_ncvs, oc_nzt,
                ic_r, ic_vp, ic_vs, ic_dr, ic_dvp, ic_dvs, ic_kp, ic_inv_kp, ic_ks, ic_inv_ks, ic_ncvp, ic_ncvs, ic_nzt):
    lst_r  = [mt_r,      mt_r,      oc_r,      oc_r,      ic_r,      ic_r]
    lst_dr = [mt_dr,     mt_dr,     oc_dr,     oc_dr,     ic_dr,     ic_dr]
    lst_v  = [mt_vp,     mt_vs,     oc_vp,     oc_vs,     ic_vp,     ic_vs]
    lst_dv = [mt_dvp,    mt_dvs,    oc_dvp,    oc_dvs,    ic_dvp,    ic_dvs]
    lst_k  = [mt_kp,     mt_ks,     oc_kp,     oc_ks,     ic_kp,     ic_ks]
    lst_ik = [mt_inv_kp, mt_inv_ks, oc_inv_kp, oc_inv_ks, ic_inv_kp, ic_inv_ks]
    lst_ncv= [mt_ncvp,   mt_ncvs,   oc_ncvp,   oc_ncvs,   ic_ncvp,   ic_ncvs]
    lst_nzt= [mt_nzt,    mt_nzt,    oc_nzt,    oc_nzt,    ic_nzt,    ic_nzt]
    lst_n  = [nP,        nS,        nK,        0,         nI,        nJ]
    xs = np.zeros(ps.size, dtype=np.float64)
    ts = np.zeros(ps.size, dtype=np.float64)
    ibs= np.zeros(ps.size, dtype=np.int64)
    for ip in range(ps.size):
        x, t, ib = 0.0, 0.0, -1
        for idx in range(6):
            n = lst_n[idx]
            if n > 0:
                r = lst_r[idx]
                dr= lst_dr[idx]
                v = lst_v[idx]
                dv= lst_dv[idx]
                k = lst_k[idx]
                ik= lst_ik[idx]
                ncv=lst_ncv[idx]
                nzt=lst_nzt[idx]
                dx, dt, tmp_ib = p2xt(ps[ip], r, v, dr, dv, k, ik, ncv, nzt)
                x += (dx*n)
                t += (dt*n)
                ib = tmp_ib if tmp_ib>ib else ib
        if ps[ip]<__zero_p_sph_tol__: # vertical down-going ray. Adjust the distance within the IC for this.
            x += (nI+nJ)*0.5*np.pi
        xs[ip] = x
        ts[ip] = t
        ibs[ip]= ib
    return xs, ts, ibs
@jit_wrapper(nopython=True, nogil=True)
def phase_ps2xt_jac(ps, nP, nS, nK, nI, nJ,  
                    mt_r, mt_vp, mt_vs, mt_dr, mt_dvp, mt_dvs, mt_kp, mt_inv_kp, mt_ks, mt_inv_ks, mt_ncvp, mt_ncvs, mt_nzt,
                    oc_r, oc_vp, oc_vs, oc_dr, oc_dvp, oc_dvs, oc_kp, oc_inv_kp, oc_ks, oc_inv_ks, oc_ncvp, oc_ncvs, oc_nzt,
                    ic_r, ic_vp, ic_vs, ic_dr, ic_dvp, ic_dvs, ic_kp, ic_inv_kp, ic_ks, ic_inv_ks, ic_ncvp, ic_ncvs, ic_nzt):
    #############
    nr = (mt_r.size + oc_r.size + ic_r.size)
    pxpvp = np.zeros((ps.size, nr), dtype=np.float64)
    pxpvs = np.zeros((ps.size, nr), dtype=np.float64)
    pxpp  = np.zeros(ps.size, dtype=np.float64)
    ptpvp = np.zeros((ps.size, nr), dtype=np.float64)
    ptpvs = np.zeros((ps.size, nr), dtype=np.float64)
    ptpp  = np.zeros(ps.size, dtype=np.float64)
    dtdvp = np.zeros((ps.size, nr), dtype=np.float64)
    dtdvs = np.zeros((ps.size, nr), dtype=np.float64)
    #############
    lst_r  = [mt_r,      mt_r,      oc_r,      oc_r,      ic_r,      ic_r]
    lst_dr = [mt_dr,     mt_dr,     oc_dr,     oc_dr,     ic_dr,     ic_dr]
    lst_v  = [mt_vp,     mt_vs,     oc_vp,     oc_vs,     ic_vp,     ic_vs]
    lst_dv = [mt_dvp,    mt_dvs,    oc_dvp,    oc_dvs,    ic_dvp,    ic_dvs]
    lst_k  = [mt_kp,     mt_ks,     oc_kp,     oc_ks,     ic_kp,     ic_ks]
    lst_ik = [mt_inv_kp, mt_inv_ks, oc_inv_kp, oc_inv_ks, ic_inv_kp, ic_inv_ks]
    lst_ncv= [mt_ncvp,   mt_ncvs,   oc_ncvp,   oc_ncvs,   ic_ncvp,   ic_ncvs]
    lst_nzt= [mt_nzt,    mt_nzt,    oc_nzt,    oc_nzt,    ic_nzt,    ic_nzt]
    lst_n  = [nP,        nS,        nK,        0,         nI,        nJ]
    #############
    icmb = mt_r.size
    iicb = icmb + oc_r.size
    px_p_mtvp = pxpvp[:, :icmb]
    px_p_mtvs = pxpvs[:, :icmb]
    pt_p_mtvp = ptpvp[:, :icmb]
    pt_p_mtvs = ptpvs[:, :icmb]
    dt_d_mtvp = dtdvp[:, :icmb]
    dt_d_mtvs = dtdvs[:, :icmb]
    #
    px_p_ocvp = pxpvp[:, icmb:iicb]
    px_p_ocvs = pxpvs[:, icmb:iicb]
    pt_p_ocvp = ptpvp[:, icmb:iicb]
    pt_p_ocvs = ptpvs[:, icmb:iicb]
    dt_d_ocvp = dtdvp[:, icmb:iicb]
    dt_d_ocvs = dtdvs[:, icmb:iicb]
    #
    px_p_icvp = pxpvp[:, iicb:]
    px_p_icvs = pxpvs[:, iicb:]
    pt_p_icvp = ptpvp[:, iicb:]
    pt_p_icvs = ptpvs[:, iicb:]
    dt_d_icvp = dtdvp[:, iicb:]
    dt_d_icvs = dtdvs[:, iicb:]
    #
    lst_pxpv = [px_p_mtvp, px_p_mtvs, px_p_ocvp, px_p_ocvs, px_p_icvp, px_p_icvs]
    lst_ptpv = [pt_p_mtvp, pt_p_mtvs, pt_p_ocvp, pt_p_ocvs, pt_p_icvp, pt_p_icvs]
    lst_dtdv = [dt_d_mtvp, dt_d_mtvs, dt_d_ocvp, dt_d_ocvs, dt_d_icvp, dt_d_icvs]
    #############
    xs = np.zeros(ps.size, dtype=np.float64)
    ts = np.zeros(ps.size, dtype=np.float64)
    ibs= np.zeros(ps.size, dtype=np.int64)
    for ip in range(ps.size):
        if np.isnan(ps[ip]):
            continue
        x, t, ib = 0.0, 0.0, -1
        for idx in range(6):
            n = lst_n[idx]
            if n > 0:
                r = lst_r[idx]
                dr= lst_dr[idx]
                v = lst_v[idx]
                dv= lst_dv[idx]
                k = lst_k[idx]
                ik= lst_ik[idx]
                ncv=lst_ncv[idx]
                nzt=lst_nzt[idx]
                #dx, dt, tmp_ib = p2xt(ps[ip], r, v, dr, dv, k, ik)
                tmp = p2xt_jac(ps[ip], r, v, dr, dv, k, ik, ncv, nzt)
                dx, dt, tmp_ib, it_pxpv, it_pxpp, it_ptpv, it_ptpp, it_dtdv = tmp
                x += (dx*n)
                t += (dt*n)
                #
                lst_pxpv[idx][ip] += it_pxpv * n
                pxpp[ip]          += it_pxpp * n
                lst_ptpv[idx][ip] += it_ptpv * n
                ptpp[ip]          += it_ptpp * n
                lst_dtdv[idx][ip] += it_dtdv * n
                ib = tmp_ib if tmp_ib>ib else ib
        if ps[ip]<__zero_p_sph_tol__: # vertical down-going ray. Adjust the distance within the IC for this.
            x += (nI+nJ)*0.5*np.pi
        xs[ip] = x
        ts[ip] = t
        ibs[ip]= ib
    return xs, ts, ibs, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs

def phase_ps2xt_ray_path(ps, phase_name, R0,
                         mt_r, mt_vp, mt_vs, mt_dr, mt_dvp, mt_dvs, mt_kp, mt_inv_kp, mt_ks, mt_inv_ks, mt_ncvp, mt_ncvs, mt_nzt,
                         oc_r, oc_vp, oc_vs, oc_dr, oc_dvp, oc_dvs, oc_kp, oc_inv_kp, oc_ks, oc_inv_ks, oc_ncvp, oc_ncvs, oc_nzt,
                         ic_r, ic_vp, ic_vs, ic_dr, ic_dvp, ic_dvs, ic_kp, ic_inv_kp, ic_ks, ic_inv_ks, ic_ncvp, ic_ncvs, ic_nzt):
    ###################################################################
    phase_type, nP, nS, nK, nI, nJ = decipher_phase_name(phase_name)
    ###################################################################
    lst_r  = [mt_r,      mt_r,      oc_r,      oc_r,      ic_r,      ic_r]
    lst_dr = [mt_dr,     mt_dr,     oc_dr,     oc_dr,     ic_dr,     ic_dr]
    lst_v  = [mt_vp,     mt_vs,     oc_vp,     oc_vs,     ic_vp,     ic_vs]
    lst_dv = [mt_dvp,    mt_dvs,    oc_dvp,    oc_dvs,    ic_dvp,    ic_dvs]
    lst_k  = [mt_kp,     mt_ks,     oc_kp,     oc_ks,     ic_kp,     ic_ks]
    lst_ik = [mt_inv_kp, mt_inv_ks, oc_inv_kp, oc_inv_ks, ic_inv_kp, ic_inv_ks]
    lst_ncv= [mt_ncvp,   mt_ncvs,   oc_ncvp,   oc_ncvs,   ic_ncvp,   ic_ncvs]
    lst_nzt= [mt_nzt,    mt_nzt,    oc_nzt,    oc_nzt,    ic_nzt,    ic_nzt]
    lst_n  = [nP,        nS,        nK,        0,         nI,        nJ]
    xs = np.zeros(ps.size, dtype=np.float64)
    ts = np.zeros(ps.size, dtype=np.float64)
    ibs= np.zeros(ps.size, dtype=np.int64)
    lst_path = [None for it in range(ps.size)]
    for ip in range(ps.size):
        if np.isnan(ps[ip]):
            continue
        ##################################################################################
        lst_path_seg_dxs  = [None,      None,       None,     None,      None,      None]
        lst_path_seg_drs  = [None,      None,       None,     None,      None,      None]
        lst_path_seg_dts  = [None,      None,       None,     None,      None,      None]
        ##################################################################################
        x, t, ib = 0.0, 0.0, -1
        for idx in range(6):
            n = lst_n[idx]
            if n > 0:
                r = lst_r[idx]
                dr= lst_dr[idx]
                v = lst_v[idx]
                dv= lst_dv[idx]
                k = lst_k[idx]
                ik= lst_ik[idx]
                ncv=lst_ncv[idx]
                nzt=lst_nzt[idx]
                tmp  = p2xt_ray_path(ps[ip], r, v, dr, dv, k, ik, ncv, nzt)
                dx, dt, tmp_ib, path_seg_dxs, path_seg_drs, path_seg_dts = tmp
                x += (dx*n)
                t += (dt*n)
                ib = tmp_ib if tmp_ib>ib else ib
                ##################################################################################
                lst_path_seg_dxs[idx] = path_seg_dxs
                lst_path_seg_drs[idx] = path_seg_drs
                lst_path_seg_dts[idx] = path_seg_dts
                ##################################################################################
        if ps[ip]<__zero_p_sph_tol__: # vertical down-going ray. Adjust the distance within the IC for this.
            x += (nI+nJ)*0.5*np.pi
        xs[ip] = x
        ts[ip] = t
        ibs[ip]= ib
        ##################################################################################
        path_xs, path_rs, path_ts = assemble_path(phase_name, phase_type, lst_path_seg_dxs, lst_path_seg_drs, lst_path_seg_dts, R0)
        lst_path[ip] = (path_xs, path_rs, path_ts)
        ##################################################################################
    return xs, ts, ibs, lst_path
@jit_wrapper(nopython=True, nogil=True)
def phase_x2pt(target_xs, nP, nS, nK, nI, nJ,
               ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks,
               mt_r, mt_vp, mt_vs, mt_dr, mt_dvp, mt_dvs, mt_kp, mt_inv_kp, mt_ks, mt_inv_ks, mt_ncvp, mt_ncvs, mt_nzt,
               oc_r, oc_vp, oc_vs, oc_dr, oc_dvp, oc_dvs, oc_kp, oc_inv_kp, oc_ks, oc_inv_ks, oc_ncvp, oc_ncvs, oc_nzt,
               ic_r, ic_vp, ic_vs, ic_dr, ic_dvp, ic_dvs, ic_kp, ic_inv_kp, ic_ks, ic_inv_ks, ic_ncvp, ic_ncvs, ic_nzt,
               fast_arrival=False,
               niter=1000, xerr=1e-6, debug=False):
    ##############################################################################################################################
    if fast_arrival:
        res =phase_x2pt(target_xs, nP, nS, nK, nI, nJ,
                        ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks,
                        mt_r, mt_vp, mt_vs, mt_dr, mt_dvp, mt_dvs, mt_kp, mt_inv_kp, mt_ks, mt_inv_ks, mt_ncvp, mt_ncvs, mt_nzt,
                        oc_r, oc_vp, oc_vs, oc_dr, oc_dvp, oc_dvs, oc_kp, oc_inv_kp, oc_ks, oc_inv_ks, oc_ncvp, oc_ncvs, oc_nzt,
                        ic_r, ic_vp, ic_vs, ic_dr, ic_dvp, ic_dvs, ic_kp, ic_inv_kp, ic_ks, ic_inv_ks, ic_ncvp, ic_ncvs, ic_nzt,
                        fast_arrival=False,
                        niter=niter, xerr=xerr, debug=False)
        p_found, x_found, t_found, ib_found, ix_found = res
        p_uniq = np.zeros(target_xs.size, dtype=np.float64)
        x_uniq = np.zeros(target_xs.size, dtype=np.float64)
        t_uniq = np.zeros(target_xs.size, dtype=np.float64)+1e100
        ib_uniq= np.zeros(target_xs.size, dtype=np.int64)
        ix_uniq= np.zeros(target_xs.size, dtype=np.int64)-1 # default -1 meaning invalid
        for idx in range(p_found.size):
            ix = ix_found[idx]
            if t_found[idx] < t_uniq[ix]:
                p_uniq[ix]  = p_found[idx]
                x_uniq[ix]  = x_found[idx]
                t_uniq[ix]  = t_found[idx]
                ib_uniq[ix] = ib_found[idx]
                ix_uniq[ix] = ix
        for ix in range(target_xs.size):
            if ix_uniq[ix] < 0:
                p_uniq[ix]  = np.nan
                x_uniq[ix]  = target_xs[ix]
                t_uniq[ix]  = np.nan
                ib_uniq[ix] = -2
        return p_uniq, x_uniq, t_uniq, ib_uniq, ix_uniq
    ##############################################################################################################################
    lst_r  = [mt_r,      mt_r,      oc_r,      oc_r,      ic_r,      ic_r]
    lst_dr = [mt_dr,     mt_dr,     oc_dr,     oc_dr,     ic_dr,     ic_dr]
    lst_v  = [mt_vp,     mt_vs,     oc_vp,     oc_vs,     ic_vp,     ic_vs]
    lst_dv = [mt_dvp,    mt_dvs,    oc_dvp,    oc_dvs,    ic_dvp,    ic_dvs]
    lst_k  = [mt_kp,     mt_ks,     oc_kp,     oc_ks,     ic_kp,     ic_ks]
    lst_ik = [mt_inv_kp, mt_inv_ks, oc_inv_kp, oc_inv_ks, ic_inv_kp, ic_inv_ks]
    lst_ncv= [mt_ncvp,   mt_ncvs,   oc_ncvp,   oc_ncvs,   ic_ncvp,   ic_ncvs]
    lst_nzt= [mt_nzt,    mt_nzt,    oc_nzt,    oc_nzt,    ic_nzt,    ic_nzt]
    lst_n  = [nP,        nS,        nK,        0,         nI,        nJ]
    #######
    p_found = np.zeros(target_xs.size*16, dtype=np.float64)
    x_found = np.zeros(target_xs.size*16, dtype=np.float64)
    t_found = np.zeros(target_xs.size*16, dtype=np.float64)
    ib_found= np.zeros(target_xs.size*16, dtype=np.int64)
    ix_found= np.zeros(target_xs.size*16, dtype=np.int64)
    nfound = 0
    for ix in range(target_xs.size):
        single_x = target_xs[ix]
        nseg = ds_ibks.size - 1
        for iseg in range(nseg):
            i0 = ds_ibks[iseg]
            i1 = ds_ibks[iseg+1]
            local_ps = ds_ps[i0:i1]
            local_xs = ds_xs[i0:i1]
            local_ts = ds_ts[i0:i1]
            local_ibs= ds_ibs[i0:i1]
            nintervals = local_ps.size-1
            for idx in range(nintervals):
                if np.abs(local_xs[idx]-single_x) < xerr: # start of the interval
                    p_found[nfound]  = local_ps[idx]
                    x_found[nfound]  = local_xs[idx]
                    t_found[nfound]  = local_ts[idx]
                    ib_found[nfound] = local_ibs[idx]
                    ix_found[nfound] = ix
                    nfound += 1
                    continue
                if np.abs(local_xs[idx+1]-single_x) < xerr: # end of the interval
                    p_found[nfound]  = local_ps[idx+1]
                    x_found[nfound]  = local_xs[idx+1]
                    t_found[nfound]  = local_ts[idx+1]
                    ib_found[nfound] = local_ibs[idx+1]
                    ix_found[nfound] = ix
                    nfound += 1
                    continue
                if (local_xs[idx] < single_x < local_xs[idx+1]) or (local_xs[idx] > single_x > local_xs[idx+1]): # within the interval
                    p_left = local_ps[idx]
                    p_right= local_ps[idx+1]
                    x_left = local_xs[idx]
                    #x_right= local_xs[idx+1]
                    #t_left = local_ts[idx]
                    #t_right= local_ts[idx+1]
                    for iter in range(niter):
                        p_mid = 0.5*(p_left + p_right)
                        ############################################################################################
                        x_mid, t_mid, ib_mid = 0.0, 0.0, -1
                        for idx in range(6):
                            n = lst_n[idx]
                            if n > 0:
                                r = lst_r[idx]
                                dr= lst_dr[idx]
                                v = lst_v[idx]
                                dv= lst_dv[idx]
                                k = lst_k[idx]
                                ik= lst_ik[idx]
                                ncv=lst_ncv[idx]
                                nzt=lst_nzt[idx]
                                dx, dt, tmp_ib = p2xt(p_mid, r, v, dr, dv, k, ik, ncv, nzt)
                                x_mid += (dx*n)
                                t_mid += (dt*n)
                                ib_mid = tmp_ib if tmp_ib>ib_mid else ib_mid
                        if p_mid<__zero_p_sph_tol__: # vertical down-going ray. Adjust the distance within the IC for this.
                            x_mid += (nI+nJ)*0.5*np.pi
                        ############################################################################################
                        if np.abs(x_mid - single_x) < xerr:
                            break
                        if (x_left <= single_x <= x_mid) or (x_left >= single_x >= x_mid):
                            p_right = p_mid
                            #x_right = x_mid
                            #t_right = t_mid
                        else:
                            p_left  = p_mid
                            x_left  = x_mid
                            #t_left  = t_mid
                    p_found[nfound]  = p_mid
                    x_found[nfound]  = x_mid
                    t_found[nfound]  = t_mid
                    ib_found[nfound] = ib_mid
                    ix_found[nfound] = ix
                    nfound += 1
            #####
    p_found = p_found[:nfound]
    x_found = x_found[:nfound]
    t_found = t_found[:nfound]
    ib_found= ib_found[:nfound]
    ix_found= ix_found[:nfound]
    ####
    # debug:
    if debug:
        for p in p_found:
            for idx in range(6):
                n = lst_n[idx]
                if n > 0:
                    r = lst_r[idx]
                    dr= lst_dr[idx]
                    v = lst_v[idx]
                    dv= lst_dv[idx]
                    k = lst_k[idx]
                    ik= lst_ik[idx]
                    ncv=lst_ncv[idx]
                    nzt=lst_nzt[idx]
                    dx, dt, tmp_ib = p2xt(p_mid, r, v, dr, dv, k, ik, ncv, nzt, debug=debug)
    ####
    return p_found, x_found, t_found, ib_found, ix_found

# Middle level functions for phase name deciphering
@jit_wrapper(nopython=True, nogil=True)
def decipher_phase_name(phase_name):
    nP, nS, nK, nI, nJ = 0, 0, 0, 0, 0
    if ('I' in phase_name) or ('J' in phase_name) or ('i' in phase_name):
        # e.g., PKIKP,  PKJKP,  PKiKP
        phase_type = 0
        nI = phase_name.count('I')*2 # Note, the p2xt contains both downgoing and upgoing paths.
        nJ = phase_name.count('J')*2
        nK = phase_name.count('K')
        nP = phase_name.count('P')
        nS = phase_name.count('S')
    elif ('K' in phase_name) or ('c' in phase_name):
        # e.g., PKP, PcP, PcS, ScS, SKS, SKP
        phase_type = 1
        nK = phase_name.count('K')*2
        nP = phase_name.count('P')
        nS = phase_name.count('S')
    else:
        # e.g., P, PP, S, SS, PS
        phase_type = 2
        nP = phase_name.count('P')*2
        nS = phase_name.count('S')*2
    return phase_type, nP, nS, nK, nI, nJ

class SphFastTaup:
    MT_PHASE = 2
    OC_PHASE = 1
    IC_PHASE = 0
    def __init__(self, r=None, vp=None, vs=None, icmb=None, iicb=None, R0=6371.0):
        if (r is None) or (vp is None) or (vs is None) or (icmb is None) or (iicb is None):
            r, vp, vs, icmb, iicb = oc_inv.rd_ak135_OC_model()
        self.init_model(r, vp, vs, icmb, iicb, R0)
    def init_model(self, r, vp, vs, icmb, iicb, R0=6371.0, zero_k_tol=__zero_k_tol__, zero_dr_tol=__zero_dr_tol__):
        self.R0 = R0
        self.r  = r
        self.vp = vp
        self.vs = vs
        self.icmb = icmb
        self.iicb = iicb
        ####
        self.mt_r,  self.oc_r,  self.ic_r  = r[ :icmb], r[ icmb:iicb], r[ iicb:]
        self.mt_vp, self.oc_vp, self.ic_vp = vp[:icmb], vp[icmb:iicb], vp[iicb:]
        self.mt_vs, self.oc_vs, self.ic_vs = vs[:icmb], vs[icmb:iicb], vs[iicb:]
        ####
        self.mt_dr = np.diff(self.mt_r)
        self.oc_dr = np.diff(self.oc_r)
        self.ic_dr = np.diff(self.ic_r)
        ####
        self.mt_dvp = np.diff(self.mt_vp)
        self.mt_dvs = np.diff(self.mt_vs)
        self.oc_dvp = np.diff(self.oc_vp)
        self.oc_dvs = np.diff(self.oc_vs)
        self.ic_dvp = np.diff(self.ic_vp)
        self.ic_dvs = np.diff(self.ic_vs)
        ####
        self.mt_kp = np.divide(self.mt_dvp, self.mt_dr, out=np.zeros_like(self.mt_dvp), where=self.mt_dr!=0)
        self.mt_ks = np.divide(self.mt_dvs, self.mt_dr, out=np.zeros_like(self.mt_dvs), where=self.mt_dr!=0)
        self.oc_kp = np.divide(self.oc_dvp, self.oc_dr, out=np.zeros_like(self.oc_dvp), where=self.oc_dr!=0)
        self.oc_ks = np.divide(self.oc_dvs, self.oc_dr, out=np.zeros_like(self.oc_dvs), where=self.oc_dr!=0)
        self.ic_kp = np.divide(self.ic_dvp, self.ic_dr, out=np.zeros_like(self.ic_dvp), where=self.ic_dr!=0)
        self.ic_ks = np.divide(self.ic_dvs, self.ic_dr, out=np.zeros_like(self.ic_dvs), where=self.ic_dr!=0)
        ####
        self.mt_inv_kp = np.divide(self.mt_dr, self.mt_dvp, out=np.zeros_like(self.mt_dvp), where=self.mt_dvp!=0)
        self.mt_inv_ks = np.divide(self.mt_dr, self.mt_dvs, out=np.zeros_like(self.mt_dvs), where=self.mt_dvs!=0)
        self.oc_inv_kp = np.divide(self.oc_dr, self.oc_dvp, out=np.zeros_like(self.oc_dvp), where=self.oc_dvp!=0)
        self.oc_inv_ks = np.divide(self.oc_dr, self.oc_dvs, out=np.zeros_like(self.oc_dvs), where=self.oc_dvs!=0)
        self.ic_inv_kp = np.divide(self.ic_dr, self.ic_dvp, out=np.zeros_like(self.ic_dvp), where=self.ic_dvp!=0)
        self.ic_inv_ks = np.divide(self.ic_dr, self.ic_dvs, out=np.zeros_like(self.ic_dvs), where=self.ic_dvs!=0)
        #### whether a non-const velocity layer for each layer. True for non-const velocity layer
        self.mt_ncvp = np.where( np.abs(self.mt_kp) > zero_k_tol , True, False )
        self.mt_ncvs = np.where( np.abs(self.mt_ks) > zero_k_tol , True, False )
        self.oc_ncvp = np.where( np.abs(self.oc_kp) > zero_k_tol , True, False )
        self.oc_ncvs = np.where( np.abs(self.oc_ks) > zero_k_tol , True, False )
        self.ic_ncvp = np.where( np.abs(self.ic_kp) > zero_k_tol , True, False )
        self.ic_ncvs = np.where( np.abs(self.ic_ks) > zero_k_tol , True, False )
        #### whether a non-zero thickness layer for each layer. True for non-zero thickness layer
        self.mt_nzt  = np.where( np.abs(self.mt_dr) > zero_dr_tol, True, False )
        self.oc_nzt  = np.where( np.abs(self.oc_dr) > zero_dr_tol, True, False )
        self.ic_nzt  = np.where( np.abs(self.ic_dr) > zero_dr_tol, True, False )
    ######################################################################################
    def get_trvt_curves(self, phase_name, max_theta_step_rad=0.0017):
        phase_type, nP, nS, nK, nI, nJ = decipher_phase_name(phase_name)
        res = self.rp4phase(phase_name, max_theta_step_rad=max_theta_step_rad)
        p   = res[0]
        ibk = res[-1]
        x, t, ib = phase_ps2xt(p, nP, nS, nK, nI, nJ,  
                                            self.mt_r, self.mt_vp, self.mt_vs, self.mt_dr, self.mt_dvp, self.mt_dvs, self.mt_kp, self.mt_inv_kp, self.mt_ks, self.mt_inv_ks, self.mt_ncvp, self.mt_ncvs, self.mt_nzt,
                                            self.oc_r, self.oc_vp, self.oc_vs, self.oc_dr, self.oc_dvp, self.oc_dvs, self.oc_kp, self.oc_inv_kp, self.oc_ks, self.oc_inv_ks, self.oc_ncvp, self.oc_ncvs, self.oc_nzt,
                                            self.ic_r, self.ic_vp, self.ic_vs, self.ic_dr, self.ic_dvp, self.ic_dvs, self.ic_kp, self.ic_inv_kp, self.ic_ks, self.ic_inv_ks, self.ic_ncvp, self.ic_ncvs, self.ic_nzt)
        return p, x, t, ib, ibk
    def get_trvts(self, phase_name, target_xs, niter=1000, xerr=1e-6, fast_arrival=False, max_theta_step_rad=0.0017, jac=False):
        phase_type, nP, nS, nK, nI, nJ = decipher_phase_name(phase_name)
        ds_p, ds_x, ds_t, ds_ib, ds_ibk = self.get_trvt_curves(phase_name, max_theta_step_rad=max_theta_step_rad)
        ##############
        xmin, xmax = ds_x.min(), ds_x.max()
        ncycle_min, ncycle_max = xmin//(np.pi*2)-2, xmax//(np.pi*2)+3
        new_target_xs = list()
        for x in np.unique(target_xs % (np.pi*2)):
            for ncycle in range(int(ncycle_min), int(ncycle_max)+1):
                new_target_xs.append( x + ncycle * (np.pi*2) )
                new_target_xs.append(-x + ncycle * (np.pi*2) )
        new_target_xs = np.array(new_target_xs, dtype=np.float64)
        idxs          = np.where((new_target_xs>=xmin) & (new_target_xs<=xmax))[0]
        new_target_xs = new_target_xs[idxs]
        new_target_xs = np.unique(new_target_xs)
        ##############
        res = phase_x2pt(new_target_xs, nP, nS, nK, nI, nJ,
                         ds_p, ds_x, ds_t, ds_ib, ds_ibk,
                         self.mt_r, self.mt_vp, self.mt_vs, self.mt_dr, self.mt_dvp, self.mt_dvs, self.mt_kp, self.mt_inv_kp, self.mt_ks, self.mt_inv_ks, self.mt_ncvp, self.mt_ncvs, self.mt_nzt,
                         self.oc_r, self.oc_vp, self.oc_vs, self.oc_dr, self.oc_dvp, self.oc_dvs, self.oc_kp, self.oc_inv_kp, self.oc_ks, self.oc_inv_ks, self.oc_ncvp, self.oc_ncvs, self.oc_nzt,
                         self.ic_r, self.ic_vp, self.ic_vs, self.ic_dr, self.ic_dvp, self.ic_dvs, self.ic_kp, self.ic_inv_kp, self.ic_ks, self.ic_inv_ks, self.ic_ncvp, self.ic_ncvs, self.ic_nzt,
                         fast_arrival=fast_arrival,
                         niter=niter, xerr=xerr)
        p_found, x_found, t_found, ib_found = res[:4]
        if jac:
            res = phase_ps2xt_jac(p_found, nP, nS, nK, nI, nJ,  
                                  self.mt_r, self.mt_vp, self.mt_vs, self.mt_dr, self.mt_dvp, self.mt_dvs, self.mt_kp, self.mt_inv_kp, self.mt_ks, self.mt_inv_ks, self.mt_ncvp, self.mt_ncvs, self.mt_nzt,
                                  self.oc_r, self.oc_vp, self.oc_vs, self.oc_dr, self.oc_dvp, self.oc_dvs, self.oc_kp, self.oc_inv_kp, self.oc_ks, self.oc_inv_ks, self.oc_ncvp, self.oc_ncvs, self.oc_nzt,
                                  self.ic_r, self.ic_vp, self.ic_vs, self.ic_dr, self.ic_dvp, self.ic_dvs, self.ic_kp, self.ic_inv_kp, self.ic_ks, self.ic_inv_ks, self.ic_ncvp, self.ic_ncvs, self.ic_nzt)
            junk_xs, junk_ts, junk_ibs, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs = res
            return p_found, x_found, t_found, ib_found, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs
        return p_found, x_found, t_found, ib_found

        ###########
    def get_paths(self, phase_name, target_xs, niter=1000, xerr=1e-6, max_theta_step_rad=0.0017):
        phase_type, nP, nS, nK, nI, nJ = decipher_phase_name(phase_name)
        ds_p, ds_x, ds_t, ds_ib, ds_ibk = self.get_trvt_curves(phase_name, max_theta_step_rad=max_theta_step_rad)
        ##############
        xmin, xmax = ds_x.min(), ds_x.max()
        ncycle_min, ncycle_max = xmin//(np.pi*2)-2, xmax//(np.pi*2)+3
        new_target_xs = list()
        for x in np.unique(target_xs % (np.pi*2)):
            for ncycle in range(int(ncycle_min), int(ncycle_max)+1):
                new_target_xs.append( x + ncycle * (np.pi*2) )
                new_target_xs.append(-x + ncycle * (np.pi*2) )
        new_target_xs = np.array(new_target_xs, dtype=np.float64)
        idxs          = np.where((new_target_xs>=xmin) & (new_target_xs<=xmax))[0]
        new_target_xs = new_target_xs[idxs]
        new_target_xs = np.unique(new_target_xs)
        ##############
        res = phase_x2pt(new_target_xs, nP, nS, nK, nI, nJ,
                         ds_p, ds_x, ds_t, ds_ib, ds_ibk,
                         self.mt_r, self.mt_vp, self.mt_vs, self.mt_dr, self.mt_dvp, self.mt_dvs, self.mt_kp, self.mt_inv_kp, self.mt_ks, self.mt_inv_ks, self.mt_ncvp, self.mt_ncvs, self.mt_nzt,
                         self.oc_r, self.oc_vp, self.oc_vs, self.oc_dr, self.oc_dvp, self.oc_dvs, self.oc_kp, self.oc_inv_kp, self.oc_ks, self.oc_inv_ks, self.oc_ncvp, self.oc_ncvs, self.oc_nzt,
                         self.ic_r, self.ic_vp, self.ic_vs, self.ic_dr, self.ic_dvp, self.ic_dvs, self.ic_kp, self.ic_inv_kp, self.ic_ks, self.ic_inv_ks, self.ic_ncvp, self.ic_ncvs, self.ic_nzt,
                         fast_arrival=False,
                         niter=niter, xerr=xerr)
        p_found, x_found, t_found, ib_found = res[:4]
        ##############
        res = phase_ps2xt_ray_path(p_found, phase_name, self.R0,
                                   self.mt_r, self.mt_vp, self.mt_vs, self.mt_dr, self.mt_dvp, self.mt_dvs, self.mt_kp, self.mt_inv_kp, self.mt_ks, self.mt_inv_ks, self.mt_ncvp, self.mt_ncvs, self.mt_nzt,
                                   self.oc_r, self.oc_vp, self.oc_vs, self.oc_dr, self.oc_dvp, self.oc_dvs, self.oc_kp, self.oc_inv_kp, self.oc_ks, self.oc_inv_ks, self.oc_ncvp, self.oc_ncvs, self.oc_nzt,
                                   self.ic_r, self.ic_vp, self.ic_vs, self.ic_dr, self.ic_dvp, self.ic_dvs, self.ic_kp, self.ic_inv_kp, self.ic_ks, self.ic_inv_ks, self.ic_ncvp, self.ic_ncvs, self.ic_nzt)
        junk_xs, junk_ts, junk_ibs, lst_path = res
        ##############
        return p_found, x_found, t_found, ib_found, lst_path
    ######################################################################################
    def plot_model(self, ax, mode='vp'):
        r_cmb = self.r[self.icmb]
        r_icb = self.r[self.iicb]
        if mode == None:
            ax.set_yticks( (0.0, r_icb, r_cmb, self.R0) )
            ax.set_yticklabels( [] )
        elif mode == 'core':
            circle = pl.Circle((0, 0), self.R0, transform=ax.transData._b, color='white', linewidth=0, alpha=0.2, zorder=0)
            ax.add_artist(circle)
            circle = pl.Circle((0, 0), r_cmb, transform=ax.transData._b, color='gray', linewidth=0, alpha=0.2, zorder=0)
            ax.add_artist(circle)
            circle = pl.Circle((0, 0), r_icb, transform=ax.transData._b, color='white', linewidth=0, alpha=1, zorder=0)
            ax.add_artist(circle)
            ax.set_yticks([])
        elif mode in ('vp', 'vs'):
            if mode == 'vp':
                vel = self.vp
            else:
                vel = self.vs
            rs = self.r
            thetas =  np.radians(np.linspace(0, 360,1440) )
            ys, xs = np.meshgrid(rs, thetas)
            values = np.zeros((thetas.size, rs.size) )
            for idx, v in enumerate(vel):
                values[:,idx] = v
            vmin, vmax = vel.min(), vel.max()
            dv = (vmax-vmin)*0.01
            vmin = vmin-(vmax-vmin)*0.6
            clev = np.arange(vmin, vmax, dv)
            ax.contourf(xs, ys, values, clev, cmap='gray', zorder=0)
            ax.set_yticks([])
        ####
        ax.set_xticks([])
        ax.set_ylim((0, self.R0))
    @staticmethod
    def benchmark_get_path():
        r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        #r = np.array((6371.,   4000, 4000, 1000., 1000.,  1e-3  ))
        #vp= np.array((6.,      10,   10,   12,    12,     13    ))
        #vs= np.array((3.5,     5.0,  5.0,  7.,    7.,     8.5   ))
        #icmb = 2
        #iicb = 4
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        target_xs = np.deg2rad(np.linspace(160, 200, 1) )
        phase_name = 'PPP'
        print(phase_name, target_xs)
        #######
        res = app.get_trvts(phase_name, target_xs, xerr=1e-16, niter=1000)
        p_found, x_found, t_found, ib_found = res
        print('pxt=', p_found, x_found, t_found, ib_found)
        #######
        res = app.get_paths(phase_name, target_xs, xerr=1e-16, niter=1000)
        p_found, x_found, t_found, ib_found, lst_path = res
        #######
        fig, ax = plt.subplots(1,1, figsize=(6,6), subplot_kw={'projection': 'polar'})
        for ip in range(p_found.size):
            if lst_path[ip]:
                path_xs, path_rs, path_ts = lst_path[ip]
                inv = 1. if (path_xs[-1] % (2.0*np.pi)) <= np.pi else -1.
                ax.plot(inv*path_xs, path_rs, '-', label=f'ip={ip}, p={p_found[ip]:.3f}, t={t_found[ip]:.3f}')
        app.plot_model(ax)
        plt.show()
        pass
    @staticmethod
    def benchmark_one_c():
        ####################################################################################################################################
        if False:
            vleft, vright = 5, 15
            niter= 1000
            for iv in range(niter):
                vmid = 0.5*(vleft + vright)
                cs   = [0.0, 0.0, 0.0]
                for ivc, vc in enumerate( (vleft, vmid, vright) ):
                    r = np.array((5000., 4000  ))
                    vp= np.array((6.,   vc,      ))
                    vs= np.array((3.5,  5.0,    ))
                    icmb = 2
                    iicb = 3
                    app = SphFastTaup(r, vp, vs, icmb, iicb)
                    target_xs = np.deg2rad(np.linspace(10, 170, 1) )
                    phase_name = 'PcP'
                    #######
                    res = app.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000, jac=True)
                    p_found, x_found, t_found, ib_found, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs = res
                    np.set_printoptions(precision=3, suppress=True)
                    k = np.diff(vp)/np.diff(r)
                    cs[ivc] = p_found*k
                if cs[0] <= -1 <= cs[1] or cs[0] >= -1 >= cs[1]:
                    vright = vmid
                else:
                    vleft  = vmid
            print('vmid=', vmid)
        ####################################################################################################################################
        r = np.array((5000., 4000  ))
        vp= np.array((6.,   11.626377213223641,      ))
        vs= np.array((3.5,  5.0,    ))
        icmb = 2
        iicb = 3
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        target_xs = np.deg2rad(np.linspace(10, 170, 1) )
        phase_name = 'PcP'
        #######
        res = app.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000, jac=True)
        p_found, x_found, t_found, ib_found, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs = res
        np.set_printoptions(precision=3, suppress=True)
        k = np.diff(vp)/np.diff(r)
        print('k=', k[0])
        print('p*k=', p_found[0]*k[0])
        print(np.abs(p_found[0]*k[0]+1.0) < __zero_k_mul_p_sph_tol__)
        print('dtdvp=', dtdvp.size, dtdvp)
        #######
        derr = 1e-1
        dtdvp2 = np.zeros_like(dtdvp)
        for iv in range(r.size):
            #print()
            #print('iv=', iv)
            vp_left = vp.copy()
            vp_right= vp.copy()
            vp_left[iv]  -= derr
            vp_right[iv] += derr
            app_left  = SphFastTaup(r, vp_left, vs, icmb, iicb)
            app_right = SphFastTaup(r, vp_right, vs, icmb, iicb)
            #print('For left')
            tmp = app_left.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000)
            x_left = tmp[1]
            t_left = tmp[2]
            #print('For right')
            tmp = app_right.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000)
            x_right= tmp[1]
            t_right= tmp[2]
            dtdvp2[:,iv] = (t_right - t_left)/(2.0*derr)
            #print(f'iv={iv}, tleft={t_left}, tright={t_right}')
            #print('xleft= ', x_left, t_left)
            #print('xright=', x_right, t_right)
        #####
        np.set_printoptions(precision=3, suppress=True)
        print('dtdvp2=', dtdvp2.size, dtdvp2)
        print('dtdvp =', dtdvp.size,  dtdvp)
    @staticmethod
    def benchmark_jac():
        #r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        #r[-1] = 1e-10 # avoid zero radius at the center
        #r = r[ :10]
        #vp= vp[:10]
        #vs= vs[:10]
        ##vp += np.random.random(vp.size)*1e-3
        #print('r=', r)
        #print('v=', vp)
        #icmb=10
        #iicb=10
        #r = np.array((5000, 4500, 4500, 4200, 4000, 4000, 3000, 3000, 1000))
        #vp= np.array((6.,   6,    10.,  10.,  12.,  9,    12,   13,   14))
        #vs= np.array((3.5,  5.0,  5.5,  5.5,  6.,  5,   7,    8,    9))
        #icmb = 2
        #iicb = 7
        r = np.array((5000., 4000  ))
        vp= np.array((6.,   13,      ))
        vs= np.array((3.5,  5.0,    ))
        icmb = 2
        iicb = 3
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        target_xs = np.deg2rad(np.linspace(0, 170, 1) )
        phase_name = 'PcP'
        print(phase_name, target_xs)
        #######
        res = app.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000, jac=True)
        p_found, x_found, t_found, ib_found, pxpvp, pxpp, ptpvp, ptpp, dtdvp, dtdvs = res
        np.set_printoptions(precision=10, suppress=True)
        k = np.diff(vp)/np.diff(r)
        print('x_found=', p_found, 1./p_found, x_found, t_found, ib_found)
        print('k=', k)
        print('p*k=', p_found*k)
        print('dtdvp=', dtdvp.size, dtdvp)
        #return
        #print()
        #return
        #######
        derr = 1e-5
        dtdvp2 = np.zeros_like(dtdvp)
        for iv in range(r.size):
            #print()
            #print('iv=', iv)
            vp_left = vp.copy()
            vp_right= vp.copy()
            vp_left[iv]  -= derr
            vp_right[iv] += derr
            app_left  = SphFastTaup(r, vp_left, vs, icmb, iicb)
            app_right = SphFastTaup(r, vp_right, vs, icmb, iicb)
            #print('For left')
            tmp = app_left.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000)
            x_left = tmp[1]
            t_left = tmp[2]
            #print('For right')
            tmp = app_right.get_trvts(phase_name, target_xs, fast_arrival=True, xerr=1e-16, niter=1000)
            x_right= tmp[1]
            t_right= tmp[2]
            dtdvp2[:,iv] = (t_right - t_left)/(2.0*derr)
            #print(f'iv={iv}, tleft={t_left}, tright={t_right}')
            #print('xleft= ', x_left, t_left)
            #print('xright=', x_right, t_right)
        #####
        np.set_printoptions(precision=10, suppress=True)
        print('dtdvp2=', dtdvp2.size, dtdvp2)
        print('dtdvp =', dtdvp.size,  dtdvp)



    @staticmethod
    def benchmark_trvts():
        r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        r[-1] = 1e-10 # avoid zero radius at the center
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        target_xs = np.linspace(1.0, np.pi, 50)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))
        phases = ['P', 'S', 'PS', 'PcP', 'ScS', 'PcS']
        phases = ['K', 'PKP', 'PKS', 'SKS', 'SKSScS' ]
        phases = ['PKP'] #, 'PKIKS', 'SKIKS', 'PKiKP', 'PKJKP', 'SKJKS', 'SKJKP']
        for phase in phases:
            p, x, t, ib, ibk = app.get_trvt_curves(phase, max_theta_step_rad=0.001)
            x = np.rad2deg(x)
            nseg = ibk.size - 1
            for iseg in range(nseg):
                i0 = ibk[iseg]
                i1 = ibk[iseg+1]
                ax1.plot(x[i0:i1], t[i0:i1], '-', label=phase)
                ax2.plot(x[i0:i1], p[i0:i1], '-', label=phase)
                ax3.plot(t[i0:i1], p[i0:i1], '-', label=phase)
            #######
            res = app.get_trvts(phase, target_xs, fast_arrival=True)
            p, x, t, ib = res[:4]
            print('p=', p)
            print('x=', x)
            print('t=', t)
            x = np.rad2deg(x)
            ax1.plot(x, t, '.', label=phase+' found')
            ax2.plot(x, p, '.', label=phase+' found')
            ax3.plot(t, p, '.', label=phase+' found')
        ax1.legend()
        #ax2.legend()
        #ax3.legend()
        plt.show()
    @staticmethod
    def benchmark_trvt_curves():
        r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        r[-1] = 1e-10 # avoid zero radius at the center
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,6))
        phases = ['P', 'S', 'PS', 'PcP', 'ScS', 'PcS']
        phases = ['K', 'PKP', 'PKS', 'SKS', 'SKSScS' ]
        phases = ['PKIKP', 'PKIKS', 'SKIKS', 'PKiKP', 'PKJKP', 'SKJKS', 'SKJKP']
        for phase in phases:
            p, x, t, ib, ibk = app.get_trvt_curves(phase, max_theta_step_rad=0.005)
            x = np.rad2deg(x)
            nseg = ibk.size - 1
            for iseg in range(nseg):
                i0 = ibk[iseg]
                i1 = ibk[iseg+1]
                ax1.plot(x[i0:i1], t[i0:i1], '.-', label=phase)
                ax2.plot(x[i0:i1], p[i0:i1], '.-', label=phase)
                ax3.plot(t[i0:i1], p[i0:i1], '.-', label=phase)
        ax1.legend()
        ax2.legend()
        ax3.legend()
        plt.show()
    ######################################################################################
    def rp4P(self, max_theta_step_rad=0.0017):
        vz = self.mt_vp / self.mt_r
        p, ib, ibk = r2p_turn_back(vz)
        return denser_ps(p, ib, ibk, v0=self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad) 
    def rp4S(self, max_theta_step_rad=0.0017):
        vz = self.mt_vs / self.mt_r
        p, ib, ibk = r2p_turn_back(vz)
        return denser_ps(p, ib, ibk, v0=self.vs[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
    def rp4PS(self, max_theta_step_rad=0.0017):
        p1, ib1, ibk1 = self.rp4P(max_theta_step_rad=max_theta_step_rad)
        p2, ib2, ibk2 = self.rp4S(max_theta_step_rad=max_theta_step_rad)
        q1 = 1.0/p1
        q2 = 1.0/p2
        q, ib_P, ib_S, ibk = common_x_two_segs(q1, ib1, ibk1, q2, ib2, ibk2)
        return 1.0/q, ib_P, ib_S, ibk
    def rp4PcP(self, max_theta_step_rad=0.0017):
        vz = self.mt_vp / self.mt_r
        p, ib, ibk = r2p_penetrate(vz)
        return denser_ps(p, ib, ibk, v0=self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
    def rp4ScS(self, max_theta_step_rad=0.0017):
        vz = self.mt_vs / self.mt_r
        p, ib, ibk = r2p_penetrate(vz)
        return denser_ps(p, ib, ibk, v0=self.vs[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
    def rp4PcS(self, max_theta_step_rad=0.0017):
        return self.rp4PcP(max_theta_step_rad=max_theta_step_rad)
    def rp4K(self, mantle_P=True, mantle_S=True, max_theta_step_rad=0.0017):
        vz = self.oc_vp / self.oc_r
        ######
        # Make sure the ray can cross the CMB if there must be some mantle legs.
        vz0 = vz[0]
        if mantle_P:
            tmp = np.max(self.mt_vp/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if mantle_S:
            tmp = np.max(self.mt_vs/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        vz[0] = vz0
        ######
        p, ib, ibk = r2p_turn_back(vz)
        ib += self.icmb
        result = denser_ps(p, ib, ibk, v0=self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
        return result
    def rp4KiK(self, mantle_P=True, mantle_S=True, max_theta_step_rad=0.0017): # For KiK(SKiKS-ScS), PKiKP, SKiS, SKiP,...
        vz = self.oc_vp / self.oc_r
        ######
        # Make sure the ray can cross the CMB if there must be some mantle legs.
        vz0 = vz[0]
        if mantle_P:
            tmp = np.max(self.mt_vp/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if mantle_S:
            tmp = np.max(self.mt_vs/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        vz[0] = vz0
        ######
        p, ib, ibk = r2p_penetrate(vz)
        ib += self.icmb
        result = denser_ps(p, ib, ibk, self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
        return result
    def rp4I(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For PKIKP, PKIKS, SKIKS, KIK, I,...
        vz = self.ic_vp / self.ic_r
        ######
        # Make sure the ray can cross the CMB and ICB if there must be some
        vz0 = vz[0]
        if mantle_P:
            tmp = np.max(self.mt_vp/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if mantle_S:
            tmp = np.max(self.mt_vs/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if oc_K:
            tmp = np.max(self.oc_vp/self.oc_r)
            vz0 = tmp if vz0 < tmp else vz0
        vz[0] = vz0
        ######
        p, ib, ibk = r2p_turn_back(vz)
        ib += self.iicb
        result = denser_ps(p, ib, ibk, self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
        return result
    def rp4J(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For PKJKP, PKJKS, SKJKS, KJK, J,...
        vz = self.ic_vs / self.ic_r
        ######
        # Make sure the ray can cross the CMB and ICB if there must be some
        vz0 = vz[0]
        if mantle_P:
            tmp = np.max(self.mt_vp/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if mantle_S:
            tmp = np.max(self.mt_vs/self.mt_r)
            vz0 = tmp if vz0 < tmp else vz0
        if oc_K:
            tmp = np.max(self.oc_vp/self.oc_r)
            vz0 = tmp if vz0 < tmp else vz0
        vz[0] = vz0
        ######
        p, ib, ibk = r2p_turn_back(vz)
        ib += self.iicb
        result = denser_ps(p, ib, ibk, self.vp[0], R0=self.R0, max_theta_step_rad=max_theta_step_rad)
        return result
    def rp4IJ(self, mantle_P=True, mantle_S=True, oc_K=True, max_theta_step_rad=0.0017): # For JKIKJ, PKIKS, JKIKP,...
        p1, ib1, ibk1 = self.rp4I(mantle_P=mantle_P, mantle_S=mantle_S, oc_K=oc_K, max_theta_step_rad=max_theta_step_rad)
        p2, ib2, ibk2 = self.rp4J(mantle_P=mantle_P, mantle_S=mantle_S, oc_K=oc_K, max_theta_step_rad=max_theta_step_rad)
        q1 = 1.0/p1
        q2 = 1.0/p2
        q, ibP, ibS, ibk = common_x_two_segs(q1, ib1, ibk1, q2, ib2, ibk2)
        p = 1.0/q
        return p, ibP, ibS, ibk    
    def rp4phase(self, phase_name, max_theta_step_rad=0.0017):
        phase_type, nP, nS, nK, nI, nJ = decipher_phase_name(phase_name)
        if phase_type == SphFastTaup.MT_PHASE:
            if nP>0 and nS==0:
                return self.rp4P(max_theta_step_rad=max_theta_step_rad)
            elif nP==0 and nS>0:
                return self.rp4S(max_theta_step_rad=max_theta_step_rad)
            else:
                return self.rp4PS(max_theta_step_rad=max_theta_step_rad)
        elif phase_type == SphFastTaup.OC_PHASE:
            if nK==0:
                if nP>0 and nS==0:
                    return self.rp4PcP(max_theta_step_rad=max_theta_step_rad)
                elif nP==0 and nS>0:
                    return self.rp4ScS(max_theta_step_rad=max_theta_step_rad)
                else:
                    return self.rp4PcS(max_theta_step_rad=max_theta_step_rad)
            else:
                return self.rp4K(mantle_P=(nP>0), mantle_S=(nS>0), max_theta_step_rad=max_theta_step_rad)
        else: # IC_PHASE
            if nI==0 and nJ==0:
                return self.rp4KiK(mantle_P=(nP>0), mantle_S=(nS>0), max_theta_step_rad=max_theta_step_rad)
            elif nI>0 and nJ==0:
                return self.rp4I(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
            elif nI==0 and nJ>0:
                return self.rp4J(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
            else:
                return self.rp4IJ(mantle_P=(nP>0), mantle_S=(nS>0), oc_K=(nK>0), max_theta_step_rad=max_theta_step_rad)
    @staticmethod
    def benchmark_rp4legs():
        r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        r[-1] = 1e-10 # avoid zero radius at the center
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        fig, (ax1) = plt.subplots(1,1, figsize=(6,8))
        r_v = app.r/app.vp
        ax1.plot(r_v, r, 's-', color='k', zorder=0)
        r_v = app.r/app.vs
        ax1.plot(r_v, r, 's-', color='gray', zorder=0)
        phases = 'P', 'S', 'PS', 'PcP', 'ScS', 'PcS', 'K', 'PKP', 'PKS', 'SKS', 'PKiKP', 'PKiKS', 'SKiKS', 'PKIKP', 'PKIKS', 'SKIKS', 'PKIJKP', 'PKIJKS', 'SKIJKS'
        phases = 'P', 'S'
        phases = 'PS', 
        phases = 'PcS', #'ScS', 'PcS'
        phases = 'K', #'SKS', #'PKS', 'SKS'
        phases = 'PKIJKP',
        for phase in phases:
            res = app.rp4phase(phase)
            if len(res) == 3:
                ps, ibs, ibreaks = res
                istarts = ibreaks[:-1]
                iends   = ibreaks[1:]
                nseg = istarts.size
                for iseg, (i0, i1) in enumerate(zip(istarts, iends)):
                    ps_seg = ps[i0:i1]
                    ibs_seg= ibs[i0:i1]
                    if True: #iseg!=nseg-1:
                        r_seg  = r[ibs_seg+1]
                        ax1.plot(ps_seg, r_seg, marker='.', label=f'{phase} {iseg}', alpha=0.7)
            else:
                ps, ibs, ibs2, ibreaks = res
                istarts = ibreaks[:-1]
                iends   = ibreaks[1:]
                nseg = istarts.size
                for iseg, (i0, i1) in enumerate(zip(istarts, iends)):
                    ps_seg = ps[i0:i1]
                    ibs_seg= ibs[i0:i1]
                    ibs2_seg= ibs2[i0:i1]
                    if  True: #iseg!=nseg-1:
                        r_seg  = r[ibs_seg+1]
                        ax1.plot(ps_seg, r_seg, marker='.', label=f'{phase} {iseg}', alpha=0.7)
                        r_seg  = r[ibs2_seg+1]
                        ax1.plot(ps_seg, r_seg, marker='.', label=f'{phase} {iseg}', alpha=0.7)
        ax1.legend(fontsize='small')
        plt.show()
    ######################################################################################
    @staticmethod
    def benchmark():
        r, vp, vs, icmb, iicb = fast_taup.rd_prem_model()
        app = SphFastTaup(r, vp, vs, icmb, iicb)
        


if __name__ == "__main__":
    #benchmark_p2xt()
    SphFastTaup.benchmark_jac()
    #benchmark_r2p()