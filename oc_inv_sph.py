#!/usr/bin/env python3

from pprint import pp
import numpy as np
from sacpy import oc_inv, fast_taup, utils
from numba import jit
import inspect
import matplotlib.pyplot as plt
import pylab as pl

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
# copied from fast_taup_sph.py

__zero_k_tol__           = 1e-9
__zero_dr_tol__          = 1e-5
__zero_p_sph_tol__       = 1e-2    # Empirical value: p=1e-2 correspond to theta = 0.0005 degree from surface with v=6km/s
__zero_k_mul_p_sph_tol__ = 1e-8    # Zero for c=p*k. Used in F2(...) function.
__next_after_zero_p_sph__= 1e-10  # The next float after zero for p in spherical model.
__ENABLE_NUMBA__ = True
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

######################################################################################
######################################################################################
######################################################################################
######################################################################################
######################################################################################
@jit_wrapper(nopython=True, nogil=True)
def rv2ps(r, vr, max_theta_step_rad=0.0017): # get ps (s/rad) given an OC model
    """
    Return (ps, ibs, ibreaks) given a model (r, vr)
    """
    vz = vr / r
    ######
    p, ib, ibk = r2p_turn_back(vz)
    result = denser_ps(p, ib, ibk, v0=vr[0], R0=r[0], max_theta_step_rad=max_theta_step_rad)
    return result
def benchmark_rv2ps():
    r, vr = oc_inv.rd_prem_OC_model()
    ####
    result = rv2ps(r, vr)
    ps, ibs, ibreaks = result
    ##########
    fig, (ax1) = plt.subplots(1,1, figsize=(6,8))
    #ax1.plot(r_v, r, color='k')
    istarts = ibreaks[:-1]
    iends   = ibreaks[1:]
    nseg = istarts.size
    for iseg, (i0, i1) in enumerate(zip(istarts, iends)):
        ps_seg = ps[i0:i1]
        ibs_seg= ibs[i0:i1]
        if True: #iseg!=nseg-1:
            r_seg  = r[ibs_seg]
            r_seg  = r[ibs_seg+1]
            ax1.plot(ps_seg, r_seg, marker='s', label=f'seg{iseg}')
    plt.show()

@jit_wrapper(nopython=True, nogil=True)
def phase_ps2xt(ps, ibs, r, v, dr, dv, k, ik, ncv, nzt): # get traveltime curve given slowness and model
    """
    r
    v
    dr   = np.diff(r)
    dv   = np.diff(v)
    k    = np.divide(dv, dr, out=np.zeros_like(dv), where=dr!=0)
    ik   = np.divide(dr, dv, out=np.zeros_like(dv), where=dv!=0)
    ncv  = np.where( np.abs(k) > zero_k_tol, True, False )   #### whether a non-const velocity layer for each layer. True for non-const velocity layer
    nzt  = np.where( np.abs(dr) > zero_dr_tol, True, False ) #### whether a non-zero thickness layer for each layer. True for non-zero thickness layer
    """
    xs = np.zeros(ps.size, dtype=np.float64)
    ts = np.zeros(ps.size, dtype=np.float64)
    for ip in range(ps.size):
        dx, dt, tmp_ib = p2xt(ps[ip], r, v, dr, dv, k, ik, ncv, nzt, ibs[ip])
        #
        xs[ip] = dx
        ts[ip] = dt
    return xs*2.0, ts*2.0
def benchmark_phase_ps2xt():
    r, vr = oc_inv.rd_prem_OC_model()
    ps, ibs, ibreaks  = rv2ps(r, vr)
    #
    dr   = np.diff(r)
    dv   = np.diff(vr)
    k    = np.divide(dv, dr, out=np.zeros_like(dv), where=dr!=0)
    ik   = np.divide(dr, dv, out=np.zeros_like(dv), where=dv!=0)
    ncv  = np.where( np.abs(k) > __zero_k_tol__, True, False )   #### whether a non-const velocity layer for each layer. True for non-const velocity layer
    nzt  = np.where( np.abs(dr) > __zero_dr_tol__, True, False ) #### whether a non-zero thickness layer for each layer. True for non-zero thickness layer
    #
    xs, ts = phase_ps2xt(ps, ibs, r, vr, dr, dv, k, ik, ncv, nzt)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(np.rad2deg(xs), ts)
    ############
    # compare with flattend Earth
    r, vr = oc_inv.rd_prem_OC_model()
    #r = r[::2]
    #vr = vr[::2]
    z, vz = oc_inv.flatten(r, vr, 6371.0)
    x_km = xs*6371.0
    inv_rp_found, dist_found, trvt_found = oc_inv.many_dist2trvt(x_km, z, vz, 0.1, 1e-20, 1000)
    ax1.plot(np.rad2deg(dist_found/6371.0), trvt_found)
    #
    ax2.plot(np.rad2deg(xs), ts-trvt_found)
    plt.show()

@jit_wrapper(nopython=True, nogil=True)
def phase_x2pt(target_xs,
               ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks,
               r, v, dr, dv, k, ik, ncv, nzt,
               fast_arrival=False,
               niter=1000, xerr=1e-6, debug=False):
    ##############################################################################################################################
    if fast_arrival:
        res =phase_x2pt(target_xs,
                        ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks,
                        r, v, dr, dv, k, ik, ncv, nzt,
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
                        ib_mid = -1
                        dx, dt, tmp_ib = p2xt(p_mid, r, v, dr, dv, k, ik, ncv, nzt)
                        x_mid = dx*2.0
                        t_mid = dt*2.0
                        ib_mid = tmp_ib if tmp_ib>ib_mid else ib_mid
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
    return p_found, x_found, t_found, ib_found, ix_found
def benchmark_phase_x2pt():
    r, vr = oc_inv.rd_prem_OC_model()
    ps, ibs, ibreaks  = rv2ps(r, vr)
    #
    dr   = np.diff(r)
    dv   = np.diff(vr)
    k    = np.divide(dv, dr, out=np.zeros_like(dv), where=dr!=0)
    ik   = np.divide(dr, dv, out=np.zeros_like(dv), where=dv!=0)
    ncv  = np.where( np.abs(k) > __zero_k_tol__, True, False )   #### whether a non-const velocity layer for each layer. True for non-const velocity layer
    nzt  = np.where( np.abs(dr) > __zero_dr_tol__, True, False ) #### whether a non-zero thickness layer for each layer. True for non-zero thickness layer
    #
    xs, ts = phase_ps2xt(ps, ibs, r, vr, dr, dv, k, ik, ncv, nzt)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    ax1.plot(np.rad2deg(xs), ts)
    #
    ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks = ps, xs, ts, ibs, ibreaks
    #########################
    target_xs_deg = np.arange(10, 120)
    target_xs_rad = np.deg2rad(target_xs_deg)
    res = phase_x2pt(target_xs_rad,
                        ds_ps, ds_xs, ds_ts, ds_ibs, ds_ibks,
                        r, vr, dr, dv, k, ik, ncv, nzt,
                        fast_arrival=False,
                        niter=1000, xerr=1e-6, debug=False)
    p_found, x_found, t_found, ib_found, ix_found = res
    ax1.plot(np.rad2deg(x_found), t_found, 'ro')
    plt.show()

@jit_wrapper(nopython=True, nogil=True)
def calculate_x_over_y(x, y):
    x_over_y = np.zeros_like(x)
    for i in range(x.size):
        if np.abs(y[i]) >= 1e-3:
            x_over_y[i] = x[i] / y[i]
    return x_over_y

@jit_wrapper(nopython=True, nogil=True)
def many_dist2trvt(x_rad, r, vr, xerr=1e-20, niter=1000):
    """
    return p_found, x_found, t_found, ib_found, ix_found
    """
    ps, ibs, ibreaks  = rv2ps(r, vr)
    #
    #print(r.dtype, vr.dtype, r.shape, vr.shape)
    dr   = np.diff(r)
    dv   = np.diff(vr)
    k    = calculate_x_over_y(dv, dr)  #np.divide(dv, dr, out=np.zeros_like(dv), where=dr!=0)
    ik   = calculate_x_over_y(dr, dv)  #np.divide(dr, dv, out=np.zeros_like(dv), where=dv!=0)
    ncv  = np.where( np.abs(k) > __zero_k_tol__, True, False )   #### whether a non-const velocity layer for each layer. True for non-const velocity layer
    nzt  = np.where( np.abs(dr) > __zero_dr_tol__, True, False ) #### whether a non-zero thickness layer for each layer. True for non-zero thickness layer
    #
    xs, ts = phase_ps2xt(ps, ibs, r, vr, dr, dv, k, ik, ncv, nzt)
    res = phase_x2pt(x_rad, ps, xs, ts, ibs, ibreaks, r, vr, dr, dv, k, ik, ncv, nzt, fast_arrival=True, niter=niter, xerr=xerr, debug=False)
    return res[:3]


def benchmark_manyx2t():
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    x_deg = np.arange(10, 130)
    x_rad = np.deg2rad(x_deg)
    r, vr = oc_inv.rd_prem_OC_model()
    _,_,tr = many_dist2trvt(x_rad, r, vr, 1e-5, 1000)
    for dr in (100, 50, 25):
        r, vr = oc_inv.rd_prem_OC_model()
        r, vr = oc_inv.denser_xy(r, vr, dr)
        _,_,tmp = many_dist2trvt(x_rad, r, vr, 1e-5, 1000)
        ax1.plot(vr, r, '.-')
        ax2.plot(x_deg, tmp-tr)
    plt.show()
    
    

def benchmark_manyx2t2():
    # compare error and time consumption
    x_deg = np.arange(10, 130)
    x_km  = np.deg2rad(x_deg)*6371.0
    x_rad = np.deg2rad(x_deg)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
    r, vr = oc_inv.rd_prem_OC_model()
    _,_,tr = many_dist2trvt(x_rad, r, vr, 1e-10, 1000)
    with utils.Timer(tag='rvr'):
        _,_,tr = many_dist2trvt(x_rad, r, vr, 1e-10, 1000)
    ax1.plot(x_deg, tr, 'r-', label='tr')
    #
    for dr in (100, 50, 20, 10, 5, 1):
        r, vr = oc_inv.rd_prem_OC_model()
        r, vr = oc_inv.denser_xy(r, vr, dr)
        z, vz = oc_inv.flatten(r, vr, 6371.0)
        _,_,tz = oc_inv.many_dist2trvt(x_km, z, vz, 0.1, 1e-3, 1000)
        with utils.Timer(tag=f'zvz_{dr}'):
            _,_,tz = oc_inv.many_dist2trvt(x_km, z, vz, 0.1, 1e-3, 1000)
        #
        ax1.plot(x_deg, tz, label='Taup')
        ax2.semilogy(x_deg, np.abs(tz - tr), label='Difference')
    plt.show()


def fix_zero_division():
    dvr= np.array([ 0.2608022938956499  , -0.06516106251189305 ,
        -0.1301056015580104  , -0.023656319912071958,
        -0.054213999653071054, -0.040712346338755895,
        -0.027027604018905582, -0.012086191383282418,
        -0.01999833857368121 , -0.005579177998868473,
            0.034494822747929685,  0.038981312301808235,
        -0.015640848269821667,  0.3100212379519294  ,
        -0.12942345187004822 ])
    x_rad= np.array([0.5235987755982988, 0.5410520681182421, 0.5585053606381855,
        0.5759586531581288, 0.5934119456780721, 0.6108652381980153,
        0.6283185307179586, 0.6457718232379019, 0.6632251157578453,
        0.6806784082777885, 0.6981317007977318, 0.7155849933176751,
        0.7330382858376184, 0.7504915783575618, 0.767944870877505 ,
        0.7853981633974483, 0.8028514559173916, 0.8203047484373349,
        0.8377580409572782, 0.8552113334772214, 0.8726646259971648,
        0.8901179185171081, 0.9075712110370514, 0.9250245035569946,
        0.9424777960769379, 0.9599310885968813, 0.9773843811168246,
        0.9948376736367679, 1.0122909661567112, 1.0297442586766545,
        1.0471975511965976, 1.064650843716541 , 1.0821041362364843,
        1.0995574287564276, 1.117010721276371 , 1.1344640137963142,
        1.1519173063162575, 1.1693705988362009, 1.1868238913561442,
        1.2042771838760873, 1.2217304763960306, 1.239183768915974 ,
        1.2566370614359172, 1.2740903539558606, 1.2915436464758039,
        1.3089969389957472, 1.3264502315156905, 1.3439035240356338,
        1.361356816555577 , 1.3788101090755203, 1.3962634015954636,
        1.413716694115407 , 1.4311699866353502, 1.4486232791552935,
        1.4660765716752369, 1.4835298641951802, 1.5009831567151235,
        1.5184364492350666, 1.53588974175501  , 1.5533430342749532,
        1.5707963267948966, 1.5882496193148399, 1.6057029118347832,
        1.6231562043547265, 1.6406094968746698, 1.6580627893946132,
        1.6755160819145565, 1.6929693744344996, 1.710422666954443 ,
        1.7278759594743862, 1.7453292519943295, 1.7627825445142729,
        1.7802358370342162, 1.7976891295541595, 1.8151424220741028,
        1.8325957145940461, 1.8500490071139892, 1.8675022996339325,
        1.8849555921538759, 1.9024088846738192, 1.9198621771937625])
    r= np.array([3480., 3400., 3300., 3200., 3100., 3000., 2800., 2600., 2400.,
        2200., 2000., 1800., 1600., 1400., 1200.])
    vr_ref= np.array([ 8.06482,  8.19939,  8.36019,  8.51298,  8.65805,  8.79573,
            9.05015,  9.27867,  9.48409,  9.66865,  9.83496,  9.98554,
        10.12291, 10.24959, 10.35568])
    #####
    many_dist2trvt(x_rad[:1], r, vr_ref+dvr, 1e-5, 1000)


if __name__ == '__main__':
    #benchmark_rv2ps()
    #benchmark_phase_ps2xt()
    #benchmark_phase_x2pt()
    #benchmark_manyx2t()
    #benchmark_manyx2t2()
    #fix_zero_division()

    y1 = np.array([0.2, 0.3])
    y2 = np.array([3480.0, 1200.0])
    y3 = np.array([10.0, 10.0])
    _,_,data_ref = many_dist2trvt(y1, y2, y3, 1e-5, 1000)
