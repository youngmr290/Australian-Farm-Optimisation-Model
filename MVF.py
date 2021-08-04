'''
Marginal value of feed.

Module is built differently to normal (pyomo and precalcs together) to keep it as simple as possible
This is because this module is an add-on. So it is built as solo as possible.
The functions are called from corepyomo.py.
'''
import pyomo.environ as pe

import PropertyInputs as pinp
import UniversalInputs as uinp
import FeedsupplyFunctions as fsfun

def f_vol():
    '''calc volume of 100mj with different digestibility.'''
    dmd_q = pinp.mvf['i_mvf_dmd_q']
    mvf_me = pinp.mvf['i_mvf_me'] #me used for mvf
    me_q = fsfun.dmd_to_md(dmd_q)

    ##calc ri quality
    clover_propn = 0.3
    ###select which equation is used for ri.
    eqn_used_g1_q1p = uinp.sheep['i_eqn_used_g1_q1p7'][:, 0:1] #take equation system from first period.
    eqn_group = 6
    eqn_system = 0  # CSIRO = 0
    if uinp.sheep['i_eqn_exists_q0q1'][eqn_group,eqn_system]:  # proceed with call & assignment if this system exists for this group
        ###dams
        eqn_used = (eqn_used_g1_q1p[eqn_group,0] == eqn_system)
        if eqn_used:
            rq_q = fsfun.f_rq_cs(dmd_q, clover_propn)
        ri_qual_q = fsfun.f_rel_intake(1, rq_q, clover_propn)  # base the quality groups on ra = 1
    volume_q = 1 / ri_qual_q
    volume_100mj_q = volume_q / me_q * mvf_me
    ##make vol a dict for pyomo
    keys = pinp.mvf['i_q_idx']
    volume = dict(zip(keys, volume_100mj_q))
    return volume


#######
#pyomo#
#######
def f1_mvf_pyomo(model):
    model.v_mvf = pe.Var(model.s_feed_periods,model.s_feed_pools, model.s_mvf_q,bounds=(0,0),
                       doc='marginal value of feed. Must be bound to 0. Can be examined in duals to see value of extra ME.')



##me and vol functions called by corepyomo
def f_mvf_me(model,p6,f):
    '''
    Calculate the total energy provided by each MVF activity.

    Used in global constraint (con_me). See CorePyomo
    '''
    return sum(model.v_mvf[p6,f,q] * pinp.mvf['i_mvf_me'] for q in model.s_mvf_q)

def f_mvf_vol(model,p6,f):
    '''
    Calculate the total volume required by each MVF activity.

    Used in global constraint (con_vol). See CorePyomo
    '''
    return sum(model.v_mvf[p6,f,q] * f_vol()[q] for q in model.s_mvf_q)
