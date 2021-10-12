##import python modules
import pyomo.environ as pe

##import AFO module
import Season as zgen


def season_precalcs(params, r_vals):
    zgen.f_season_precalcs(params)



def f1_seasonpyomo_local(params, model):
    #############
    #parameters #
    #############
    model.p_wyear_inc_qs = pe.Param(model.s_sequence_year, model.s_sequence, initialize=params['p_wyear_inc_qs'], default=0.0,
                                    mutable=False, doc='weather year included in sequence')
    model.p_between_req_qs = pe.Param(model.s_sequence_year, model.s_sequence, initialize=params['p_between_req_qs'], default=0.0,
                                      mutable=False, doc='sequence require')
    model.p_season_prob_qsz = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, initialize=params['p_season_prob_qsz'],
                                       default=0.0, mutable=False, doc='sequence probability')
    model.p_endstart_prov_qsz = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, initialize=params['p_endstart_prov_qsz'], default=0.0,
                                         mutable=False, doc='transfer at the end of the sequence to the start')
    model.p_sequence_prov_qs8zs9 = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_sequence,
                                            initialize=params['p_sequence_prov_qs8zs9'], default=0.0,
                                            mutable=False, doc='transfer at the end of the sequence to the start')

