##import python modules
import pyomo.environ as pe

##import AFO module
import Season as zgen


def season_precalcs(params, r_vals):
    zgen.f_season_precalcs(params, r_vals)



def f1_seasonpyomo_local(params, model):
    #############
    #parameters #
    #############
    model.p_parentz_provwithin_season = pe.Param(model.s_season_periods, model.s_season_types, model.s_season_types,
                                                  initialize=params['p_parentz_provwithin_season'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous node to z9 constraint in the current node within years')
    model.p_parentz_provbetween_season = pe.Param(model.s_season_periods, model.s_season_types, model.s_season_types,
                                                  initialize=params['p_parentz_provbetween_season'], default=0.0,
                                                  mutable=False, doc='Transfer of z8 dv in the previous node to z9 constraint in the current node between years')
    model.p_mask_childz_within_season = pe.Param(model.s_season_periods, model.s_season_types, initialize=params['p_mask_childz_within_season'],
                                           default=0.0, mutable=False, doc='mask child season require in each node within year')
    model.p_mask_childz_between_season = pe.Param(model.s_season_periods, model.s_season_types, initialize=params['p_mask_childz_between_season'],
                                            default=0.0, mutable=False, doc='mask child season require in each node between years')
    model.p_wyear_inc_qs = pe.Param(model.s_sequence_year, model.s_sequence, initialize=params['p_wyear_inc_qs'], default=0.0,
                                    mutable=False, doc='weather year included in sequence')
    model.p_season_prob_qsz = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, initialize=params['p_season_prob_qsz'],
                                       default=0.0, mutable=False, doc='sequence probability')
    model.p_endstart_prov_qsz = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, initialize=params['p_endstart_prov_qsz'], default=0.0,
                                         mutable=False, doc='transfer at the end of the sequence to the start')
    model.p_sequence_prov_qs8zs9 = pe.Param(model.s_sequence_year, model.s_sequence, model.s_season_types, model.s_sequence,
                                            initialize=params['p_sequence_prov_qs8zs9'], default=0.0,
                                            mutable=False, doc='transfer at the end of the sequence to the start')

