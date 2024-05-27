import pandas as pd
import numpy as np
import sys
import os


from ..AfoLogic import ReportFunctions as rfun
from ..AfoLogic import relativeFile

#todo should be able to simplify this module using loops with a couple of additions for the unique reports ie writing to excel needs to be diffferent for reports where multiple tables are written to the same sheet. Maybe loop over all report that are standard then do the weird ones by hand after.
def f_create_report_dfs(non_exist_trials):
    reports = {}
    ##create empty df to stack each trial results into
    reports["stacked_infeasible"] = pd.DataFrame().rename_axis('Trial')  # name of any infeasible trials
    reports["stacked_non_exist"] = pd.DataFrame(non_exist_trials).rename_axis('Trial')  # name of any infeasible trials
    reports["stacked_summary"] = pd.DataFrame()  # 1 line summary of each trial
    reports["stacked_areasum"] = pd.DataFrame()  # area summary
    reports["stacked_cropsum"] = pd.DataFrame()  # area summary
    reports["stacked_profit"] = pd.DataFrame()  # profit
    reports["stacked_numbers_qsz"] = pd.DataFrame()  # total dse by qsz
    reports["stacked_croparea_qsz"] = pd.DataFrame()  # total crop by qsz
    reports["stacked_pnl"] = pd.DataFrame()  # profit and loss statement
    reports["stacked_wc"] = pd.DataFrame()  # max bank overdraw
    reports["stacked_penalty"] = pd.DataFrame()  # biomass penalty from seeding timeliness and crop grazing
    reports["stacked_sowing_date"] = pd.DataFrame()  # average sowing date
    reports["stacked_profitarea"] = pd.DataFrame()  # profit by land area
    reports["stacked_feed"] = pd.DataFrame()  # feed budget
    reports["stacked_feed2"] = pd.DataFrame()  # feed budget
    reports["stacked_grazing"] = pd.DataFrame()  # grazing summary
    reports["stacked_ewe_numbers_summary"] = pd.DataFrame()  # grazing summary
    reports["stacked_wethers_n_crossys_numbers_summary"] = pd.DataFrame()  # grazing summary
    reports["stacked_emissions"] = pd.DataFrame()  # GHG emission summary
    reports["stacked_season_nodes"] = pd.DataFrame()  # season periods
    reports["stacked_feed_periods"] = pd.DataFrame()  # feed periods
    reports["stacked_dam_dvp_dates"] = pd.DataFrame()  # dam dvp dates
    reports["stacked_repro_dates"] = pd.DataFrame()  # dam repro dates
    reports["stacked_offs_dvp_dates"] = pd.DataFrame()  # offs dvp dates
    reports["stacked_saleprice"] = pd.DataFrame()  # sale price
    reports["stacked_salegrid_dams"] = pd.DataFrame()  # sale grid
    reports["stacked_salegrid_yatf"] = pd.DataFrame()  # sale grid
    reports["stacked_salegrid_offs"] = pd.DataFrame()  # sale grid
    reports["stacked_saleage_offs"] = pd.DataFrame()  # sale grid
    reports["stacked_salevalue_dams"] = pd.DataFrame()  # average sale value dams
    reports["stacked_salevalue_offs"] = pd.DataFrame()  # average sale value offs
    reports["stacked_salevalue_prog"] = pd.DataFrame()  # average sale value offs
    reports["stacked_woolvalue_dams"] = pd.DataFrame()  # average wool value dams
    reports["stacked_woolvalue_offs"] = pd.DataFrame()  # average wool value offs
    reports["stacked_saledate_offs"] = pd.DataFrame()  # offs sale date
    reports["stacked_cfw_dams"] = pd.DataFrame()  # clean fleece weight dams
    reports["stacked_fd_dams"] = pd.DataFrame()  # fibre diameter dams
    reports["stacked_cfw_offs"] = pd.DataFrame()  # clean fleece weight dams
    reports["stacked_fd_offs"] = pd.DataFrame()  # fibre diameter dams
    reports["stacked_wbe_dams"] = pd.DataFrame()  # whole body energy content dams
    reports["stacked_wbe_cut_dams"] = pd.DataFrame()  # whole body energy content dams for select p period
    reports["stacked_fat_dams"] = pd.DataFrame()  # fat mass dams
    reports["stacked_fat_cut_dams"] = pd.DataFrame()  # fat mass dams for select p period
    reports["stacked_muscle_dams"] = pd.DataFrame()  # muscle mass dams
    reports["stacked_muscle_cut_dams"] = pd.DataFrame()  # muscle mass dams for select p period
    reports["stacked_viscera_dams"] = pd.DataFrame()  # viscera mass dams
    reports["stacked_viscera_cut_dams"] = pd.DataFrame()  # viscera mass dams for select p period
    reports["stacked_wbe_offs"] = pd.DataFrame()  # whole body energy content offs
    reports["stacked_lw_dams"] = pd.DataFrame()  # live weight dams (large array with p, e and b axis)
    reports["stacked_ebw_dams"] = pd.DataFrame()  # empty body weight dams (large array with p, e and b axis)
    reports["stacked_ebw_cut_dams"] = pd.DataFrame()  #empty body weight dams for select p period
    reports["stacked_cs_dams"] = pd.DataFrame()  # condition score dams (large array with p, e and b axis)
    reports["stacked_fs_dams"] = pd.DataFrame()  # fat score dams (large array with p, e and b axis)
    reports["stacked_nv_dams"] = pd.DataFrame()  # diet nutritive value for dams (large array with p, e and b axis)
    reports["stacked_ebw_yatf"] = pd.DataFrame()  # empty body weight yatf (large array with p, e and b axis)
    reports["stacked_ebw_prog"] = pd.DataFrame()  # empty body weight prog (large array with p, e and b axis)
    reports["stacked_ebw_offs"] = pd.DataFrame()  # empty body weight offs (large array with p, e and b axis)
    reports["stacked_cs_offs"] = pd.DataFrame()  # condition score offs (large array with p, e and b axis)
    reports["stacked_fs_offs"] = pd.DataFrame()  # fat score offs (large array with p, e and b axis)
    reports["stacked_nv_offs"] = pd.DataFrame()  # diet nutritive value for offs (large array with p, e and b axis)
    reports["stacked_weanper"] = pd.DataFrame()  # weaning percent
    reports["stacked_scanper"] = pd.DataFrame()  # scan percent
    reports["stacked_dry_propn"] = pd.DataFrame()  # dry ewe proportion
    reports["stacked_lamb_survival"] = pd.DataFrame()  # lamb survival
    reports["stacked_daily_mei_dams"] = pd.DataFrame()  # mei dams
    reports["stacked_daily_pi_dams"] = pd.DataFrame()  # potential intake dams
    reports["stacked_daily_mei_offs"] = pd.DataFrame()  # mei offs
    reports["stacked_daily_pi_offs"] = pd.DataFrame()  # potential intake offs
    reports["stacked_numbers_dams"] = pd.DataFrame()  # numbers dams
    reports["stacked_numbers_dams_p"] = pd.DataFrame()  # numbers dams with p axis (large array)
    reports["stacked_numbers_prog"] = pd.DataFrame()  # numbers prog
    reports["stacked_numbers_offs"] = pd.DataFrame()  # numbers offs
    reports["stacked_numbers_offs_p"] = pd.DataFrame()  # numbers offs with p axis (large array)
    reports["stacked_mort_dams"] = pd.DataFrame()  # mort dams with p axis (large array)
    reports["stacked_mort_offs"] = pd.DataFrame()  # mort offs with p axis (large array)
    reports["stacked_dse_sire"] = pd.DataFrame()  # dse based on normal weight
    reports["stacked_dse_dams"] = pd.DataFrame()  # dse based on normal weight
    reports["stacked_dse_offs"] = pd.DataFrame()  # dse based on normal weight
    reports["stacked_dse1_sire"] = pd.DataFrame()  # dse based on mei
    reports["stacked_dse1_dams"] = pd.DataFrame()  # dse based on mei
    reports["stacked_dse1_offs"] = pd.DataFrame()  # dse based on mei
    reports["stacked_pgr"] = pd.DataFrame()  # pasture growth
    reports["stacked_grnfoo"] = pd.DataFrame()  # green foo
    reports["stacked_dryfoo"] = pd.DataFrame()  # dry foo
    reports["stacked_napfoo"] = pd.DataFrame()  # non-arable pasture foo
    reports["stacked_grncon"] = pd.DataFrame()  # green pasture consumed
    reports["stacked_drycon"] = pd.DataFrame()  # dry pasture consumed
    reports["stacked_napcon"] = pd.DataFrame()  # non-arable pasture feed consumed
    reports["stacked_poccon"] = pd.DataFrame()  # pasture on crop paddocks feed consumed
    reports["stacked_supcon"] = pd.DataFrame()  # supplement feed consumed
    reports["stacked_stubcon"] = pd.DataFrame()  # stubble feed consumed
    reports["stacked_cropcon"] = pd.DataFrame()  # crop consumed from early season crop grazing
    reports["stacked_cropcon_available"] = pd.DataFrame()  # crop consumed from early season crop grazing
    reports["stacked_grnnv"] = pd.DataFrame()  # NV of green pas
    reports["stacked_grndmd"] = pd.DataFrame()  # dmd of green pas
    reports["stacked_avegrnfoo"] = pd.DataFrame()  # Average Foo of green pas
    reports["stacked_drynv"] = pd.DataFrame()  # NV of dry pas
    reports["stacked_drydmd"] = pd.DataFrame()  # dmd of dry pas
    reports["stacked_avedryfoo"] = pd.DataFrame()  # Average Foo of dry pas
    reports["stacked_mvf"] = pd.DataFrame()  # Marginal value of feed
    reports["stacked_pasture_area"] = pd.DataFrame()  # web app analysis
    reports["stacked_stocking_rate"] = pd.DataFrame()  # web app analysis
    reports["stacked_legume"] = pd.DataFrame()  # web app analysis

    return reports

def f_concat_reports(stacked_reports, reports, report_run, trial_name):
    stacked_reports["stacked_infeasible"] = rfun.f_append_dfs(stacked_reports["stacked_infeasible"], reports["infeasible"])

    if report_run.loc['run_summary', 'Run']:
        summary = pd.concat([reports["summary"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_summary"] = rfun.f_append_dfs(stacked_reports["stacked_summary"], summary)

    if report_run.loc['run_areasum', 'Run']:
        areasum = pd.concat([reports["areasum"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_areasum"] = rfun.f_append_dfs(stacked_reports["stacked_areasum"], areasum)

    if report_run.loc['run_cropsum', 'Run']:
        cropsum = pd.concat([reports["cropsum"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cropsum"] = rfun.f_append_dfs(stacked_reports["stacked_cropsum"], cropsum)

    if report_run.loc['run_profit', 'Run']:
        profit = pd.concat([reports["profit"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_profit"] = rfun.f_append_dfs(stacked_reports["stacked_profit"], profit)

    if report_run.loc['run_numbers_qsz', 'Run']:
        numbers_qsz = pd.concat([reports["numbers_qsz"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_qsz"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_qsz"], numbers_qsz)

    if report_run.loc['run_croparea_qsz', 'Run']:
        croparea_qsz = pd.concat([reports["croparea_qsz"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_croparea_qsz"] = rfun.f_append_dfs(stacked_reports["stacked_croparea_qsz"], croparea_qsz)

    if report_run.loc['run_pnl', 'Run']:
        pnl = pd.concat([reports["pnl"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_pnl"] = rfun.f_append_dfs(stacked_reports["stacked_pnl"], pnl)

    if report_run.loc['run_wc', 'Run']:
        wc = pd.concat([reports["wc"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_wc"] = rfun.f_append_dfs(stacked_reports["stacked_wc"], wc)

    if report_run.loc['run_biomass_penalty', 'Run']:
        penalty = pd.concat([reports["penalty"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_penalty"] = rfun.f_append_dfs(stacked_reports["stacked_penalty"], penalty)

    if report_run.loc['run_sowing_date', 'Run']:
        penalty = pd.concat([reports["sowing_date"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_sowing_date"] = rfun.f_append_dfs(stacked_reports["stacked_sowing_date"], penalty)

    if report_run.loc['run_profitarea', 'Run']:
        profitarea = pd.concat([reports["profitarea"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_profitarea"] = rfun.f_append_dfs(stacked_reports["stacked_profitarea"], profitarea)

    if report_run.loc['run_feedbudget', 'Run']:
        feed = pd.concat([reports["feed"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_feed"] = rfun.f_append_dfs(stacked_reports["stacked_feed"], feed)

    if report_run.loc['run_feedbudget', 'Run']:
        feed = pd.concat([reports["feed_total"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_feed2"] = rfun.f_append_dfs(stacked_reports["stacked_feed2"], feed)

    if report_run.loc['run_feedbudget', 'Run']:
        grazing = pd.concat([reports["grazing"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_grazing"] = rfun.f_append_dfs(stacked_reports["stacked_grazing"], grazing)

    if report_run.loc['run_numbers_summary', 'Run']:
        ewe_numbers_summary = pd.concat([reports["ewe_numbers_summary"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        wethers_n_crossys_numbers_summary = pd.concat([reports["wethers_n_crossys_numbers_summary"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ewe_numbers_summary"] = rfun.f_append_dfs(stacked_reports["stacked_ewe_numbers_summary"], ewe_numbers_summary)
        stacked_reports["stacked_wethers_n_crossys_numbers_summary"] = rfun.f_append_dfs(stacked_reports["stacked_wethers_n_crossys_numbers_summary"], wethers_n_crossys_numbers_summary)

    if report_run.loc['run_emissions', 'Run']:
        grazing = pd.concat([reports["emissions"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_emissions"] = rfun.f_append_dfs(stacked_reports["stacked_emissions"], grazing)

    if report_run.loc['run_period_dates', 'Run']:
        season_nodes = pd.concat([reports["season_nodes"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_season_nodes"] = rfun.f_append_dfs(stacked_reports["stacked_season_nodes"], season_nodes)

        ###feed periods (p6)
        feed_periods = pd.concat([reports["feed_periods"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_feed_periods"] = rfun.f_append_dfs(stacked_reports["stacked_feed_periods"], feed_periods)

        ###dams dvp
        dam_dvp_dates = pd.concat([reports["dam_dvp_dates"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_dam_dvp_dates"] = rfun.f_append_dfs(stacked_reports["stacked_dam_dvp_dates"], dam_dvp_dates)

        ###dams repro dates
        repro_dates = pd.concat([reports["repro_dates"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_repro_dates"] = rfun.f_append_dfs(stacked_reports["stacked_repro_dates"], repro_dates)

        ###offs dvp
        offs_dvp_dates = pd.concat([reports["offs_dvp_dates"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_offs_dvp_dates"] = rfun.f_append_dfs(stacked_reports["stacked_offs_dvp_dates"], offs_dvp_dates)

    if report_run.loc['run_saleprice', 'Run']:
        saleprice = pd.concat([reports["saleprice"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_saleprice"] = rfun.f_append_dfs(stacked_reports["stacked_saleprice"], saleprice)

    if report_run.loc['run_salegrid_dams', 'Run']:
        salegrid_dams = pd.concat([reports["salegrid_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salegrid_dams"] = rfun.f_append_dfs(stacked_reports["stacked_salegrid_dams"], salegrid_dams)

    if report_run.loc['run_salegrid_yatf', 'Run']:
        salegrid_yatf = pd.concat([reports["salegrid_yatf"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salegrid_yatf"] = rfun.f_append_dfs(stacked_reports["stacked_salegrid_yatf"], salegrid_yatf)

    if report_run.loc['run_salegrid_offs', 'Run']:
        salegrid_offs = pd.concat([reports["salegrid_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salegrid_offs"] = rfun.f_append_dfs(stacked_reports["stacked_salegrid_offs"], salegrid_offs)

    if report_run.loc['run_saleage_offs', 'Run']:
        saleage_offs = pd.concat([reports["saleage_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_saleage_offs"] = rfun.f_append_dfs(stacked_reports["stacked_saleage_offs"], saleage_offs)

    if report_run.loc['run_salevalue_dams', 'Run']:
        salevalue_dams = pd.concat([reports["salevalue_dams"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salevalue_dams"] = rfun.f_append_dfs(stacked_reports["stacked_salevalue_dams"], salevalue_dams)

    if report_run.loc['run_salevalue_offs', 'Run']:
        salevalue_offs = pd.concat([reports["salevalue_offs"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salevalue_offs"] = rfun.f_append_dfs(stacked_reports["stacked_salevalue_offs"], salevalue_offs)

    if report_run.loc['run_salevalue_prog', 'Run']:
        salevalue_prog = pd.concat([reports["salevalue_prog"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_salevalue_prog"] = rfun.f_append_dfs(stacked_reports["stacked_salevalue_prog"], salevalue_prog)

    if report_run.loc['run_woolvalue_dams', 'Run']:
        woolvalue_dams = pd.concat([reports["woolvalue_dams"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_woolvalue_dams"] = rfun.f_append_dfs(stacked_reports["stacked_woolvalue_dams"], woolvalue_dams)

    if report_run.loc['run_woolvalue_offs', 'Run']:
        woolvalue_offs = pd.concat([reports["woolvalue_offs"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_woolvalue_offs"] = rfun.f_append_dfs(stacked_reports["stacked_woolvalue_offs"], woolvalue_offs)

    if report_run.loc['run_saledate_offs', 'Run']:
        saledate_offs = pd.concat([reports["saledate_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_saledate_offs"] = rfun.f_append_dfs(stacked_reports["stacked_saledate_offs"], saledate_offs)

    if report_run.loc['run_cfw_dams', 'Run']:
        cfw_dams = pd.concat([reports["cfw_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cfw_dams"] = rfun.f_append_dfs(stacked_reports["stacked_cfw_dams"], cfw_dams)

    if report_run.loc['run_fd_dams', 'Run']:
        fd_dams = pd.concat([reports["fd_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fd_dams"] = rfun.f_append_dfs(stacked_reports["stacked_fd_dams"], fd_dams)

    if report_run.loc['run_cfw_offs', 'Run']:
        cfw_offs = pd.concat([reports["cfw_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cfw_offs"] = rfun.f_append_dfs(stacked_reports["stacked_cfw_offs"], cfw_offs)

    if report_run.loc['run_fd_offs', 'Run']:
        fd_offs = pd.concat([reports["fd_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fd_offs"] = rfun.f_append_dfs(stacked_reports["stacked_fd_offs"], fd_offs)

    if report_run.loc['run_wbe_dams', 'Run']:
        wbe_dams = pd.concat([reports["wbe_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_wbe_dams"] = rfun.f_append_dfs(stacked_reports["stacked_wbe_dams"], wbe_dams)

    if report_run.loc['run_wbe_cut_dams', 'Run']:
        wbe_cut_dams = pd.concat([reports["wbe_cut_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_wbe_cut_dams"] = rfun.f_append_dfs(stacked_reports["stacked_wbe_cut_dams"], wbe_cut_dams)

    if report_run.loc['run_fat_dams', 'Run']:
        fat_dams = pd.concat([reports["fat_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fat_dams"] = rfun.f_append_dfs(stacked_reports["stacked_fat_dams"], fat_dams)

    if report_run.loc['run_fat_cut_dams', 'Run']:
        fat_cut_dams = pd.concat([reports["fat_cut_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fat_cut_dams"] = rfun.f_append_dfs(stacked_reports["stacked_fat_cut_dams"], fat_cut_dams)

    if report_run.loc['run_muscle_dams', 'Run']:
        muscle_dams = pd.concat([reports["muscle_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_muscle_dams"] = rfun.f_append_dfs(stacked_reports["stacked_muscle_dams"], muscle_dams)

    if report_run.loc['run_muscle_cut_dams', 'Run']:
        muscle_cut_dams = pd.concat([reports["muscle_cut_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_muscle_cut_dams"] = rfun.f_append_dfs(stacked_reports["stacked_muscle_cut_dams"], muscle_cut_dams)

    if report_run.loc['run_viscera_dams', 'Run']:
        viscera_dams = pd.concat([reports["viscera_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_viscera_dams"] = rfun.f_append_dfs(stacked_reports["stacked_viscera_dams"], viscera_dams)

    if report_run.loc['run_viscera_cut_dams', 'Run']:
        viscera_cut_dams = pd.concat([reports["viscera_cut_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_viscera_cut_dams"] = rfun.f_append_dfs(stacked_reports["stacked_viscera_cut_dams"], viscera_cut_dams)

    if report_run.loc['run_wbe_offs', 'Run']:
        wbe_offs = pd.concat([reports["wbe_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_wbe_offs"] = rfun.f_append_dfs(stacked_reports["stacked_wbe_offs"], wbe_offs)

    if report_run.loc['run_lw_dams', 'Run']:
        lw_dams = pd.concat([reports["lw_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_lw_dams"] = rfun.f_append_dfs(stacked_reports["stacked_lw_dams"], lw_dams)

    if report_run.loc['run_ebw_dams', 'Run']:
        ebw_dams = pd.concat([reports["ebw_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ebw_dams"] = rfun.f_append_dfs(stacked_reports["stacked_ebw_dams"], ebw_dams)

    if report_run.loc['run_ebw_cut_dams', 'Run']:
        ebw_cut_dams = pd.concat([reports["ebw_cut_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ebw_cut_dams"] = rfun.f_append_dfs(stacked_reports["stacked_ebw_cut_dams"], ebw_cut_dams)

    if report_run.loc['run_cs_dams', 'Run']:
        cs_dams = pd.concat([reports["cs_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cs_dams"] = rfun.f_append_dfs(stacked_reports["stacked_cs_dams"], cs_dams)

    if report_run.loc['run_fs_dams', 'Run']:
        fs_dams = pd.concat([reports["fs_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fs_dams"] = rfun.f_append_dfs(stacked_reports["stacked_fs_dams"], fs_dams)

    if report_run.loc['run_nv_dams', 'Run']:
        nv_dams = pd.concat([reports["nv_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_nv_dams"] = rfun.f_append_dfs(stacked_reports["stacked_nv_dams"], nv_dams)

    if report_run.loc['run_ebw_yatf', 'Run']:
        ebw_yatf = pd.concat([reports["ebw_yatf"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ebw_yatf"] = rfun.f_append_dfs(stacked_reports["stacked_ebw_yatf"], ebw_yatf)

    if report_run.loc['run_ebw_prog', 'Run']:
        ebw_prog = pd.concat([reports["ebw_prog"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ebw_prog"] = rfun.f_append_dfs(stacked_reports["stacked_ebw_prog"], ebw_prog)

    if report_run.loc['run_ebw_offs', 'Run']:
        ebw_offs = pd.concat([reports["ebw_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_ebw_offs"] = rfun.f_append_dfs(stacked_reports["stacked_ebw_offs"], ebw_offs)

    if report_run.loc['run_cs_offs', 'Run']:
        cs_offs = pd.concat([reports["cs_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cs_offs"] = rfun.f_append_dfs(stacked_reports["stacked_cs_offs"], cs_offs)

    if report_run.loc['run_fs_offs', 'Run']:
        fs_offs = pd.concat([reports["fs_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_fs_offs"] = rfun.f_append_dfs(stacked_reports["stacked_fs_offs"], fs_offs)

    if report_run.loc['run_nv_offs', 'Run']:
        nv_offs = pd.concat([reports["nv_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_nv_offs"] = rfun.f_append_dfs(stacked_reports["stacked_nv_offs"], nv_offs)

    if report_run.loc['run_lamb_survival', 'Run']:
        lamb_survival = pd.concat([reports["lamb_survival"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_lamb_survival"] = rfun.f_append_dfs(stacked_reports["stacked_lamb_survival"], lamb_survival)

    if report_run.loc['run_weanper', 'Run']:
        weanper = pd.concat([reports["weanper"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_weanper"] = rfun.f_append_dfs(stacked_reports["stacked_weanper"], weanper)

    if report_run.loc['run_scanper', 'Run']:
        scanper = pd.concat([reports["scanper"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_scanper"] = rfun.f_append_dfs(stacked_reports["stacked_scanper"], scanper)

    if report_run.loc['run_dry_propn', 'Run']:
        dry_propn = pd.concat([reports["dry_propn"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_dry_propn"] = rfun.f_append_dfs(stacked_reports["stacked_dry_propn"], dry_propn)

    if report_run.loc['run_daily_mei_dams', 'Run']:
        daily_mei_dams = pd.concat([reports["daily_mei_dams"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_daily_mei_dams"] = rfun.f_append_dfs(stacked_reports["stacked_daily_mei_dams"], daily_mei_dams)

    if report_run.loc['run_daily_pi_dams', 'Run']:
        daily_pi_dams = pd.concat([reports["daily_pi_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_daily_pi_dams"] = rfun.f_append_dfs(stacked_reports["stacked_daily_pi_dams"], daily_pi_dams)

    if report_run.loc['run_daily_mei_offs', 'Run']:
        daily_mei_offs = pd.concat([reports["daily_mei_offs"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_daily_mei_offs"] = rfun.f_append_dfs(stacked_reports["stacked_daily_mei_offs"], daily_mei_offs)

    if report_run.loc['run_daily_pi_offs', 'Run']:
        daily_pi_offs = pd.concat([reports["daily_pi_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_daily_pi_offs"] = rfun.f_append_dfs(stacked_reports["stacked_daily_pi_offs"], daily_pi_offs)

    if report_run.loc['run_numbers_dams', 'Run']:
        numbers_dams = pd.concat([reports["numbers_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_dams"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_dams"], numbers_dams)

    if report_run.loc['run_numbers_dams_p', 'Run']:
        numbers_dams_p = pd.concat([reports["numbers_dams_p"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_dams_p"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_dams_p"], numbers_dams_p)

    if report_run.loc['run_numbers_prog', 'Run']:
        numbers_prog = pd.concat([reports["numbers_prog"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_prog"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_prog"], numbers_prog)

    if report_run.loc['run_numbers_offs', 'Run']:
        numbers_offs = pd.concat([reports["numbers_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_offs"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_offs"], numbers_offs)
    if report_run.loc['run_numbers_offs_p', 'Run']:
        numbers_offs_p = pd.concat([reports["numbers_offs_p"]], keys=[trial_name],
                                   names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_numbers_offs_p"] = rfun.f_append_dfs(stacked_reports["stacked_numbers_offs_p"], numbers_offs_p)

    if report_run.loc['run_mort_dams', 'Run']:
        mort_dams = pd.concat([reports["mort_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_mort_dams"] = rfun.f_append_dfs(stacked_reports["stacked_mort_dams"], mort_dams)

    if report_run.loc['run_mort_offs', 'Run']:
        mort_offs = pd.concat([reports["mort_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_mort_offs"] = rfun.f_append_dfs(stacked_reports["stacked_mort_offs"], mort_offs)

    if report_run.loc['run_dse', 'Run']:
        dse_sire = pd.concat([reports["dse_sire"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        dse_dams = pd.concat([reports["dse_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        dse_offs = pd.concat([reports["dse_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_dse_sire"] = rfun.f_append_dfs(stacked_reports["stacked_dse_sire"], dse_sire)
        stacked_reports["stacked_dse_dams"] = rfun.f_append_dfs(stacked_reports["stacked_dse_dams"], dse_dams)
        stacked_reports["stacked_dse_offs"] = rfun.f_append_dfs(stacked_reports["stacked_dse_offs"], dse_offs)

        dse1_sire = pd.concat([reports["dse1_sire"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        dse1_dams = pd.concat([reports["dse1_dams"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        dse1_offs = pd.concat([reports["dse1_offs"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_dse1_sire"] = rfun.f_append_dfs(stacked_reports["stacked_dse1_sire"], dse1_sire)
        stacked_reports["stacked_dse1_dams"] = rfun.f_append_dfs(stacked_reports["stacked_dse1_dams"], dse1_dams)
        stacked_reports["stacked_dse1_offs"] = rfun.f_append_dfs(stacked_reports["stacked_dse1_offs"], dse1_offs)

    if report_run.loc['run_grnfoo', 'Run']:
        grnfoo = pd.concat([reports["grnfoo"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_grnfoo"] = rfun.f_append_dfs(stacked_reports["stacked_grnfoo"], grnfoo)

    if report_run.loc['run_pgr', 'Run']:
        pgr = pd.concat([reports["pgr"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_pgr"] = rfun.f_append_dfs(stacked_reports["stacked_pgr"], pgr)

    if report_run.loc['run_dryfoo', 'Run']:
        dryfoo = pd.concat([reports["dryfoo"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_dryfoo"] = rfun.f_append_dfs(stacked_reports["stacked_dryfoo"], dryfoo)

    if report_run.loc['run_napfoo', 'Run']:
        napfoo = pd.concat([reports["napfoo"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_napfoo"] = rfun.f_append_dfs(stacked_reports["stacked_napfoo"], napfoo)

    if report_run.loc['run_grncon', 'Run']:
        grncon = pd.concat([reports["grncon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_grncon"] = rfun.f_append_dfs(stacked_reports["stacked_grncon"], grncon)

    if report_run.loc['run_drycon', 'Run']:
        drycon = pd.concat([reports["drycon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_drycon"] = rfun.f_append_dfs(stacked_reports["stacked_drycon"], drycon)

    if report_run.loc['run_grnnv', 'Run']:
        grnnv = pd.concat([reports["grnnv"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_grnnv"] = rfun.f_append_dfs(stacked_reports["stacked_grnnv"], grnnv)

    if report_run.loc['run_grndmd', 'Run']:
        grndmd = pd.concat([reports["grndmd"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_grndmd"] = rfun.f_append_dfs(stacked_reports["stacked_grndmd"], grndmd)

    if report_run.loc['run_avegrnfoo', 'Run']:
        grnfoo = pd.concat([reports["avegrnfoo"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_avegrnfoo"] = rfun.f_append_dfs(stacked_reports["stacked_avegrnfoo"], grnfoo)

    if report_run.loc['run_drynv', 'Run']:
        drynv = pd.concat([reports["drynv"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_drynv"] = rfun.f_append_dfs(stacked_reports["stacked_drynv"], drynv)

    if report_run.loc['run_drydmd', 'Run']:
        drydmd = pd.concat([reports["drydmd"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_drydmd"] = rfun.f_append_dfs(stacked_reports["stacked_drydmd"], drydmd)

    if report_run.loc['run_avedryfoo', 'Run']:
        dryfoo = pd.concat([reports["avedryfoo"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_avedryfoo"] = rfun.f_append_dfs(stacked_reports["stacked_avedryfoo"], dryfoo)

    if report_run.loc['run_napcon', 'Run']:
        napcon = pd.concat([reports["napcon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_napcon"] = rfun.f_append_dfs(stacked_reports["stacked_napcon"], napcon)

    if report_run.loc['run_poccon', 'Run']:
        poccon = pd.concat([reports["poccon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_poccon"] = rfun.f_append_dfs(stacked_reports["stacked_poccon"], poccon)

    if report_run.loc['run_supcon', 'Run']:
        supcon = pd.concat([reports["supcon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_supcon"] = rfun.f_append_dfs(stacked_reports["stacked_supcon"], supcon)

    if report_run.loc['run_stubcon', 'Run']:
        stubcon = pd.concat([reports["stubcon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_stubcon"] = rfun.f_append_dfs(stacked_reports["stacked_stubcon"], stubcon)

    if report_run.loc['run_cropcon', 'Run']:
        cropcon = pd.concat([reports["cropcon"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cropcon"] = rfun.f_append_dfs(stacked_reports["stacked_cropcon"], cropcon)

    if report_run.loc['run_cropcon', 'Run']:
        cropcon_available = pd.concat([reports["cropcon_available"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_cropcon_available"] = rfun.f_append_dfs(stacked_reports["stacked_cropcon_available"], cropcon_available)

    if report_run.loc['run_mvf', 'Run']:
        mvf = pd.concat([reports["mvf"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_mvf"] = rfun.f_append_dfs(stacked_reports["stacked_mvf"], mvf)
    
    if report_run.loc['run_pasture_area', 'Run']:
        pasture_area = pd.concat([reports["pasture_area"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_pasture_area"] = rfun.f_append_dfs(stacked_reports["stacked_pasture_area"], pasture_area)

    if report_run.loc['run_stocking_rate', 'Run']:
        stocking_rate = pd.concat([reports["stocking_rate"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_stocking_rate"] = rfun.f_append_dfs(stacked_reports["stacked_stocking_rate"], stocking_rate)

    if report_run.loc['run_legume', 'Run']:
        legume = pd.concat([reports["legume"]], keys=[trial_name], names=['Trial'])  # add trial name as index level
        stacked_reports["stacked_legume"] = rfun.f_append_dfs(stacked_reports["stacked_legume"], legume)

    return stacked_reports

def f_save_reports(report_run, reports, processor):
    ####################################
    #run between trial reports and save#
    ####################################
    print("Writing to Excel")
    ##first check that Excel is not open (Microsoft puts a lock on files, so they can't be updated from elsewhere while open)
    report_file_path = relativeFile.find(__file__, "../../Output", "Report{0}.xlsx".format(processor))
    if os.path.isfile(report_file_path): #to check if report.xl exists
        while True:   # repeat until the try statement succeeds
            try:
                myfile = open(report_file_path,"w") # chucks an error if Excel file is open
                break                             # exit the loop
            except IOError:
                input("Could not open file! Please close Excel. Press Enter to retry.")
                # restart the loop

    ## Create a Pandas Excel writer using XlsxWriter as the engine. used to write to multiple sheets in Excel
    writer = pd.ExcelWriter(report_file_path, engine='xlsxwriter')

    ##make empty df to store row and col index settings. Used when combining multiple report.xl
    df_settings = pd.DataFrame(columns=['index', 'cols'])

    ##write to excel
    ###determine the method of reporting rows and columns that are all zeros
    ### mode 0: df straight into Excel
    ### mode 1: df into Excel - collapsing rows/cols that contain only 0's.
    ### mode 2: df into Excel - removing rows/cols that contain only 0's. This method make the writing process faster.
    try:
        xl_display_mode = int(sys.argv[4])  # reads in as string so need to convert to int, the script path is the first value.
    except IndexError:  # in case no arg passed to python
        xl_display_mode = 1 #default is to collapse rows/cols that are all 0's (ie they exist in Excel but are hidden)

    df_settings = rfun.f_df2xl(writer, reports["stacked_infeasible"], 'infeasible', df_settings, option=xl_display_mode)
    df_settings = rfun.f_df2xl(writer, reports["stacked_non_exist"],'Non-exist',df_settings,option=0,colstart=0)
    if report_run.loc['run_summary', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_summary"], 'summary', df_settings, option=xl_display_mode)
    if report_run.loc['run_areasum', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_areasum"], 'areasum', df_settings, option=xl_display_mode)
    if report_run.loc['run_cropsum', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cropsum"], 'cropsum', df_settings, option=xl_display_mode)
    if report_run.loc['run_profit', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_profit"], 'profit', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_qsz', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_qsz"], 'numbers_qsz', df_settings, option=xl_display_mode)
    if report_run.loc['run_croparea_qsz', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_croparea_qsz"], 'croparea_qsz', df_settings, option=xl_display_mode)
    if report_run.loc['run_pnl', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_pnl"], 'pnl', df_settings, option=xl_display_mode)
    if report_run.loc['run_wc', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_wc"], 'wc', df_settings, option=xl_display_mode)
    if report_run.loc['run_period_dates', 'Run']:
        fp_start_col = len(reports["stacked_season_nodes"].columns) + reports["stacked_season_nodes"].index.nlevels + 1
        dam_dvp_start_col = fp_start_col + len(reports["stacked_feed_periods"].columns) + reports["stacked_feed_periods"].index.nlevels + 1
        repro_start_col = dam_dvp_start_col + len(reports["stacked_dam_dvp_dates"].columns) + reports["stacked_dam_dvp_dates"].index.nlevels + 1
        offs_start_col = repro_start_col + len(reports["stacked_repro_dates"].columns) + reports["stacked_repro_dates"].index.nlevels + 1
        df_settings = rfun.f_df2xl(writer, reports["stacked_season_nodes"], 'period_dates', df_settings, option=0, colstart=0)
        df_settings = rfun.f_df2xl(writer, reports["stacked_feed_periods"], 'period_dates', df_settings, option=0, colstart=fp_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dam_dvp_dates"], 'period_dates', df_settings, option=0, colstart=dam_dvp_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_repro_dates"], 'period_dates', df_settings, option=0, colstart=repro_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_offs_dvp_dates"], 'period_dates', df_settings, option=0, colstart=offs_start_col)
    if report_run.loc['run_biomass_penalty', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_penalty"], 'biomass_penalty', df_settings, option=xl_display_mode)
    if report_run.loc['run_sowing_date', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_sowing_date"], 'sowing_date', df_settings, option=xl_display_mode)
    if report_run.loc['run_profitarea', 'Run']:
        plot = rfun.f_xy_graph(reports["stacked_profitarea"])
        plot.savefig('Output/profitarea_curve.png')
    if report_run.loc['run_feedbudget', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_feed"], 'feed budget', df_settings, option=xl_display_mode)
        df_settings = rfun.f_df2xl(writer, reports["stacked_feed2"], 'feed budget total', df_settings, option=xl_display_mode)
        df_settings = rfun.f_df2xl(writer, reports["stacked_grazing"], 'grazing summary', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_summary', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ewe_numbers_summary"], 'ewe_numbers_summary', df_settings, option=xl_display_mode)
        df_settings = rfun.f_df2xl(writer, reports["stacked_wethers_n_crossys_numbers_summary"], 'wethers_n_xb_numbers_summary', df_settings, option=xl_display_mode)
    if report_run.loc['run_emissions', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_emissions"], 'emissions', df_settings, option=xl_display_mode)
    if report_run.loc['run_saleprice', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_saleprice"], 'saleprice', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salegrid_dams"], 'salegrid_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_yatf', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salegrid_yatf"], 'salegrid_yatf', df_settings, option=xl_display_mode)
    if report_run.loc['run_salegrid_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salegrid_offs"], 'salegrid_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_saleage_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_saleage_offs"], 'saleage_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_saledate_offs', 'Run']:
        reports["stacked_saledate_offs"] = reports["stacked_saledate_offs"].astype(object)
        reports["stacked_saledate_offs"][reports["stacked_saledate_offs"]==np.datetime64('1970-01-01')] = 0
        df_settings = rfun.f_df2xl(writer, reports["stacked_saledate_offs"], 'saledate_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salevalue_offs"], 'salevalue_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salevalue_dams"], 'salevalue_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_salevalue_prog', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_salevalue_prog"], 'salevalue_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_woolvalue_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_woolvalue_offs"], 'woolvalue_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_woolvalue_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_woolvalue_dams"], 'woolvalue_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_cfw_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cfw_dams"], 'cfw_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_fd_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fd_dams"], 'fd_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_cfw_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cfw_offs"], 'cfw_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_fd_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fd_offs"], 'fd_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_wbe_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_wbe_dams"], 'wbe_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_wbe_cut_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_wbe_cut_dams"], 'wbe_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_fat_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fat_dams"], 'fat_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_fat_cut_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fat_cut_dams"], 'fat_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_muscle_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_muscle_dams"], 'muscle_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_muscle_cut_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_muscle_cut_dams"], 'muscle_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_viscera_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_viscera_dams"], 'viscera_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_viscera_cut_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_viscera_cut_dams"], 'viscera_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_wbe_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_wbe_offs"], 'wbe_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_lw_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_lw_dams"], 'lw_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ebw_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ebw_dams"], 'ebw_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ebw_cut_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ebw_cut_dams"], 'ebw_cut_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ebw_yatf', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ebw_yatf"], 'ebw_yatf', df_settings, option=xl_display_mode)
    if report_run.loc['run_cs_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cs_dams"], 'cs_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_fs_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fs_dams"], 'fs_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_nv_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_nv_dams"], 'nv_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_ebw_prog', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ebw_prog"], 'ebw_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_ebw_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_ebw_offs"], 'ebw_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_cs_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cs_offs"], 'cs_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_fs_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_fs_offs"], 'fs_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_nv_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_nv_offs"], 'nv_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_lamb_survival', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_lamb_survival"], 'lamb_survival', df_settings, option=xl_display_mode)
    if report_run.loc['run_weanper', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_weanper"], 'wean_per', df_settings, option=xl_display_mode)
    if report_run.loc['run_scanper', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_scanper"], 'scan_per', df_settings, option=xl_display_mode)
    if report_run.loc['run_dry_propn', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_dry_propn"], 'dry_propn', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_mei_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_daily_mei_dams"], 'daily_mei_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_pi_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_daily_pi_dams"], 'daily_pi_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_mei_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_daily_mei_offs"], 'daily_mei_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_daily_pi_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_daily_pi_offs"], 'daily_pi_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_dams"], 'numbers_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_dams_p', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_dams_p"], 'numbers_dams_p', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_prog', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_prog"], 'numbers_prog', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_offs"], 'numbers_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_numbers_offs_p', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_numbers_offs_p"], 'numbers_offs_p', df_settings, option=xl_display_mode)
    if report_run.loc['run_mort_dams', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_mort_dams"], 'mort_dams', df_settings, option=xl_display_mode)
    if report_run.loc['run_mort_offs', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_mort_offs"], 'mort_offs', df_settings, option=xl_display_mode)
    if report_run.loc['run_dse', 'Run']:
        dams_start_col = len(reports["stacked_dse_sire"].columns) + reports["stacked_dse_sire"].index.nlevels + 1
        offs_start_col = dams_start_col + len(reports["stacked_dse_dams"].columns) + reports["stacked_dse_dams"].index.nlevels + 1
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse_sire"], 'dse_wt', df_settings, option=0, colstart=0)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse_dams"], 'dse_wt', df_settings, option=0, colstart=dams_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse_offs"], 'dse_wt', df_settings, option=0, colstart=offs_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse1_sire"], 'dse_mei', df_settings, option=0, colstart=0)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse1_dams"], 'dse_mei', df_settings, option=0, colstart=dams_start_col)
        df_settings = rfun.f_df2xl(writer, reports["stacked_dse1_offs"], 'dse_mei', df_settings, option=0, colstart=offs_start_col)
    if report_run.loc['run_pgr', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_pgr"], 'Total pg', df_settings, option=xl_display_mode)
    if report_run.loc['run_grnfoo', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_grnfoo"], 'grnfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_dryfoo', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_dryfoo"], 'dryfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_napfoo', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_napfoo"], 'napfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_grnnv', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_grnnv"], 'grnnv', df_settings, option=xl_display_mode)
    if report_run.loc['run_grndmd', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_grndmd"], 'grndmd', df_settings, option=xl_display_mode)
    if report_run.loc['run_avegrnfoo', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_avegrnfoo"], 'avegrnfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_drynv', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_drynv"], 'drynv', df_settings, option=xl_display_mode)
    if report_run.loc['run_drydmd', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_drydmd"], 'drydmd', df_settings, option=xl_display_mode)
    if report_run.loc['run_avedryfoo', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_avedryfoo"], 'avedryfoo', df_settings, option=xl_display_mode)
    if report_run.loc['run_grncon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_grncon"], 'grncon', df_settings, option=xl_display_mode)
    if report_run.loc['run_drycon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_drycon"], 'drycon', df_settings, option=xl_display_mode)
    if report_run.loc['run_napcon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_napcon"], 'napcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_poccon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_poccon"], 'poccon', df_settings, option=xl_display_mode)
    if report_run.loc['run_supcon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_supcon"], 'supcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_stubcon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_stubcon"], 'stubcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_cropcon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cropcon"], 'cropcon', df_settings, option=xl_display_mode)
    if report_run.loc['run_cropcon', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_cropcon_available"], 'cropcon_avail', df_settings, option=xl_display_mode)
    if report_run.loc['run_mvf', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_mvf"], 'mvf', df_settings, option=xl_display_mode)
    if report_run.loc['run_pasture_area', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_pasture_area"], 'pasture_area_analysis', df_settings, option=xl_display_mode)
    if report_run.loc['run_stocking_rate', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_stocking_rate"], 'stocking_rate_analysis', df_settings, option=xl_display_mode)
    if report_run.loc['run_legume', 'Run']:
        df_settings = rfun.f_df2xl(writer, reports["stacked_legume"], 'lupin_analysis', df_settings, option=xl_display_mode)


    df_settings.to_excel(writer, 'df_settings')
    writer.close()

    print("Report complete. Processor: {0}".format(processor))
