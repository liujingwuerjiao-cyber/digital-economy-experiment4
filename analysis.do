clear all
set more off

import delimited "data_exp4_auction.csv", clear varnames(1) numericcols(_all)

* Verify Overbid definition for winners only
gen overbid_check = cond(win==1, max(bid-quality, 0), 0)

* OLS with heteroskedasticity-robust SE
reg price quality, robust

* Scenario comparison for Winner's Curse metric
ttest overbid, by(scenario)

* Graphs
graph box overbid, over(scenario) name(g_box, replace)
graph export "scenario_box.png", name(g_box) replace

graph bar (mean) overbid, over(scenario) blabel(bar) name(g_bar, replace)
graph export "scenario_bar.png", name(g_bar) replace
