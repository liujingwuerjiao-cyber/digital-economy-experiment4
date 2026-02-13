' EViews program: second-price auction analysis
close @all
wfcreate auction_wf u 1 1
import(type=csv) "data_exp4_auction.csv"

' Overbid check variable
series Overbid_check = @recode(Win=1,@max(Bid-Quality,0),0)

' OLS with White robust covariance
equation eq_price.ls(cov=white) Price c Quality

' Basic scenario summaries
smpl if scenario=1
scalar mean_overbid_s1 = @mean(Overbid)
smpl if scenario=2
scalar mean_overbid_s2 = @mean(Overbid)
smpl @all

' Optional charts in EViews UI:
' series Overbid
' freeze(g_box) Overbid.boxplot(scenario)
' freeze(g_bar) scenario.bar(Overbid)
