/*
cap log close
clear all
set more off
pause on

cd "C:\Users\User\Desktop\5-2학기\applied microeconometrics\proposal_appliedmicro"
log using proposal_appliedmicro.log, replace

import excel subsidy_case.xlsx, first clear
codebook, c

asdoc sum

gen lcase = ln(case)
gen gyeonggi = (city=="gyeonggi")
gen after = (month>=2 & year==2021)
gen treatment = gyeonggi*after  //same as subsidy


reg case subsidy gyeonggi after, robust
outreg2 using subsidy(1), replace word bdec(3) sdec(3) adjr2 ///
	addtext(Logarithm on Dependent Variable?, no, Clustered Standard Error?, no) 

reg case subsidy gyeonggi after, vce(cl gyeonggi)
outreg2 using subsidy(1), append word bdec(3) sdec(3) adjr2 ///
	addtext(Logarithm on Dependent Variable?, no, Clustered Standard Error?, yes) 

reg lcase subsidy gyeonggi after, vce(cl gyeonggi)
outreg2 using subsidy(1), append word bdec(3) sdec(3) adjr2 ///
	addtext(Logarithm on Dependent Variable?, yes, Clustered Standard Error?, yes) 
	
reg csi_living subsidy gyeonggi after, robust
outreg2 using subsidy(1), append word bdec(3) sdec(3) adjr2 ///
	addtext(Logarithm on Dependent Variable?, no, Clustered Standard Error?, no) 

reg csi_living subsidy gyeonggi after, vce(cl gyeonggi)
outreg2 using subsidy(1), append word bdec(3) sdec(3) adjr2 ///
	addtext(Logarithm on Dependent Variable?, no, Clustered Standard Error?, yes) 
*/
	
	
	
******event study*******
clear
//ssc install eventdd
// ssc install matsort
cd "C:\Users\User\Desktop\5-2학기\applied microeconometrics\proposal_appliedmicro\코로나데이터"
import excel stimulus_corona_2020인천경기.xlsx, first clear

drop if size == "other"  // 논문에서 사용하지 않은 도시들 제외

gen timeToTreat = week - event_week
encode size, generate(size2)
encode local, generate(local2)
encode state, generate(state2)

gen gyeonggi = (state=="gyeonggi")
gen after = (week>=11)
gen treatment = gyeonggi*after 

codebook, c
//asdoc sum


*** covid-19 case trend depending on subsidy size 
preserve
 collapse week_case_per_1m, by(size2 week)
 twoway line week_case_per_1m week, sort(week) by(size) ytitle("Weekly Average Covid-19 Case") xsize(2) ysize(1.5) 
restore

preserve
 collapse week_case_per_1m, by(state2 week)
 twoway line week_case_per_1m week, sort(week) by(state) ytitle("Weekly Average Covid-19 Case") xsize(2.3) ysize(1.5) 
restore

/* //위 코드로 하면 노가다 불필요
preserve
 keep if size == "100-150"
 egen sum_case = mean(week_case_per_1m), by(week)
 twoway connected sum_case week, sort(week) title("Weekly Average Covid-19 Confirmed Case for 100-150 districts") ytitle("Weekly Average Covid-19 Case") xsize(2) ysize(1.3) 
restore

preserve
 keep if size == "200-250"
 egen sum_case = mean(week_case_per_1m), by(week)
 twoway connected sum_case week, sort(week) title("Weekly Average Covid-19 Confirmed Case for 200-250 districts") ytitle("Weekly Average Covid-19 Case") xsize(2) ysize(1.3) 
restore

preserve
 keep if size == "300-350"
 egen sum_case = mean(week_case_per_1m), by(week)
 twoway connected sum_case week, sort(week) title("Weekly Average Covid-19 Confirmed Case for 300-350 districts") ytitle("Weekly Average Covid-19 Case") xsize(2) ysize(1.3) 
restore

preserve
 keep if size == "none"
 egen sum_case = mean(week_case_per_1m), by(week)
 twoway connected sum_case week, sort(week) title("Weekly Average Covid-19 Confirmed Case for Incheon districts") ytitle("Weekly Average Covid-19 Case") xsize(2) ysize(1.3) 
restore
*/

*** DD regression model
reg week_case_per_1m gyeonggi, r // metro level
outreg2 using regulardd, replace word bdec(3) sdec(3) 2aster ///
	addtext(Level of Anlysis, metro city, Local district effects?, -, Week effects?, -, Clustered Standard Error?, no) 

reg week_case_per_1m gyeonggi after, r 
outreg2 using regulardd, append word bdec(3) sdec(3) 2aster ///
	addtext(Level of Anlysis, metro city, Local district effects?, -, Week effects?, -, Clustered Standard Error?, no) 

reg week_case_per_1m gyeonggi after treatment, r
outreg2 using regulardd, append word bdec(3) sdec(3) 2aster ///
	addtext(Level of Anlysis, metro city, Local district effects?, -, Week effects?, -, Clustered Standard Error?, no) 

*local level 
reg week_case_per_1m treatment i.local2 i.week, vce(cl local2) 
outreg2 using regulardd, append word bdec(3) sdec(3) 2aster keep(treatment) ///
	addtext(Level of Anlysis, local district, Local district effects?, yes, Week effects?, yes, Clustered Standard Error?, yes) 

*local specific time trend
reg week_case_per_1m treatment i.local2 i.week i.local2#c.week, vce(cl local2) 
outreg2 using regulardd, append word bdec(3) sdec(3) 2aster keep(treatment) ///
	addtext(Level of Anlysis, local district, Local district effects?, yes, Week effects?, yes, Clustered Standard Error?, yes, Local specific time trend?, yes) 

/*
*** OLS estimate
eventdd week_case_per_1m i.week i.local2, timevar(timeToTreat) method( , cluster(local2)) graph_op(ytitle("Weekly Covid19 Confirmed Case per 1M Residents"))

eventdd week_case_per_1m i.week i.local2, timevar(timeToTreat) method(ols , cluster(local2)) graph_op(ytitle("Weekly Covid19 Confirmed Case per 1M Residents"))

* local specific time trend
eventdd week_case_per_1m i.week i.local2 i.local2#c.week , timevar(timeToTreat) method(ols , cluster(local2)) graph_op(ytitle("Weekly Covid19 Confirmed Case per 1M Residents"))

* heterogeneity according to stimulus check size
eventdd week_case_per_1m i.week i.local2 i.size2 , timevar(timeToTreat) method(ols , cluster(local2)) graph_op(ytitle("Weekly Covid19 Confirmed Case per 1M Residents"))
*/


*****Event Study*****
*** fixed effects 사용
preserve
 xtset local2 week 
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title("Effect of Stimulus Check on Covid-19 Case") ytitle("Covid-19 Case per 1M Residents"))
restore

*** heterogeneity according to stimulus check size ***
* 100-150
preserve
 drop if size == "300-350"
 drop if size == "200-250"
 xtset local2 week 
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title("Effect of Stimulus Check on Covid-19 Case: 100-150") ytitle("Covid-19 Case per 1M Residents"))
restore

* 200-250
preserve
 drop if size == "300-350"
 drop if size == "100-150"
 xtset local2 week 
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title("Effect of Stimulus Check on Covid-19 Case: 200-250") ytitle("Covid-19 Case per 1M Residents"))
restore

* 300-350
preserve
 drop if size == "100-150"
 drop if size == "200-250"
 xtset local2 week 
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title("Effect of Stimulus Check on Covid-19 Case: 300-350") ytitle("Covid-19 Case per 1M Residents"))
restore



*** Robust checks ***
* inner-ring districts
preserve 
 xtset local2 week 
 keep if local=="g4"|local=="g5"|local=="g8"|local=="g9"|local=="g11"|local=="g14"|local=="g15"|local=="g17"|local=="g20"|local=="g22"|local=="g25"|local=="g29"|state=="incheon"
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title(" Inner-ring districts vs Incheon")ytitle("Covid-19 Case per 1M Residents"))
restore

*  outer-ring districts 
preserve 
 xtset local2 week 
 keep if local=="g7"|local=="g10"|local=="g12"|local=="g13"|local=="g16"|local=="g18"|local=="g21"|local=="g23"|local=="g26"|local=="g28"|local=="g30"|local=="g31"|state=="incheon"
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title("Outer-ring districts vs Incheon")ytitle("Covid-19 Case per 1M Residents"))
restore


*comparing adjacent local districs
preserve 
 xtset local2 week 
 keep if local=="g5"|local=="g13"|local=="g14"|local=="g15"|local=="i5"|local=="i6"|local=="i7"
 eventdd week_case_per_1m i.week , timevar(timeToTreat) method(fe , cluster(local2)) graph_op(title(" Adjacent Local Districts Comparison")ytitle("Covid-19 Case per 1M Residents"))
restore

