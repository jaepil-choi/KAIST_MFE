* SAS 3: Calculate the monthly value-weighted portfolio returns ******************;

data portfolio1;
 set My_lib.assignment2_data;
 eretadj_ME = (eretadj*100)*ME_lag1; ** value weighted 조정초과수익률 구함. ME lag 1을 쓰는 것을 주목. ; 
run;

proc sort data = portfolio1;
 by date p1 p2;
run;

proc means data = portfolio1 sum noprint;
 by date p1 p2;
 var eretadj_ME ME_lag1;
 output out = portfolio2 (drop = _FREQ_ _TYPE_) sum = / autoname;
run;

data portfolio2;
 set portfolio2;
 vw_pret = eretadj_ME_Sum/ME_lag1_Sum;
 keep date p1 p2 vw_pret;
run;

* Calculate the return difference between the fifth (p2 = 5) and first (p2 = 1) BM sorted portfolios within each Size sorted portfolio;
data portfolio3;
 set portfolio2(where = (p2 in (1,5))); ** 사이즈 내에서 BM1과 BM5 차이 ;
run;

proc sort data = portfolio3;
 by date p1 p2;
run;

proc transpose data = portfolio3 out = portfolio4; ** pivot table해서 index date, column vw_pret ;
 by date p1;
 id p2;
 var vw_pret;
run;

data portfolio4;
 set portfolio4;
 p2 = 51;
 vw_pret = _5 - _1;
 keep date p1 p2 vw_pret; ** 똑같이 요것만 남긴다. 테이블 port 2 아래로 append 하기 위해 ;
run;

* Append the two datasets;
data portfolio5;
 set portfolio2 portfolio4;
 year = year(date);
 month = month(date);
run;

proc sort data = portfolio5;
 by year month date p1 p2;
run;

* SAS4: Add FF-3 factors to the portfolio return data set **********************************;

* Load monthly factors data;
proc import out = factors 
 datafile = "&my_directory\factors_monthly.csv" 
 dbms = csv replace; ** 팩터 데이터는 그냥 파마프렌치에서 불러온다. ;
run;

* Be careful. The "date" column in the monthly factor data set is NOT in a date format. We will merge the factor data set with the portfolio return data set using "year" and "month" variables as the key variables;
data factors;
 set factors;

 * convert factors from decimal to percent;
 mktrf = mktrf*100;
 smb = smb*100;
 hml = hml*100;
 keep year month mktrf smb hml;
run;

* Merge;
data portfolio6; ** 팩터가 더 길지만 left join이라 길이가 변하지 않는다. ; 
 merge portfolio5 (in = a) factors (in = b);
 by year month;
 if a;
run;

* SAS 5: Test if the BM5 portfolio has a higher expected return than BM1 portfolio within each size group using time-series regressions **********************************;

proc sort data = portfolio6;
 by p1 p2 date;
run;

* To perform Newey-West standard error correction, PROC MODEL is run specifying the GMM estimation method in the FIT statement. KERNEL=(BART, L+1, 0) is also specified which requests the Bartlett kernel with a lag length of L. The VARDEF(specify the denominator for computing variances and covariances)=n option is specified to be consistent with the original Newey-West formula;

* Calculate the FF3 alpha of the long-short portfolio (p2=51);
proc model data = portfolio6 (where = (p2 = 51)); ** BM5 - BM1 이 51;
 by p1;
 exog mktrf hml smb; ** FF3 팩터들을 regression ;
 instruments _exog_;
 vw_pret = a + b1*mktrf + b2*hml + b3*smb;
 fit vw_pret / gmm kernel = (bart, 7, 0) vardef = n; ** gmm kernel이 newey west를 쓰는 것이다. 7이 max lag ;
 ** newey west 쓰는 이유: time series에선 error term의 autocorrelation이 있을 수 있기 때문에 이를 보정하기 위해 ;
 ** 안쓰면, autocorrelation에 의해 standard error가 작게 나올 수 있다. ;
 ** 즉, 원래는 유의미하지 않은데, autocorrelation으로 인해 유의미하게 나올 수 있다. ;
 ** SAS에선 7 = max_lag + 1이다. python에선 그냥 max_lag를 넣어야 한다.;
 ods output parameterestimates = table3;
 quit;
ods exclude none;

** 이 아래부터는 알아서 python으로 CAPM alpha, FF3 alpha를 각 size group 내의 BM5 - BM1에 대해 구해봐야 한다. ;