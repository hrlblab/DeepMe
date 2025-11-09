clear variables;
clear all;
clc;
% read litterfall and soilT files
% sth = xlsread('NCEP_HF_soilT_19892010_hourly_obsMerged_cont_heat.csv','A:A');  % Heated deg K
% stc = xlsread('NCEP_HF_soilT_19892010_hourly_obsMerged_cont_heat.csv','B:B');  % CONTROL deg K
M = readmatrix('NCEP_HF_soilT_19892010_hourly_obsMerged_cont_heat.csv');
sth = M(:,1);  % Heated deg K
stc = M(:,2);  % CONTROL deg K
st=sth; % define control/heat trt
% lfh = xlsread('hf008-09-litterfall-x-1989-2010 hourly_cont_heat.csv','C:C'); % heated mg C m-2 h-1
% lfc = xlsread('hf008-09-litterfall-x-1989-2010 hourly_cont_heat.csv','B:B'); % control mg C m-2 h-1
M = readmatrix('hf008-09-litterfall-x-1989-2010 hourly_cont_heat.csv');
lfc = M(:,2);  % control mg C m-2 h-1
lfh = M(:,3);  % heated mg C m-2 h-1
lf=lfh; % define control/heat trt
% smc = xlsread('NARR_HF_soilM_19892010_hourly.csv','B:B'); % control Mpa
% smh = xlsread('NARR_HF_soilM_19892010_hourly.csv','D:D'); % heated Mpa
M = readmatrix('NARR_HF_soilM_19892010_hourly.csv');
smc = M(:,2);
smh = M(:,4);
sm=smh; % define control/heat trt  % 12/2017 jianwei added moisture
rho = 1; % cm-3 g soil-1
lf = lf/(rho*10^5); % mg C g soil-1 h-1
% SOC DOC input 
IS = lf*0.98; % 98% lf is SOC input % unit mg C g soil-1 h-1
ID = lf*0.02; % 2% lf is DOC input % unit mg C g soil-1 h-1
%%%% from Control get best ECref, Initial pool sizes %%%%
%cont = xlsread('da_parameters_summary_mean_cont.csv','B:B');  % achieved from Control after DA
          % next lines 26 for cont1 applied in the heated trt
%%%% from Control get best ECref, Initial pool sizes END %%%%
% Forcing data
Temp=st;
IP = IS;
ID = ID;
TempK = Temp;
Tref = 293; % 20 degree C
ECref = 0.31; % a 
m = -0.008; % b
Vmax_ref = 1;
Vmax_uptake_ref = 0.01;
Km_ref = 250;
Km_uptake_ref = 0.26;
mRref = 0.00028; % same as rB: MBC turnover rate    c 
Ea = 47;
Ea_uptake = 47;
Ea_K = 30;
Ea_Kuptake = 30;
R = 0.008314;
EaKM = 30;
EaKP = 30;
EaKD = 30;
EaVM = 47;
EaVD = 47;
EaVP = 45; 
EaKads = 5;
EaKdes = 20;
EamR = 20;
fD = 0.5; % fraction of MBC to SOC; sane as aBS in AWB;    d
gD = 0.5; % fraction of POC to DOC;   e  
pEM =0.01; % fraction of mR allocated to enzyme production
pEP = 0.01;
rEM = 0.001; % loss rate, mg C mg-1 h-1
rEP = 0.001;
Qmax = 1.7;
KMref = 250;
VMref = 1;
KPref = 50;
VPref = 2.5;
KDref = 0.26;
VDref = 0.0005;
Kadsref = 0.006;
Kdesref = 0.001;
% Temporary pools and flux
POCh=NaN(192840,1);
POCd=NaN(8035,1);
MOCh=NaN(192840,1);
MOCd=NaN(8035,1);
QOCh=NaN(192840,1);
QOCd=NaN(8035,1);
SOCh=NaN(192840,1);
SOCd=NaN(8035,1);
MBCh=NaN(192840,1);
MBCd=NaN(8035,1);
DOCh=NaN(192840,1);
DOCd=NaN(8035,1);
EPh=NaN(192840,1);
EPd=NaN(8035,1);
EMh=NaN(192840,1);
EMd=NaN(8035,1);
ENCh=NaN(192840,1);
ENCd=NaN(8035,1);
CO2h=NaN(192840,1);
CO2d=NaN(8035,1);

POCh_obs=NaN(192840,1);
POCd_obs=NaN(8035,1);
MOCh_obs=NaN(192840,1);
MOCd_obs=NaN(8035,1);
QOCh_obs=NaN(192840,1);
QOCd_obs=NaN(8035,1);
SOCh_obs=NaN(192840,1);
SOCd_obs=NaN(8035,1);
MBCh_obs=NaN(192840,1);
MBCd_obs=NaN(8035,1);
DOCh_obs=NaN(192840,1);
DOCd_obs=NaN(8035,1);
EPh_obs=NaN(192840,1);
EPd_obs=NaN(8035,1);
EMh_obs=NaN(192840,1);
EMd_obs=NaN(8035,1);
ENCh_obs=NaN(192840,1);
ENCd_obs=NaN(8035,1);
CO2h_obs=NaN(192840,1);
CO2d_obs=NaN(8035,1);

POCh_sim=NaN(192840,1);
POCd_sim=NaN(8035,1);
MOCh_sim=NaN(192840,1);
MOCd_sim=NaN(8035,1);
QOCh_sim=NaN(192840,1);
QOCd_sim=NaN(8035,1);
SOCh_sim=NaN(192840,1);
SOCd_sim=NaN(8035,1);
MBCh_sim=NaN(192840,1);
MBCd_sim=NaN(8035,1);
DOCh_sim=NaN(192840,1);
DOCd_sim=NaN(8035,1);
EPh_sim=NaN(192840,1);
EPd_sim=NaN(8035,1);
EMh_sim=NaN(192840,1);
EMd_sim=NaN(8035,1);
ENCh_sim=NaN(192840,1);
ENCd_sim=NaN(8035,1);
CO2h_sim=NaN(192840,1);
CO2d_sim=NaN(8035,1);
% Initial pool sizes  
POC = 8.7;  % alternative     f
%POC = 8.7; % initial SOC 48-49
MOC = 40;  % alternative     g
%MOC = 40; % initial SOC 48-49
QOC = 0.8;  % alternative     h
%QOC = 0.8; % initial SOC 48-49
SOC = POC+MOC+QOC;
MBC = 0.25; %                 i
%MBC = 0.25; %                
DOC = 0.15; %                 j
%DOC = 0.15; %                 
EP = 0.0007; %               k
%EP = 0.0007; %               
EM = 0.0007; %               l
%EM = 0.0007; %               
ENC = EP+EM; 
CO2 = 0.00016;
%%% simulation starts
for z = 1:192840;
EC = (ECref+m*(TempK(z)-Tref));
mR = mRref*exp((-EamR/R)*(1/TempK(z)-1/Tref));
KP = KPref*exp((-EaKP/R)*(1/TempK(z)-1/Tref));
%fSWP  % 12/2017 added
% -0.033 Mpa field capacity SWP; SWPmin = -13.86 Mpa; exponent = 1.20
if sm(z) < -13.86;
    fSWP=0;
        elseif sm(z) < -0.033;
            fSWP = 1-(log(sm(z)/(-0.033))/log((-13.86)/(-0.033))).^1.2;
        else
            fSWP = 1;
end
% fSWP
  fSWP_A2D = abs(sm(z)).^4/(abs(sm(z)).^4+abs(-0.4)*4); % -0.4 Mpa; exponent = 4
VP = VPref*exp((-EaVP/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KM = KMref*exp((-EaKM/R)*(1/TempK(z)-1/Tref));
VM = VMref*exp((-EaVM/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KD = KDref*exp((-EaKD/R)*(1/TempK(z)-1/Tref));
VD = VDref*exp((-EaVD/R)*(1/TempK(z)-1/Tref));
Kads = Kadsref*exp((-EaKads/R)*(1/TempK(z)-1/Tref));
Kdes = Kdesref*exp((-EaKdes/R)*(1/TempK(z)-1/Tref));
KBA = Kads/Kdes;
  F1 = (VD+mR)*DOC*MBC/(EC*(KD+DOC));
  F2 = VP*EP*POC/(KP+POC);
  F3 = VM*EM*MOC/(KM+MOC);
  F4 = (1/EC-1)*VD*MBC*DOC/(KD+DOC);
  F5 = (1/EC-1)*mR*MBC*DOC/(KD+DOC);
  F6 = Kads*DOC*(1-QOC/Qmax);
  F7 = Kdes*QOC/Qmax;
  F8 = mR*MBC*(1-pEP-pEM); % *fSWP_A2D;
  F9ep = pEP*mR*MBC;
  F9em = pEM*mR*MBC;
  F10ep = rEP*EP;
  F10em = rEM*EM;
  POC = POC + IP(z) +(1-gD)*F8 - F2;
  MOC = MOC + (1-fD)*F2 - F3;
  QOC = QOC + F6 - F7;
  SOC = POC + MOC +QOC;
  MBC = MBC + F1 - (F4+F5) - F8 - (F9ep+F9em);
  DOC = DOC + ID(z) + fD*F2 + gD*F8 + F3 + (F10ep+F10em) - F1 - (F6-F7);
  EP = EP + F9ep - F10ep;
  EM = EM + F9em - F10em;
  ENC = EP + EM;
  CO2 = F4 + F5;
 % hourly
    POCh(z)=POC;
    MOCh(z)=MOC;
    QOCh(z)=QOC;
    SOCh(z)=SOC;
    MBCh(z)=MBC;
    DOCh(z)=DOC;
    EPh(z)=EP;
    EMh(z)=EM;
    ENCh(z)=ENC;
    CO2h(z)=CO2;
end
% daily
    for z=1:8035;
    POCd(z)=mean(POCh(((z-1)*24+1):(z*24)));
    MOCd(z)=mean(MOCh(((z-1)*24+1):(z*24)));
    QOCd(z)=mean(QOCh(((z-1)*24+1):(z*24)));
    SOCd(z)=mean(SOCh(((z-1)*24+1):(z*24)));
    MBCd(z)=mean(MBCh(((z-1)*24+1):(z*24)));
    DOCd(z)=mean(DOCh(((z-1)*24+1):(z*24)));
    EPd(z)=mean(EPh(((z-1)*24+1):(z*24)));
    EMd(z)=mean(EMh(((z-1)*24+1):(z*24)));
    ENCd(z)=mean(ENCh(((z-1)*24+1):(z*24)));    
    CO2d(z)=sum(CO2h(((z-1)*24+1):(z*24)));
    end   
% Simulation ends
% Simulation output
POCd_sim=POCd; 
MOCd_sim=MOCd; 
QOCd_sim=QOCd; 
SOCd_sim=SOCd; 
MBCd_sim=MBCd; 
DOCd_sim=DOCd; 
EPd_sim=EPd; 
EMd_sim=EMd; 
ENCd_sim=ENCd;
CO2d_sim=CO2d;
CO2h_sim=CO2h;
% Simulation output ends

figure(1)
% simulation with time
y = CO2h_sim;
ax = subplot(2,3,1);
plot(ax,1:192840,y,'o');
xlabel('Time: hour');
ylabel('Sim');
xlim([1,192840]);
ylim([-0.000,0.002]);
title('Ori Hourly CO2, mg C g soil-1 hr-1');
%CO2 daily
y = CO2d_sim;
ax = subplot(2,3,2);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([1,8035]);
ylim([-0.00,0.04]);
title('Daily CO2, mg C g soil-1 day-1');
%SOC
y = SOCd_sim;
ax = subplot(2,3,3);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,60]);
title('SOC, mg C g soil-1');
%MBC
y = MBCd_sim;
ax = subplot(2,3,4);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,4]);
title('MBC, mg C g soil-1');
%DOC
y = DOCd_sim;
ax = subplot(2,3,5);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,0.5]);
title('DOC, mg C g soil-1');
%ENC
y = ENCd_sim;
ax = subplot(2,3,6);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,0.04]);
title('ENC, mg C g soil-1');

% Observations import
% soc mbc starts
M = readmatrix('hf_socmbc_obs_cont_heat.csv');
SOCd_obs = M(:,2);  % mg C g-1 soil; Control
SOCd_obs = M(:,6);  % mg C g-1 soil; Heated
a=SOCd_obs;
a(a==0)=NaN;
SOCd_obs=a;  % 0 to NaN
%SOCd_obs(1035)=35; %42.333 changed to 35, which is close average; 11/19/2015%
M = readmatrix('hf_socmbc_obs_cont_heat.csv');
SOCd_obsSD = M(:,3);  % mg C g-1 soil; Control
SOCd_obsSD = M(:,7);  % mg C g-1 soil; HEAT
a=SOCd_obsSD;
a(a==0)=NaN;
SOCd_obsSD=a;  % 0 to NaN
M = readmatrix('hf_socmbc_obs_cont_heat.csv');
MBCd_obs = M(:,4);  % mg C g-1 soil; Control
MBCd_obs = M(:,8);  % mg C g-1 soil; Heated
a=MBCd_obs;
a(a==0)=NaN;
MBCd_obs=a;  % 0 to NaN
M = readmatrix('hf_socmbc_obs_cont_heat.csv');
MBCd_obsSD = M(:,5);  % mg C g-1 soil; Control
MBCd_obsSD = M(:,9);  % mg C g-1 soil; Heated
a=MBCd_obsSD;
a(a==0)=NaN;
MBCd_obsSD=a;  % 0 to NaN
% soc mbc ends
% doc enc starts
M = readmatrix('hf_docenc_obs_cont_heat.csv');
DOCd_obs = M(:,2);  % mg C g soil-1; Control
DOCd_obs = M(:,4);  % mg C g soil-1; Heated
a=DOCd_obs;
a(a==0)=NaN;
DOCd_obs=a;  % 0 to NaN
M = readmatrix('hf_docenc_obs_cont_heat.csv');
DOCd_obsSD = M(:,3);  % mg C g soil-1; Control
DOCd_obsSD = M(:,5);  % mg C g soil-1; Heated
a=DOCd_obsSD;
a(a==0)=NaN;
DOCd_obsSD=a;  % 0 to NaN
M = readmatrix('hf_docenc_obs_cont_heat.csv');
ENCd_obs = M(:,6);  % mg C g soil-1; Control
ENCd_obs = M(:,8);  % mg C g soil-1; Heated
a=ENCd_obs;
a(a==0)=NaN;
ENCd_obs=a;  % 0 to NaN
M = readmatrix('hf_docenc_obs_cont_heat.csv');
ENCd_obsSD = M(:,7);  % mg C g soil-1; Control
ENCd_obsSD = M(:,9);  % mg C g soil-1; HEATED
a=ENCd_obsSD;
a(a==0)=NaN;
ENCd_obsSD=a;  % 0 to NaN
% doc enc ends
% Daily CO2
M = readmatrix('hf005-01-trace-gas_daily_cont_heat.csv');
CO2d_obs = M(:,5);   % mg C m-2 day-1; Control
CO2d_obs = M(:,17);  % mg C m-2 day-1; Heated
CO2d_obs = CO2d_obs/(10^5/rho); % rho=1; depth 10cm
CO2d_obs = CO2d_obs*0.67; % adjust proportion of autotrophic and heterotrophic resps 
a=CO2d_obs;
a(a==0)=NaN;
CO2d_obs=a; % 0 to NaN
M = readmatrix('hf005-01-trace-gas_daily_cont_heat.csv');
CO2d_obsSD = M(:,6);   % mg C m-2 day-1; Control
CO2d_obsSD = M(:,18);  % mg C m-2 day-1; Heated
CO2d_obsSD = CO2d_obsSD/(10^5/rho); % rho=1; depth 10cm
CO2d_obsSD = CO2d_obsSD*0.67;
a=CO2d_obsSD;
a(a==0)=NaN;
CO2d_obsSD=a; % 0 to NaN
% Daily co2 ends
% Hourly CO2
M = readmatrix('hf005-05-soil-respiration_hourly_cont_heat.csv');
CO2h_obs = M(:,6);   % mg C m-2 hr-1; Control
CO2h_obs = M(:,20);  % mg C m-2 hr-1; HEATED
CO2h_obs = CO2h_obs/(10^5/rho); % rho=1; depth 10cm
CO2h_obs = CO2h_obs*0.67; % adjust proportion of autotrophic and heterotrophic resps
a=CO2h_obs;
a(a==0)=NaN;
CO2h_obs=a; % 0 to NaN
M = readmatrix('hf005-05-soil-respiration_hourly_cont_heat.csv');
CO2h_obsSD = M(:,7);   % mg C m-2 hr-1; Control
CO2h_obsSD = M(:,21);  % mg C m-2 hr-1; Heated
CO2h_obsSD = CO2h_obsSD/(10^5/rho); % rho=1; depth 10cm
CO2h_obsSD = CO2h_obsSD*0.67;
a=CO2h_obsSD;
a(a==0)=NaN;
CO2h_obsSD=a; % 0 to NaN 
% Hourly CO2 ends
% Observations import ends
figure(2)
%comparsion between obs and sim
%CO2 hourly
x = CO2h_obs;
y = CO2h_sim;
ax = subplot(2,3,1);
plot(ax,x,y,'o');
xlabel('Obs');
ylabel('Sim');
xlim([0,0.0005]);
ylim([0,0.0005]);
title('Ori Hourly CO2');
%CO2 daily
x = CO2d_obs;
y = CO2d_sim;
ax = subplot(2,3,2);
plot(ax,x,y,'o');
xlabel('Obs');
ylabel('Sim');
xlim([-0.00,0.02]);
ylim([-0.00,0.02]);
title('Daily CO2');
%SOC
x1 = SOCd_obs;
y1 = SOCd_sim;
ax1 = subplot(2,3,3);
plot(ax1,x1,y1,'o');
xlabel('Obs');
ylabel('Sim');
xlim([5,75]);
ylim([5,75]);
title('SOC');
%MBC
x2 = MBCd_obs;
y2 = MBCd_sim;
ax2 = subplot(2,3,4);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,3]);
ylim([0,3]);
title('MBC');
%DOC
x2 = DOCd_obs;
y2 = DOCd_sim;
ax2 = subplot(2,3,5);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,0.5]);
ylim([0,0.5]);
title('DOC');
%ENC
x2 = ENCd_obs;
y2 = ENCd_sim;
ax2 = subplot(2,3,6);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,0.02]);
ylim([0,0.02]);
title('ENC');
%time series plots
%CO2 hourly
% output observation and simulation results
A=[CO2d_sim, SOCd_sim, DOCd_sim, MBCd_sim, ENCd_sim];
csvwrite('all_simulations.csv',A);
A=[CO2h_obs];
csvwrite('hCO2_obs.csv',A);
A=[CO2d_obs];
csvwrite('dCO2_obs.csv',A);
A=[SOCd_obs];
csvwrite('dSOC_obs.csv',A);
A=[DOCd_obs];
csvwrite('dDOC_obs.csv',A);
A=[MBCd_obs];
csvwrite('dMBC_obs.csv',A);
A=[ENCd_obs];
csvwrite('dENC_obs.csv',A);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%  Simulation Above --- DA down   %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data assimilation starts
% Parameters to be constrained
% a Ecref 
% b m
% c mRref
% d fD POC->DOC  
% e gD MBC->DOC
% f iPOC
% g iMOC
% h iQOC
% i iMBC
% j iDOC
% k iEP
% l iEM
% Initial pool sizes  
POC = 14.7; % initial SOC 48-49
MOC = 46; % initial SOC 48-49
QOC = 1.2; % initial SOC 48-49
SOC = POC+MOC+QOC;
MBC = 0.85; %                
DOC = 0.55; %                 
EP = 0.0024; %               
EM = 0.0024; %               
ENC = EP+EM; 
CO2 = 0.00016;
% initial values and Min&Max
par=[0.31  -0.008            0.00028   0.50   0.50   8.7 40  0.8  0.25  0.15 0.0007 0.0007]';

Min=[0.00  -0.020           0.00002   0.30   0.30   1   30  0.1  0.02  0.02 0.0001 0.0001]';

Max=[0.60  -0.000          0.003     0.70   0.70   23  55  1.9  0.90  0.90 0.0070 0.0070]';
% initial values and Min&Max ends
% extract 1st simulation results
SOC1=reshape(SOCd_sim,8035*1,1);
MBC1=reshape(MBCd_sim,8035*1,1);
DOC1=reshape(DOCd_sim,8035*1,1);
ENC1=reshape(ENCd_sim,8035*1,1);
CO2d1=reshape(CO2d_sim,8035*1,1);
CO2h1=reshape(CO2h_sim,192840*1,1);
% transfer
mod1SOC=SOC1;
mod1MBC=MBC1;
mod1DOC=DOC1;
mod1ENC=ENC1;
mod1CO2d=CO2d1;
mod1CO2h=CO2h1;
% 1st simulation resutls ends
% extract observation results
obscSOC=reshape(SOCd_obs,8035*1,1);
obscMBC=reshape(MBCd_obs,8035*1,1);
obscDOC=reshape(DOCd_obs,8035*1,1);
obscENC=reshape(ENCd_obs,8035*1,1);
obscCO2d=reshape(CO2d_obs,8035*1,1);
obscCO2h=reshape(CO2h_obs,192840*1,1);
% ends
% cost function 
 J_old= sum((mod1SOC-obscSOC).^2./SOCd_obsSD,'omitnan') + sum((mod1MBC-obscMBC).^2./MBCd_obsSD,'omitnan') + sum((mod1DOC-obscDOC).^2./DOCd_obsSD,'omitnan') + sum((mod1ENC-obscENC).^2./ENCd_obsSD,'omitnan') + sum((mod1CO2d-obscCO2d).^2./CO2d_obsSD,'omitnan')+sum((mod1CO2h-obscCO2h).^2./CO2h_obsSD,'omitnan');
% cost function ends
% Parmater sets selection starts
Parameters_keep= [0 0 0 0 0 0 0 0 0 0 0 0]';
Para_keep_final= [0 0 0 0 0 0 0 0 0 0 0 0 0]'; % jianwei added
nsimu=100000;                 % simulation iteration times 10000000
keep_count=1;
upgrade=1;
allow=10;
L_P=length(par); 
par_old=par;
diff=Max-Min;

for simu=1:nsimu;
		while (true)
        par_new = par_old+(rand(12,1)-0.5).*diff/allow;        
           if (par_new(1)>Min(1)&par_new(1)<Max(1)...
             & par_new(2)>Min(2)&par_new(2)<Max(2)...
             & par_new(3)>Min(3)&par_new(3)<Max(3)...
             & par_new(4)>Min(4)&par_new(4)<Max(4)...   
             & par_new(5)>Min(5)&par_new(5)<Max(5)...
             & par_new(6)>Min(6)&par_new(6)<Max(6)...
             & par_new(7)>Min(7)&par_new(7)<Max(7)...
             & par_new(8)>Min(8)&par_new(8)<Max(8)...
             & par_new(9)>Min(9)&par_new(9)<Max(9)...
             & par_new(10)>Min(10)&par_new(10)<Max(10)...
             & par_new(11)>Min(11)&par_new(11)<Max(11)... 
             & par_new(12)>Min(12)&par_new(12)<Max(12))  
            break;
            end
        end
         a=par_new(1);
         b=par_new(2);
         c=par_new(3);
         d=par_new(4);
         e=par_new(5);
         f=par_new(6);
         g=par_new(7);
         h=par_new(8);
         i=par_new(9);
         j=par_new(10);
         k=par_new(11);
         l=par_new(12);

% Simulation starts
POC = f;  % alternative     f
%POC = 8.74557002525951; % initial SOC 48-49
MOC = g;  % alternative     g
%MOC = 39.9719369229464; % initial SOC 48-49
QOC = h;  % alternative     h
%QOC = 0.79265584970119; % initial SOC 48-49
SOC = POC+MOC+QOC;
MBC = i; %                 i
DOC = j; %                 j
EP = k; %               k
EM = l; %               l
ENC = EP+EM; 
CO2 = 0.000160000000005213;

for z = 1:192840;
EC = (ECref+m*(TempK(z)-Tref));
mR = mRref*exp((-EamR/R)*(1/TempK(z)-1/Tref));
KP = KPref*exp((-EaKP/R)*(1/TempK(z)-1/Tref));
%fSWP  % 12/2017 added
% -0.033 Mpa field capacity SWP; SWPmin = -13.86 Mpa; exponent = 1.20
if sm(z) < -13.86;
    fSWP=0;
        elseif sm(z) < -0.033;
            fSWP = 1-(log(sm(z)/(-0.033))/log((-13.86)/(-0.033))).^1.2;
        else
            fSWP = 1;
end
% fSWP
  fSWP_A2D = abs(sm(z)).^4/(abs(sm(z)).^4+abs(-0.4)*4); % -0.4 Mpa; exponent = 4
VP = VPref*exp((-EaVP/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KM = KMref*exp((-EaKM/R)*(1/TempK(z)-1/Tref));
VM = VMref*exp((-EaVM/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KD = KDref*exp((-EaKD/R)*(1/TempK(z)-1/Tref));
VD = VDref*exp((-EaVD/R)*(1/TempK(z)-1/Tref));
Kads = Kadsref*exp((-EaKads/R)*(1/TempK(z)-1/Tref));
Kdes = Kdesref*exp((-EaKdes/R)*(1/TempK(z)-1/Tref));
KBA = Kads/Kdes;
  F1 = (VD+mR)*DOC*MBC/(EC*(KD+DOC));
  F2 = VP*EP*POC/(KP+POC);
  F3 = VM*EM*MOC/(KM+MOC);
  F4 = (1/EC-1)*VD*MBC*DOC/(KD+DOC);
  F5 = (1/EC-1)*mR*MBC*DOC/(KD+DOC);
  F6 = Kads*DOC*(1-QOC/Qmax);
  F7 = Kdes*QOC/Qmax;
  F8 = mR*MBC*(1-pEP-pEM); % *fSWP_A2D;
  F9ep = pEP*mR*MBC;
  F9em = pEM*mR*MBC;
  F10ep = rEP*EP;
  F10em = rEM*EM;
  POC = POC + IP(z) +(1-e)*F8 - F2;
  MOC = MOC + (1-d)*F2 - F3;
  QOC = QOC + F6 - F7;
  SOC = POC + MOC +QOC;
  MBC = MBC + F1 - (F4+F5) - F8 - (F9ep+F9em);
  DOC = DOC + ID(z) + d*F2 + e*F8 + F3 + (F10ep+F10em) - F1 - (F6-F7);
  EP = EP + F9ep - F10ep;
  EM = EM + F9em - F10em;
  ENC = EP + EM;
  CO2 = F4 + F5;
 % hourly
    POCh(z)=POC;
    MOCh(z)=MOC;
    QOCh(z)=QOC;
    SOCh(z)=SOC;
    MBCh(z)=MBC;
    DOCh(z)=DOC;
    EPh(z)=EP;
    EMh(z)=EM;
    ENCh(z)=ENC;
    CO2h(z)=CO2;
end
% daily
    for z=1:8035;
    POCd(z)=mean(POCh(((z-1)*24+1):(z*24)));
    MOCd(z)=mean(MOCh(((z-1)*24+1):(z*24)));
    QOCd(z)=mean(QOCh(((z-1)*24+1):(z*24)));
    SOCd(z)=mean(SOCh(((z-1)*24+1):(z*24)));
    MBCd(z)=mean(MBCh(((z-1)*24+1):(z*24)));
    DOCd(z)=mean(DOCh(((z-1)*24+1):(z*24)));
    EPd(z)=mean(EPh(((z-1)*24+1):(z*24)));
    EMd(z)=mean(EMh(((z-1)*24+1):(z*24)));
    ENCd(z)=mean(ENCh(((z-1)*24+1):(z*24)));    
    CO2d(z)=sum(CO2h(((z-1)*24+1):(z*24)));
    end   
% Simulation ends
% simulation result extractions
SOC2=reshape(SOCd,8035*1,1);
MBC2=reshape(MBCd,8035*1,1);
DOC2=reshape(DOCd,8035*1,1);
ENC2=reshape(ENCd,8035*1,1);
CO2d2=reshape(CO2d,8035*1,1);
CO2h2=reshape(CO2h,192840*1,1);
mod2SOC=SOC2;
mod2MBC=MBC2;
mod2DOC=DOC2;
mod2ENC=ENC2;
mod2CO2d=CO2d2;
mod2CO2h=CO2h2;
% extraction ends
% cost function recalculates
J_new = sum((mod2SOC-obscSOC).^2./SOCd_obsSD,'omitnan') + sum((mod2MBC-obscMBC).^2./MBCd_obsSD,'omitnan') + sum((mod2DOC-obscDOC).^2./DOCd_obsSD,'omitnan') + sum((mod2ENC-obscENC).^2./ENCd_obsSD,'omitnan') + sum((mod2CO2d-obscCO2d).^2./CO2d_obsSD,'omitnan') + sum((mod2CO2h-obscCO2h).^2./CO2h_obsSD,'omitnan');
% cost function ends
% parameter sets selection
delta_J=J_new-J_old;
if min(1,exp(-delta_J))>rand;
                 Parameters_keep(:,upgrade)=par_new;
                 Para_keep_final(:,upgrade)= vertcat(simu, par_new); % jianwei added
                 J_keep(upgrade)=J_new;
                 upgrade=upgrade+1
                 par_old=par_new;
                 J_old=J_new;          
   	end
   simu
    %keep_count
     Parameters_rec(:,simu)=par_old;
     J_rec(:,simu)=J_old;   
     
end
% Parameter sets selection ends
A=[simu,upgrade,upgrade/simu];
csvwrite('da_acceptance_rate.csv',A);
csvwrite('da_para_keep.csv',[Para_keep_final]');
% extract maximum likelyhood estimator
a=mle(Parameters_keep(1,ceil(upgrade/2):upgrade-1),'distribution','gev');
b=mle(Parameters_keep(2,ceil(upgrade/2):upgrade-1),'distribution','gev');
c=mle(Parameters_keep(3,ceil(upgrade/2):upgrade-1),'distribution','gev');
d=mle(Parameters_keep(4,ceil(upgrade/2):upgrade-1),'distribution','gev');
e=mle(Parameters_keep(5,ceil(upgrade/2):upgrade-1),'distribution','gev');
f=mle(Parameters_keep(6,ceil(upgrade/2):upgrade-1),'distribution','gev');
g=mle(Parameters_keep(7,ceil(upgrade/2):upgrade-1),'distribution','gev');
h=mle(Parameters_keep(8,ceil(upgrade/2):upgrade-1),'distribution','gev');
i=mle(Parameters_keep(9,ceil(upgrade/2):upgrade-1),'distribution','gev');
j=mle(Parameters_keep(10,ceil(upgrade/2):upgrade-1),'distribution','gev');
k=mle(Parameters_keep(11,ceil(upgrade/2):upgrade-1),'distribution','gev');
l=mle(Parameters_keep(12,ceil(upgrade/2):upgrade-1),'distribution','gev');
% exxtraction ends
% mle selection starts
B=[a;b;c;d;e;f;g;h;i;j;k;l]; % low, upper limite and mle
csvwrite('da_parameters_summary.csv',B);
a=a(3);
b=b(3);
c=c(3);
d=d(3);
e=e(3);
f=f(3);
g=g(3);
h=h(3);
i=i(3);
j=j(3);
k=k(3);
l=l(3);
% mle selection ends
% two sets of means integrated
mleU=[a;b;c;d;e;f;g;h;i;j;k;l];
csvwrite('da_parameters_summary_mle_heat.csv',mleU);  % 12/2017 added
% mle selection ends
% mean para value extracted starts
% all para 
tmpALL=mean(Para_keep_final,2);
tmpALL(1,:) = [];
% latter half mean, comparable to range for mle
Para_keep_final=Para_keep_final(1:13,ceil(upgrade/2):upgrade-1);
tmp=mean(Para_keep_final,2);
tmp(1,:) = [];
a=tmp(1);
b=tmp(2);
c=tmp(3);
d=tmp(4);
e=tmp(5);
f=tmp(6);
g=tmp(7);
h=tmp(8);
i=tmp(9);
j=tmp(10);
k=tmp(11);
l=tmp(12);
% two sets of means integrated
ZZ=[tmpALL,tmp];
csvwrite('da_parameters_summary_mean.csv',ZZ);
% mean para ends
% Simulation starts
POC = f;
MOC = g;
QOC = h;
SOC = POC+MOC+QOC;
MBC = i;
DOC = j;
EP = k;
EM = l;
ENC = EP+EM; 
CO2 = 0.000160000000005213;

for z = 1:192840;
EC = (ECref+m*(TempK(z)-Tref));
mR = mRref*exp((-EamR/R)*(1/TempK(z)-1/Tref));
KP = KPref*exp((-EaKP/R)*(1/TempK(z)-1/Tref));
%fSWP  % 12/2017 added
% -0.033 Mpa field capacity SWP; SWPmin = -13.86 Mpa; exponent = 1.20
if sm(z) < -13.86;
    fSWP=0;
        elseif sm(z) < -0.033;
            fSWP = 1-(log(sm(z)/(-0.033))/log((-13.86)/(-0.033))).^1.2;
        else
            fSWP = 1;
end
% fSWP
  fSWP_A2D = abs(sm(z)).^4/(abs(sm(z)).^4+abs(-0.4)*4); % -0.4 Mpa; exponent = 4
VP = VPref*exp((-EaVP/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KM = KMref*exp((-EaKM/R)*(1/TempK(z)-1/Tref));
VM = VMref*exp((-EaVM/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
KD = KDref*exp((-EaKD/R)*(1/TempK(z)-1/Tref));
VD = VDref*exp((-EaVD/R)*(1/TempK(z)-1/Tref));
Kads = Kadsref*exp((-EaKads/R)*(1/TempK(z)-1/Tref));
Kdes = Kdesref*exp((-EaKdes/R)*(1/TempK(z)-1/Tref));
KBA = Kads/Kdes;
  F1 = (VD+mR)*DOC*MBC/(EC*(KD+DOC));
  F2 = VP*EP*POC/(KP+POC);
  F3 = VM*EM*MOC/(KM+MOC);
  F4 = (1/EC-1)*VD*MBC*DOC/(KD+DOC);
  F5 = (1/EC-1)*mR*MBC*DOC/(KD+DOC);
  F6 = Kads*DOC*(1-QOC/Qmax);
  F7 = Kdes*QOC/Qmax;
  F8 = mR*MBC*(1-pEP-pEM); % *fSWP_A2D;
  F9ep = pEP*mR*MBC;
  F9em = pEM*mR*MBC;
  F10ep = rEP*EP;
  F10em = rEM*EM;
  POC = POC + IP(z) +(1-e)*F8 - F2;
  MOC = MOC + (1-d)*F2 - F3;
  QOC = QOC + F6 - F7;
  SOC = POC + MOC +QOC;
  MBC = MBC + F1 - (F4+F5) - F8 - (F9ep+F9em);
  DOC = DOC + ID(z) + d*F2 + e*F8 + F3 + (F10ep+F10em) - F1 - (F6-F7);
  EP = EP + F9ep - F10ep;
  EM = EM + F9em - F10em;
  ENC = EP + EM;
  CO2 = F4 + F5;
 % hourly
    POCh(z)=POC;
    MOCh(z)=MOC;
    QOCh(z)=QOC;
    SOCh(z)=SOC;
    MBCh(z)=MBC;
    DOCh(z)=DOC;
    EPh(z)=EP;
    EMh(z)=EM;
    ENCh(z)=ENC;
    CO2h(z)=CO2;
end
% daily
    for z=1:8035;
    POCd(z)=mean(POCh(((z-1)*24+1):(z*24)));
    MOCd(z)=mean(MOCh(((z-1)*24+1):(z*24)));
    QOCd(z)=mean(QOCh(((z-1)*24+1):(z*24)));
    SOCd(z)=mean(SOCh(((z-1)*24+1):(z*24)));
    MBCd(z)=mean(MBCh(((z-1)*24+1):(z*24)));
    DOCd(z)=mean(DOCh(((z-1)*24+1):(z*24)));
    EPd(z)=mean(EPh(((z-1)*24+1):(z*24)));
    EMd(z)=mean(EMh(((z-1)*24+1):(z*24)));
    ENCd(z)=mean(ENCh(((z-1)*24+1):(z*24)));    
    CO2d(z)=sum(CO2h(((z-1)*24+1):(z*24)));
    end   
% Simulation ends
% extract final simulation results     
SOCfinal=reshape(SOCd,8035*1,1);
MBCfinal=reshape(MBCd,8035*1,1);
DOCfinal=reshape(DOCd,8035*1,1);
ENCfinal=reshape(ENCd,8035*1,1);
CO2dfinal=reshape(CO2d,8035*1,1);
CO2hfinal=reshape(CO2h,192840*1,1);
% extraction ends
%Export final pool sizes or flux to .txt files
fid=fopen('da_SOCfinal.txt','w');
fprintf(fid,'%6.8f \r\n', SOCfinal);
fclose(fid);
fid=fopen('da_MBCfinal.txt','w');
fprintf(fid,'%6.8f \r\n', MBCfinal);
fclose(fid);
fid=fopen('da_DOCfinal.txt','w');
fprintf(fid,'%6.8f \r\n', DOCfinal);
fclose(fid);
fid=fopen('da_ENCfinal.txt','w');
fprintf(fid,'%6.8f \r\n', ENCfinal);
fclose(fid);
fid=fopen('da_CO2dfinal.txt','w');
fprintf(fid,'%6.8f \r\n', CO2dfinal);
fclose(fid); 
fid=fopen('da_CO2hfinal.txt','w');
fprintf(fid,'%6.8f \r\n', CO2hfinal);
fclose(fid); 
% exports ends
%comparsion between obs and final sim
% transfer variable names 
SOCd_sim=SOCfinal;
MBCd_sim=MBCfinal;
DOCd_sim=DOCfinal;
ENCd_sim=ENCfinal;
CO2h_sim=CO2hfinal;
CO2d_sim=CO2dfinal;
% transfer ends
%%%%%%%%%%%%%
figure(3)
%%%%%%%%%%%%%
% simulation with time
y = CO2h_sim;
ax = subplot(2,3,1);
plot(ax,1:192840,y,'o');
xlabel('Time: hour');
ylabel('Sim');
xlim([1,192840]);
ylim([-0.000,0.002]);
title('Sim Hourly CO2, mg C g soil-1 hr-1');
%CO2 daily
y = CO2d_sim;
ax = subplot(2,3,2);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([1,8035]);
ylim([-0.00,0.04]);
title('Daily CO2, mg C g soil-1 day-1');
%SOC
y = SOCd_sim;
ax = subplot(2,3,3);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,60]);
title('SOC, mg C g soil-1');
%MBC
y = MBCd_sim;
ax = subplot(2,3,4);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,4]);
title('MBC, mg C g soil-1');
%DOC
y = DOCd_sim;
ax = subplot(2,3,5);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,0.5]);
title('DOC, mg C g soil-1');
%ENC
y = ENCd_sim;
ax = subplot(2,3,6);
plot(ax,1:8035,y,'o');
xlabel('Time: day');
ylabel('Sim');
xlim([0,8035]);
ylim([0,0.04]);
title('ENC, mg C g soil-1');
%%%%%%%%%%%%%%%%%
figure(4)
%%%%%%%%%%%%%%%%%
%comparsion between obs and BEST sim
%CO2 hourly
x = CO2h_obs;
y = CO2h_sim;
ax = subplot(2,3,1);
plot(ax,x,y,'o');
xlabel('Obs');
ylabel('Sim');
xlim([0,0.0005]);
ylim([0,0.0005]);
title('Sim Hourly CO2');
%CO2 daily
x = CO2d_obs;
y = CO2d_sim;
ax = subplot(2,3,2);
plot(ax,x,y,'o');
xlabel('Obs');
ylabel('Sim');
xlim([-0.00,0.02]);
ylim([-0.00,0.02]);
title('Daily CO2');
%SOC
x1 = SOCd_obs;
y1 = SOCd_sim;
ax1 = subplot(2,3,3);
plot(ax1,x1,y1,'o');
xlabel('Obs');
ylabel('Sim');
xlim([5,75]);
ylim([5,75]);
title('SOC');
%MBC
x2 = MBCd_obs;
y2 = MBCd_sim;
ax2 = subplot(2,3,4);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,3]);
ylim([0,3]);
title('MBC');
%DOC
x2 = DOCd_obs;
y2 = DOCd_sim;
ax2 = subplot(2,3,5);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,0.5]);
ylim([0,0.5]);
title('DOC');
%ENC
x2 = ENCd_obs;
y2 = ENCd_sim;
ax2 = subplot(2,3,6);
plot(ax2,x2,y2,'o')
xlabel('Obs');
ylabel('Sim');
xlim([0,0.02]);
ylim([0,0.02]);
title('ENC');
% comparison ends

% if final simulations are better, then 
% write all selected parameter sets to .txt file 
fid=fopen('Parakeep_1.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(1,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_2.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(2,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_3.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(3,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_4.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(4,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_5.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(5,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_6.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(6,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_7.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(7,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_8.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(8,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_9.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(9,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_10.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(10,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_11.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(11,ceil(upgrade/2):upgrade-1));
fclose(fid);
fid=fopen('Parakeep_12.txt','w');
fprintf(fid,'%6.8f \r\n', Parameters_keep(12,ceil(upgrade/2):upgrade-1));
fclose(fid);

% write ends
% Display all slected parameters in plots 
nn=880; 
mm=150;
figure(5)  
  subplot(2,6,1), hist(Parameters_keep(1,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(1) Max(1) 0 nn])
    ylabel('Frequency','fontsize',25)
    set(gca,'fontsize',25)
  title('Ecref','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
  subplot(2,6,2), hist(Parameters_keep(2,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(2) Max(2) 0 nn])
    set(gca,'fontsize',25)
  title('m','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
   subplot(2,6,3), hist(Parameters_keep(3,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(3) Max(3) 0 nn])
    set(gca,'fontsize',25)
  title('mR_,_r_e_f','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
   subplot(2,6,4), hist(Parameters_keep(4,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(4) Max(4) 0 nn])
    set(gca,'fontsize',25)
  title('gD','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')

   subplot(2,6,5), hist(Parameters_keep(5,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(5) Max(5) 0 nn])
    set(gca,'fontsize',25)
  title('fD','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')

    subplot(2,6,6), hist(Parameters_keep(6,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(6) Max(6) 0 nn])
    set(gca,'fontsize',25)
  title('iPOC','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
    subplot(2,6,7), hist(Parameters_keep(7,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(7) Max(7) 0 nn])
    ylabel('Frequency','fontsize',25)
    set(gca,'fontsize',25)
  title('iMOC','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
     subplot(2,6,8), hist(Parameters_keep(8,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(8) Max(8) 0 nn])
    set(gca,'fontsize',25)
  title('iQOC','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
     subplot(2,6,9), hist(Parameters_keep(9,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(9) Max(9) 0 nn])
    set(gca,'fontsize',25)
  title('iMBC','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
     subplot(2,6,10), hist(Parameters_keep(10,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(10) Max(10) 0 nn])
    set(gca,'fontsize',25)
  title('iDOC','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')
  
     subplot(2,6,11), hist(Parameters_keep(11,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(11) Max(11) 0 nn])
  ylabel('Frequency','fontsize',25)
    set(gca,'fontsize',25)
  title('iEP','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')

     subplot(2,6,12), hist(Parameters_keep(12,ceil(upgrade/2):upgrade-1),mm)
  axis([Min(12) Max(12) 0 nn])
  ylabel('Frequency','fontsize',25)
    set(gca,'fontsize',25)
  title('iEM','FontSize',25)
  h=findobj(gca,'Type','patch');
  set(h,'FaceColor',[0.5,0.5,0.5],'EdgeColor','k')  
  

%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   % long-term predictions (2011~2054) under heated plot; 
%   % soil temp and litterfall based on two repeated cycle of 1989~2010;
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TempK=repmat(TempK,3,1); 
% sm=repmat(sm,3,1); % 12/2017 added
% IP=repmat(IP,3,1);
% ID=repmat(ID,3,1);
% % Simulation starts
% tmp=mean(Para_keep_final,2);
% tmp(1,:) = [];
% %tmp=mleU; % mle or mean 12/2017 added
% a=tmp(1);
% b=tmp(2);
% c=tmp(3);
% d=tmp(4);
% e=tmp(5);
% f=tmp(6);
% g=tmp(7);
% h=tmp(8);
% i=tmp(9);
% j=tmp(10);
% k=tmp(11);
% l=tmp(12);
% 
% POC = f;
% MOC = g;
% QOC = h;
% SOC = POC+MOC+QOC;
% MBC = i;
% DOC = j;
% EP = k;
% EM = l;
% ENC = EP+EM; 
% CO2 = 0.000160000000005213;
% 
% for z = 1:(192840*3);
% EC = (ECref+m*(TempK(z)-Tref));
% mR = mRref*exp((-EamR/R)*(1/TempK(z)-1/Tref));
% KP = KPref*exp((-EaKP/R)*(1/TempK(z)-1/Tref));
% %fSWP  % 12/2017 added
% % -0.033 Mpa field capacity SWP; SWPmin = -13.86 Mpa; exponent = 1.20
% if sm(z) < -13.86;
%     fSWP=0;
%         elseif sm(z) < -0.033;
%             fSWP = 1-(log(sm(z)/(-0.033))/log((-13.86)/(-0.033))).^1.2;
%         else
%             fSWP = 1;
% end
% % fSWP
%   fSWP_A2D = abs(sm(z)).^4/(abs(sm(z)).^4+abs(-0.4)*4); % -0.4 Mpa; exponent = 4
% VP = VPref*exp((-EaVP/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
% KM = KMref*exp((-EaKM/R)*(1/TempK(z)-1/Tref));
% VM = VMref*exp((-EaVM/R)*(1/TempK(z)-1/Tref))*fSWP;  % 12/2017 added
% KD = KDref*exp((-EaKD/R)*(1/TempK(z)-1/Tref));
% VD = VDref*exp((-EaVD/R)*(1/TempK(z)-1/Tref));
% Kads = Kadsref*exp((-EaKads/R)*(1/TempK(z)-1/Tref));
% Kdes = Kdesref*exp((-EaKdes/R)*(1/TempK(z)-1/Tref));
% KBA = Kads/Kdes;
%   F1 = (VD+mR)*DOC*MBC/(EC*(KD+DOC));
%   F2 = VP*EP*POC/(KP+POC);
%   F3 = VM*EM*MOC/(KM+MOC);
%   F4 = (1/EC-1)*VD*MBC*DOC/(KD+DOC);
%   F5 = (1/EC-1)*mR*MBC*DOC/(KD+DOC);
%   F6 = Kads*DOC*(1-QOC/Qmax);
%   F7 = Kdes*QOC/Qmax;
%   F8 = mR*MBC*(1-pEP-pEM); % *fSWP_A2D;
%   F9ep = pEP*mR*MBC;
%   F9em = pEM*mR*MBC;
%   F10ep = rEP*EP;
%   F10em = rEM*EM;
%   POC = POC + IP(z) +(1-e)*F8 - F2;
%   MOC = MOC + (1-d)*F2 - F3;
%   QOC = QOC + F6 - F7;
%   SOC = POC + MOC +QOC;
%   MBC = MBC + F1 - (F4+F5) - F8 - (F9ep+F9em);
%   DOC = DOC + ID(z) + d*F2 + e*F8 + F3 + (F10ep+F10em) - F1 - (F6-F7);
%   EP = EP + F9ep - F10ep;
%   EM = EM + F9em - F10em;
%   ENC = EP + EM;
%   CO2 = F4 + F5;
%  % hourly
%     POCh(z)=POC;
%     MOCh(z)=MOC;
%     QOCh(z)=QOC;
%     SOCh(z)=SOC;
%     MBCh(z)=MBC;
%     DOCh(z)=DOC;
%     EPh(z)=EP;
%     EMh(z)=EM;
%     ENCh(z)=ENC;
%     CO2h(z)=CO2;
% end
% % daily
%     for z=1:(8035*3);
%     POCd(z)=mean(POCh(((z-1)*24+1):(z*24)));
%     MOCd(z)=mean(MOCh(((z-1)*24+1):(z*24)));
%     QOCd(z)=mean(QOCh(((z-1)*24+1):(z*24)));
%     SOCd(z)=mean(SOCh(((z-1)*24+1):(z*24)));
%     MBCd(z)=mean(MBCh(((z-1)*24+1):(z*24)));
%     DOCd(z)=mean(DOCh(((z-1)*24+1):(z*24)));
%     EPd(z)=mean(EPh(((z-1)*24+1):(z*24)));
%     EMd(z)=mean(EMh(((z-1)*24+1):(z*24)));
%     ENCd(z)=mean(ENCh(((z-1)*24+1):(z*24)));    
%     CO2d(z)=sum(CO2h(((z-1)*24+1):(z*24)));
%     end   
% % Simulation ends
% % extract final prediction results
% SOCfinal=reshape(SOCd,8035*3,1);
% MBCfinal=reshape(MBCd,8035*3,1);
% DOCfinal=reshape(DOCd,8035*3,1);
% ENCfinal=reshape(ENCd,8035*3,1);
% CO2dfinal=reshape(CO2d,8035*3,1);
% CO2hfinal=reshape(CO2h,192840*3,1);
% 
% %Export final pool sizes or flux to .txt files
% fid=fopen('prediction_SOCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', SOCfinal);
% fclose(fid);
% fid=fopen('prediction_MBCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', MBCfinal);
% fclose(fid);
% fid=fopen('prediction_DOCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', DOCfinal);
% fclose(fid);
% fid=fopen('prediction_ENCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', ENCfinal);
% fclose(fid);
% fid=fopen('prediction_CO2dfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', CO2dfinal);
% fclose(fid); 
% fid=fopen('prediction_CO2hfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', CO2hfinal);
% fclose(fid); 
% % exports ends
% 
% % Plot outputs with time
% SOCd_sim=SOCfinal;
% MBCd_sim=MBCfinal;
% DOCd_sim=DOCfinal;
% ENCd_sim=ENCfinal;
% CO2h_sim=CO2hfinal;
% CO2d_sim=CO2dfinal;
% % transfer ends
% %%%%%%%%%%%%%
% figure(6)
% %%%%%%%%%%%%%
% % prediction with time
% y = CO2h_sim;
% ax = subplot(2,3,1);
% plot(ax,1:578520,y,'o');
% xlabel('Time: hour');
% ylabel('Sim');
% xlim([1,578520]);
% ylim([-0.000,0.002]);
% title('Hourly CO2, mg C g soil-1 hr-1');
% %CO2 daily
% y = CO2d_sim;
% ax = subplot(2,3,2);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([1,24105]);
% ylim([-0.00,0.04]);
% title('Daily CO2, mg C g soil-1 day-1');
% %SOC
% y = SOCd_sim;
% ax = subplot(2,3,3);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,60]);
% title('SOC, mg C g soil-1');
% %MBC
% y = MBCd_sim;
% ax = subplot(2,3,4);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,4]);
% title('MBC, mg C g soil-1');
% %DOC
% y = DOCd_sim;
% ax = subplot(2,3,5);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,0.5]);
% title('DOC, mg C g soil-1');
% %ENC
% y = ENCd_sim;
% ax = subplot(2,3,6);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,0.04]);
% title('ENC, mg C g soil-1');
% %%%%%%%%%%%%%%%%%  
% 
% 
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   % long-term predictions (2011~2054) under heated plot; 
%   % soil temp and litterfall based on two repeated cycle of 1989~2010;
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TempK=repmat(TempK,3,1); 
% sm=repmat(sm,3,1); % 12/2017 added
% IP=repmat(IP,3,1);
% ID=repmat(ID,3,1);
% % Simulation starts
% %tmp=mean(Para_keep_final,2); % using mean parameter values
% %tmp(1,:) = [];
% tmp=mleU; % mle or mean 12/2017 added
% a=tmp(1);
% b=tmp(2);
% c=tmp(3);
% d=tmp(4);
% e=tmp(5);
% f=tmp(6);
% g=tmp(7);
% h=tmp(8);
% i=tmp(9);
% j=tmp(10);
% k=tmp(11);
% l=tmp(12);
% 
% POC = f;  % alternative          f
% %POC = 8.74557002525951; % initial SOC 48-49
% MOC = g;  % alternative          g
% %MOC = 39.9719369229464; % initial SOC 48-49
% QOC = h;  % alternative          h
% %QOC = 0.79265584970119; % initial SOC 48-49
% SOC = POC+MOC+QOC;
% MBC = i;  %                      i
% DOC = j;  %                      j
% EP = k;   %                      k
% EM = l;   %                      l
% ENC = EP+EM;
% CO2 = 0.000160000000005213;
% 
% for z = 1:(192840*3);
% EC = (a+b*(TempK(z)-Tref));
% mR = c*exp((-EamR/R)*(1/TempK(z)-1/Tref));
% KP = KPref*exp((-EaKP/R)*(1/TempK(z)-1/Tref));
% %fSWP  % 12/2017 added
% % -0.033 Mpa field capacity SWP; SWPmin = -13.86 Mpa; exponent = 1.20
% if sm(z) < -13.86 ;
%     fSWP=0;
%         elseif sm(z) < -0.033;
%             fSWP = 1-(log(sm(z)/(-0.033))/log((-13.86)/(-0.033))).^1.2;
%         else
%             fSWP = 1;
% end
% % fSWP
%   fSWP_A2D = abs(sm(z)).^4/(abs(sm(z)).^4+abs(-0.4)*4); % -0.4 Mpa; exponent = 4
% VP = VPref*exp((-EaVP/R)*(1/TempK(z)-1/Tref))*fSWP;
% KM = KMref*exp((-EaKM/R)*(1/TempK(z)-1/Tref));
% VM = VMref*exp((-EaVM/R)*(1/TempK(z)-1/Tref))*fSWP;
% KD = KDref*exp((-EaKD/R)*(1/TempK(z)-1/Tref));
% VD = VDref*exp((-EaVD/R)*(1/TempK(z)-1/Tref));
% Kads = Kadsref*exp((-EaKads/R)*(1/TempK(z)-1/Tref));
% Kdes = Kdesref*exp((-EaKdes/R)*(1/TempK(z)-1/Tref));
% KBA = Kads/Kdes;
%   F1 = (VD+mR)*DOC*MBC/(EC*(KD+DOC));
%   F2 = VP*EP*POC/(KP+POC);
%   F3 = VM*EM*MOC/(KM+MOC);
%   F4 = (1/EC-1)*VD*MBC*DOC/(KD+DOC);
%   F5 = (1/EC-1)*mR*MBC*DOC/(KD+DOC);
%   F6 = Kads*DOC*(1-QOC/Qmax);
%   F7 = Kdes*QOC/Qmax;
%   F8 = mR*MBC*(1-pEP-pEM)*fSWP_A2D; % *fSWP_A2D;
%   F9ep = pEP*mR*MBC;
%   F9em = pEM*mR*MBC;
%   F10ep = rEP*EP;
%   F10em = rEM*EM;
%   POC = POC + IP(z) +(1-e)*F8 - F2;
%   MOC = MOC + (1-d)*F2 - F3;
%   QOC = QOC + F6 - F7;
%   SOC = POC + MOC +QOC;
%   MBC = MBC + F1 - (F4+F5) - F8 - (F9ep+F9em);
%   DOC = DOC + ID(z) + d*F2 + e*F8 + F3 + (F10ep+F10em) - F1 - (F6-F7);
%   EP = EP + F9ep - F10ep;
%   EM = EM + F9em - F10em;
%   ENC = EP + EM;
%   CO2 = F4 + F5;
%  % hourly
%     POCh(z)=POC;
%     MOCh(z)=MOC;
%     QOCh(z)=QOC;
%     SOCh(z)=SOC;
%     MBCh(z)=MBC;
%     DOCh(z)=DOC;
%     EPh(z)=EP;
%     EMh(z)=EM;
%     ENCh(z)=ENC;
%     CO2h(z)=CO2;
% end
% % daily
%     for z=1:(8035*3);
%     POCd(z)=mean(POCh(((z-1)*24+1):(z*24)));
%     MOCd(z)=mean(MOCh(((z-1)*24+1):(z*24)));
%     QOCd(z)=mean(QOCh(((z-1)*24+1):(z*24)));
%     SOCd(z)=mean(SOCh(((z-1)*24+1):(z*24)));
%     MBCd(z)=mean(MBCh(((z-1)*24+1):(z*24)));
%     DOCd(z)=mean(DOCh(((z-1)*24+1):(z*24)));
%     EPd(z)=mean(EPh(((z-1)*24+1):(z*24)));
%     EMd(z)=mean(EMh(((z-1)*24+1):(z*24)));
%     ENCd(z)=mean(ENCh(((z-1)*24+1):(z*24)));    
%     CO2d(z)=sum(CO2h(((z-1)*24+1):(z*24)));
%     end   
% % Simulation ends
% % extract final prediction results
% SOCfinal=reshape(SOCd,8035*3,1);
% MBCfinal=reshape(MBCd,8035*3,1);
% DOCfinal=reshape(DOCd,8035*3,1);
% ENCfinal=reshape(ENCd,8035*3,1);
% CO2dfinal=reshape(CO2d,8035*3,1);
% CO2hfinal=reshape(CO2h,192840*3,1);
% 
% %Export final pool sizes or flux to .txt files
% fid=fopen('prediction_SOCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', SOCfinal);
% fclose(fid);
% fid=fopen('prediction_MBCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', MBCfinal);
% fclose(fid);
% fid=fopen('prediction_DOCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', DOCfinal);
% fclose(fid);
% fid=fopen('prediction_ENCfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', ENCfinal);
% fclose(fid);
% fid=fopen('prediction_CO2dfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', CO2dfinal);
% fclose(fid); 
% fid=fopen('prediction_CO2hfinal.txt','w');
% fprintf(fid,'%6.8f \r\n', CO2hfinal);
% fclose(fid); 
% % exports ends
% 
% % Plot outputs with time
% SOCd_sim=SOCfinal;
% MBCd_sim=MBCfinal;
% DOCd_sim=DOCfinal;
% ENCd_sim=ENCfinal;
% CO2h_sim=CO2hfinal;
% CO2d_sim=CO2dfinal;
% % transfer ends
% %%%%%%%%%%%%%
% figure(7)
% %%%%%%%%%%%%%
% % prediction with time
% y = CO2h_sim;
% ax = subplot(2,3,1);
% plot(ax,1:578520,y,'o');
% xlabel('Time: hour');
% ylabel('Sim');
% xlim([1,578520]);
% ylim([-0.000,0.002]);
% title('Hourly CO2, mg C g soil-1 hr-1');
% %CO2 daily
% y = CO2d_sim;
% ax = subplot(2,3,2);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([1,24105]);
% ylim([-0.00,0.04]);
% title('Daily CO2, mg C g soil-1 day-1');
% %SOC
% y = SOCd_sim;
% ax = subplot(2,3,3);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,60]);
% title('SOC, mg C g soil-1');
% %MBC
% y = MBCd_sim;
% ax = subplot(2,3,4);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,4]);
% title('MBC, mg C g soil-1');
% %DOC
% y = DOCd_sim;
% ax = subplot(2,3,5);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,0.5]);
% title('DOC, mg C g soil-1');
% %ENC
% y = ENCd_sim;
% ax = subplot(2,3,6);
% plot(ax,1:24105,y,'o');
% xlabel('Time: day');
% ylabel('Sim');
% xlim([0,24105]);
% ylim([0,0.04]);
% title('ENC, mg C g soil-1');
% %%%%%%%%%%%%%%%%%  

figHandles = findall(0, 'Type', 'figure');
figHandles = sort(figHandles);
for i = 1:length(figHandles)
    filename = sprintf('figure_%d.png', i);
    saveas(figHandles(i), filename);
end