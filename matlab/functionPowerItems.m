function [cpupwr, bckhpwr, appwr, uepwr] = functionPowerItems(L,N,K, SE_UL, SE_DL, eta)
% SE: size (K,1), the specturm efficiency for each UE
% eta: size(K,1), the power control coefficients

M=L*N;
B = 20e6; %Transmission bandwidth (Hz)
Bc = 180e3; %Channel coherence bandwidth (Hz)
Tc = 10e-3; %Channel coherence time (s)
U = Bc * Tc; %Coherence block (number of channel uses)
%sigma2B = 10^(-9.6-3); %Total noise power (B*sigma2 in W)
sigma2 = 290*1.38e-23*B*9;
N_0 = sigma2/B;
%Traffic assumptions (From Table 2)
zetaDL = 0.6; %Fraction of downlink transmission
zetaUL = 0.4; %Fraction of uplink transmission

%Relative lengths of pilot sequences (From Table 2)
tauDL = 1; %Relative pilot length in the downlink
tauUL = 1; %Relative pilot length in the uplink

%Hardware characterization (From Table 2)
alphaDL = 0.39; %PA efficiency at the APs
alphaUL = 0.3; %PA efficiency at the UEs
L_CPU = 12.8e9; %Computational efficiency at BSs (flops/W)
L_UE = 5e9; %Computational efficiency at UEs (flops/W)
P_FIX_CPU = 18; %Fixed power consumption (control signals, backhaul, etc.) (W)
P_FIX_AP = 2; %Power consumed by local oscillator at a BS (W)

P_UE = 0.1; %Power required to run the circuit components at a UE (W)
P_AP = 0.1; %total power budget for antenna 
%rho = 0.1;  %total power budget for antenna 
P_COD = 0.1e-9; %Power required for channel coding (W/(bit/s))
P_DEC = 0.8e-9; %Power required for channel decoding (W/(bit/s))
P_BT = 0.25e-9; %Power required for backhaul traffic (W/(bit/s))
p_0 =  0.2;

P_CE_CPU = B/U * 2*tauUL * M *K^2 / L_CPU;
P_CD = (sum( B*SE_UL) + sum( B*SE_DL))*(P_COD + P_DEC);
P_LP = B*(1-(tauUL+tauDL)*K/U)*2*M*K/L_CPU + B/U * 3*M*K / L_CPU;
cpupwr = P_FIX_CPU+ P_CE_CPU + P_CD + P_LP;

bckhpwr = p_0 + P_BT*(sum( B*SE_UL) + sum( B*SE_DL));
appwr = P_FIX_AP +  1/alphaDL * P_AP * N_0 * N * sum(eta);

P_CE_UE = B/U * 4*tauDL *K^2 / L_UE;
P_UE_UL = 1/alphaDL * P_UE * N_0;
uepwr = P_CE_UE + P_UE_UL;

end