function rtn = function_ee_rl(UEpositions, APpositions,cvtype)
% UE Locations is given by the parameter, that is generated at env reset
% and keep consistent over each session.
if nargin==2
    cvtype='MR';
end
nbrOfSetups = 1;  %Number of Monte-Carlo setups
rho_tot = 1000;
nbrOfRealizations = 10;  %Number of channel realizations per setup
K = length(UEpositions);
L = length(APpositions);
UEpositions = UEpositions(:,1)+1i*UEpositions(:,2);
APpositions = APpositions(:,1)+1i*APpositions(:,2);
N=8;
tau_c = 200;  %Length of coherence block
%Length of pilot sequences
tau_p = 10;

p = 100;

%Prepare to save simulation results
SE_scalable_MR_tot = zeros(K,nbrOfSetups);
SE_scalable_LP_MMSE_tot = zeros(K,nbrOfSetups);
SE_scalable_P_MMSE_tot = zeros(K,nbrOfSetups);
SE_scalable_MMSE_tot = zeros(K,nbrOfSetups);
SE_all_MR_tot = zeros(K,nbrOfSetups);
SE_all_LP_MMSE_tot = zeros(K,nbrOfSetups);
SE_all_P_MMSE_tot = zeros(K,nbrOfSetups);
SE_all_MMSE_tot = zeros(K,nbrOfSetups);

%% general setup
B = 20e6;
squareLength = 2000; %meter
%Noise figure (in dB)
noiseFigure = 7;

%Compute noise power
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;

%Pathloss exponent
alpha = 3.76;

%Standard deviation of shadow fading
sigma_sf = 10;

%Average channel gain in dB at a reference distance of 1 meter. Note that
%-35.3 dB corresponds to -148.1 dB at 1 km, using pathloss exponent 3.76
constantTerm = -35.3;

%Define the antenna spacing (in number of wavelengths)
antennaSpacing = 1/2; %Half wavelength distance

%Angular standard deviation around the nominal angle (measured in degrees)
ASDdeg = 20;

%Set threshold for when a non-master AP decides to serve a UE
threshold = -20; %dB

mu_path_gain = zeros(K,1);

%Prepare to save results
gainOverNoisedB = zeros(L,K,nbrOfSetups);
R = zeros(N,N,L,K,nbrOfSetups);
distances = zeros(L,K,nbrOfSetups);
pilotIndex = zeros(K,nbrOfSetups);
D = zeros(L,K,nbrOfSetups);
masterAPs = zeros(K,1);


%Compute alternative AP locations by using wrap around
wrapHorizontal = repmat([-squareLength 0 squareLength],[3 1]);
wrapVertical = wrapHorizontal';
wrapLocations = wrapHorizontal(:)' + 1i*wrapVertical(:)';
APpositionsWrapped = repmat(APpositions,[1 length(wrapLocations)]) + repmat(wrapLocations,[L 1]);
n=nbrOfSetups;
for k = 1:K
    
    %Compute distances assuming that the APs are 10 m above the UEs
    [distancetoUE,whichpos] = min(abs(APpositionsWrapped - repmat(UEpositions(k),size(APpositionsWrapped))),[],2);
    distances(:,k,n) = sqrt(10^2+distancetoUE.^2);
    
    %Compute the channel gain divided by noise power  % only distance
    %realted, not related to f!!!
    gainOverNoisedB(:,k,n) = constantTerm - alpha*10*log10(distances(:,k,n)) + sigma_sf*randn(size(distances(:,k,n))) - noiseVariancedBm;
    
    %Determine the master AP for UE k by looking for AP with best
    %channel condition
    [~,master] = max(gainOverNoisedB(:,k,n));
    D(master,k,n) = 1;
    masterAPs(k) = master;

    term  = alpha*10*log10(distances(master,k,n)) + sigma_sf*randn(size(distances(master,k,n)));
    term = -term/10;
    mu_path_gain(k) =  10^term;
    
    %Assign orthogonal pilots to the first tau_p UEs
    if k <= tau_p

        pilotIndex(k,n) = k;

    else %Assign pilot for remaining UEs

        %Compute received power from to the master AP from each pilot
        pilotinterference = zeros(tau_p,1);

        for t = 1:tau_p

            pilotinterference(t) = sum(db2pow(gainOverNoisedB(master,pilotIndex(1:k-1,n)==t,n)));

        end

        %Find the pilot with the least receiver power
        [~,bestpilot] = min(pilotinterference);
        pilotIndex(k,n) = bestpilot;

    end
    
    

    %Go through all APs
    for l = 1:L
        
        %Compute nominal angle between UE k and AP l
        angletoUE = angle(UEpositions(k)-APpositionsWrapped(l,whichpos(l)));
        
        %Generate normalized spatial correlation matrix using the local
        %scattering model
        R(:,:,l,k,n) = db2pow(gainOverNoisedB(l,k,n))*functionRlocalscattering(N,angletoUE,ASDdeg,antennaSpacing);
        
    end
    
end

% mode 2: remove this
% Each AP serves the UE with the strongest channel condition on each of
% the pilots where the AP isn't the master AP, but only if its channel
% is not too weak compared to the master AP
% for l = 1:L
%     for t = 1:tau_p
%         pilotUEs = find(t==pilotIndex(:,n));
%         if sum(D(l,pilotUEs,n)) == 0 %If the AP is not a master AP
%             %Find the UE with pilot t with the best channel
%             [gainValue,UEindex] = max(gainOverNoisedB(l,pilotUEs,n));
%             %Serve this UE if the channel is at most "threshold" weaker
%             %than the master AP's channel
%             %[gainValue gainOverNoisedB(masterAPs(pilotUEs(UEindex)),pilotUEs(UEindex),n)]   
%             if gainValue - gainOverNoisedB(masterAPs(pilotUEs(UEindex)),pilotUEs(UEindex),n) >= threshold
%                 D(l,pilotUEs(UEindex),n) = 1;
%             end
%         end
%     end
% end


%% channel, se
ap_wo_ue = length(find(sum(D,2)<1));
    
%Generate channel realizations with estimates and estimation
%error correlation matrices
[Hhat,H,B,C] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p);

%Compute SE using Propositions 1 and 2
[SE_MR_UL,SE_LP_MMSE_UL,SE_P_MMSE_UL,SE_MMSE] = functionComputeSE_uplink(Hhat,H,D,B,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,R,pilotIndex);

%Compute the equal power allocation for centralized precoding
rho_central = (rho_tot/tau_p)*ones(K,1);

%Compute the power allocation in Eq. (43) for distributed precoding
rho_dist = zeros(L,K);
gainOverNoise = db2pow(gainOverNoisedB);

for l = 1:L
    
    %Extract which UEs are served by AP l
    servedUEs = find(D(l,:)==1);
    
    %Compute denominator in Eq. (43)
    normalizationAPl = sum(sqrt(gainOverNoise(l,servedUEs)));
    
    for ind = 1:length(servedUEs)
        
        rho_dist(l,servedUEs(ind)) = rho_tot*sqrt(gainOverNoise(l,servedUEs(ind)))/normalizationAPl;
        
    end
    
end

%Compute SE using the capacity bound in Proposition 3
[SE_MR,SE_LP_MMSE,SE_MR_perfect,SE_LP_MMSE_perfect,SE_P_MMSE,SE_P_MMSE_perfect] = functionComputeSE_downlink(Hhat,H,D,B,C,tau_c,tau_p,nbrOfRealizations,N,K,L,p,rho_dist,R,pilotIndex,rho_central);

%% ee
B = 20e6;
if strcmp(cvtype , 'MR')
    [cpupwr, bckhpwr, appwr_mr, uepwr] = functionPowerItems(L,N,K, SE_MR_UL, SE_MR, ones(K,1)/K);
    p_total = cpupwr+bckhpwr+appwr_mr*L +uepwr*K;
    ee = B*(sum(SE_MR)+sum(SE_MR_UL))/p_total;
    se = mean(SE_MR);
elseif strcmp(cvtype ,'LP_MMSE')
    [cpupwr, bckhpwr, appwr_lp, uepwr] = functionPowerItems(L,N,K, SE_LP_MMSE_UL, SE_LP_MMSE, ones(K,1)/K);
    p_total = cpupwr+bckhpwr+appwr_lp*L +uepwr*K;
    ee = B*(sum(SE_LP_MMSE)+sum(SE_LP_MMSE_UL))/p_total;
    se = mean(SE_LP_MMSE);
elseif strcmp(cvtype , 'P_MMSE')
    [cpupwr, bckhpwr, appwr_p, uepwr] = functionPowerItems(L,N,K, SE_P_MMSE_UL, SE_P_MMSE, ones(K,1)/K);
    p_total = cpupwr+bckhpwr+appwr_p*L +uepwr*K;
    ee = B*(sum(SE_P_MMSE)+sum(SE_P_MMSE_UL))/p_total;
    se = mean(SE_P_MMSE);
end

rtn.ee = ee/1e6;
rtn.se = se;

end