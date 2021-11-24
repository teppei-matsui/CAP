% Sample code for "On co-activation pattern analysis and non-stationarity
% of resting brain activity" (Matsui et al., 2021)
%
% Written by Teppei Matsui at Okayama University under MIT license
% contact : Teppei Matsui (tematsui at okayama-u.ac.jp)



load('SampleTC.mat');
nsubject = size(TC,3);

% z-normalize TC
temp = TC;
temp = temp - repmat(mean(temp,1),[size(TC,1),1,1]); % demean
temp = temp ./ repmat(std(temp,[],1),[size(TC,1),1,1]); % z normalize
TC = temp;

% Global Signal Regression (GSR) and surrogate data generation
flag_GSR = 1; % set this to 0 when skipping GSR
disp('Simlating Data and Removing Global Signal...');
TC_sim = zeros(size(TC));
TC_sim_static =  zeros(size(TC));
TC_sim_ARR =  zeros(size(TC));
TC_sim_PRR =  zeros(size(TC));
for i =1:nsubject
    myTC = squeeze(TC(:,:,i));
    
    %------------------------------------------------------------%
    % The Laumann Null (retains spectral and covariance structures)
    % Requires simulate_BOLD_timecourse_func_v2.m obtained from the following
    % website
    % https://sites.wustl.edu/petersenschlaggarlab/resources/
    %------------------------------------------------------------%
    Y = permute(myTC,[2 1]); % d_ROI x n_frames
    C = cov(Y');    
    cov_target = C;
    P_target = mean((abs(fft(Y,[],2)).^2),1);
    timelength = size(Y,2);
    TRin = 0.72; TRout = 0.72;
    [myTC_sim]= ...
        simulate_BOLD_timecourse_func_v2(timelength,TRin,TRout,cov_target,P_target); %Laumann Null model

    %------------------------------------------------------------%
    % Static Null (retains only covariance strucrue)
    %------------------------------------------------------------%
    myTC_sim_static = mvnrnd(zeros(size(myTC,2),1),cov_target,size(myTC,1));
    
    %------------------------------------------------------------%
    % Auto-Regressive Randomization with lag = 1
    % Based on the procedure described in Liegeois et al., Neuroimage,
    % 2017
    %------------------------------------------------------------%
    data = permute(myTC,[2,1]);
    X = data(:,2:end);
    Z = zeros(size(data,1) + 1, size(data,2) - 1);
    Z(1,:)=ones(1,size(data,2) - 1); %intercept
    Z(2:end,:) = data(:,1:end-1);
    
    A = X*Z'*inv(Z*Z');
    data_sim = zeros(size(data));
    data_sim(:,1) = data(:,randi([1,size(data,2)],1));
    resdata = data(:,2:end) - A(:,2:end)*data_sim(:,1:end-1);
    mycov_target = cov(resdata');
    for myframe = 2:size(myTC,1)
        epsi = mvnrnd(zeros(size(data,1),1),mycov_target);
        data_sim(:,myframe) = A(:,1) + A(:,2:end)*data_sim(:,myframe-1)+epsi(:);
    end
    
    myTC_sim_ARR = permute(data_sim,[2,1]);

    %------------------------------------------------------%
    % Phase Randomized Data
    % Code adapted from:
    % https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/fMRI_dynamics/Liegeois2017_Surrogates
    %------------------------------------------------------%
    ft_TC = fft(myTC,[],1);
    T  = size(myTC,1);
    len_ser = ceil((T-1)/2);
    interv1 = 1:len_ser; 
    interv2 = len_ser+1:T;
    ph_rnd = rand([len_ser, 1]);
    
    randphase = exp(1i*2*pi*ph_rnd);
    randphase_1stHalf = repmat(randphase,1,size(myTC,2));
    randphase_2ndHalf = conj(flipud(randphase_1stHalf));
    
    pr_ft_TC = ft_TC;
    pr_ft_TC(interv1,:) = ft_TC(interv1,:).*randphase_1stHalf;
    pr_ft_TC(interv2,:) = ft_TC(interv2,:).*randphase_2ndHalf;
    
    myTC_sim_PRR = real(ifft(pr_ft_TC));
  
    if flag_GSR==1
        % Global Signal Regression
        GS = mean(myTC,2);
        for j = 1:size(TC,2)
            [b,bint,Res]=regress(myTC(:,j),GS);
            TC(:,j,i)=Res;
        end

        GS = mean(myTC_sim,2);
        for j = 1:size(TC,2)
            [b,bint,Res]=regress(myTC_sim(:,j),GS);
            TC_sim(:,j,i)=Res;
        end
        
        GS = mean(myTC_sim_static,2);
        for j = 1:size(TC,2)
            [b,bint,Res]=regress(myTC_sim_static(:,j),GS);
            TC_sim_static(:,j,i)=Res;
        end
        
        GS = mean(myTC_sim_ARR,2);
        for j = 1:size(TC,2)
            [b,bint,Res]=regress(myTC_sim_ARR(:,j),GS);
            TC_sim_ARR(:,j,i)=Res;
        end
        
        GS = mean(myTC_sim_PRR,2);
        for j = 1:size(TC,2)
            [b,bint,Res]=regress(myTC_sim_PRR(:,j),GS);
            TC_sim_PRR(:,j,i)=Res;
        end
    else
        TC_sim(:,:,i) = myTC_sim;
        TC_sim_static(:,:,i) = myTC_sim_static;
        TC_sim_ARR(:,:,i) = myTC_sim_ARR;
        TC_sim_PRR(:,:,i) = myTC_sim_PRR;
    end
end
disp('done');


% concatenate TC across subjects
TC = permute(TC,[1 3 2]);
dim = size(TC);
TC = reshape(TC,[dim(1)*dim(2), dim(3)]);

TC_sim = permute(TC_sim,[1 3 2]);
dim = size(TC_sim);
TC_sim = reshape(TC_sim,[dim(1)*dim(2), dim(3)]);

TC_sim_static = permute(TC_sim_static,[1 3 2]);
dim = size(TC_sim_static);
TC_sim_static = reshape(TC_sim_static,[dim(1)*dim(2), dim(3)]);

TC_sim_ARR = permute(TC_sim_ARR,[1 3 2]);
dim = size(TC_sim_ARR);
TC_sim_ARR = reshape(TC_sim_ARR,[dim(1)*dim(2), dim(3)]);

TC_sim_PRR = permute(TC_sim_PRR,[1 3 2]);
dim = size(TC_sim_PRR);
TC_sim_PRR = reshape(TC_sim_PRR,[dim(1)*dim(2), dim(3)]);

disp('done')


%% CAP detection and clustering
myROI = 10;
nclust = 8;

[temp, cap_frames] = sort(TC(:,myROI),'descend');
cap_frames = cap_frames(1:floor(size(TC,1)*0.15));
cap_frames = sort(cap_frames);
cap = TC(cap_frames,:);
cap_mean_data =mean(cap,1);

idx= kmeans(cap,nclust,'Distance','correlation');
cap_data = zeros(size(TC,2),nclust);
for i = 1:nclust
    cap_data(:,i) = mean(cap(find(idx==i),:),1);
end
idx_data = idx;

% % Laumann
[temp, cap_frames] = sort(TC_sim(:,myROI),'descend');
cap_frames = cap_frames(1:floor(size(TC,1)*0.15));
cap_frames = sort(cap_frames);
cap = TC_sim(cap_frames,:);
cap_mean_sim =mean(cap,1);

idx= kmeans(cap,nclust,'Distance','correlation');
cap_sim = zeros(size(TC,2),nclust);
for i = 1:nclust
    cap_sim(:,i) = mean(cap(find(idx==i),:),1);
end
idx_sim = idx;

% % static
[temp, cap_frames] = sort(TC_sim_static(:,myROI),'descend');
cap_frames = cap_frames(1:floor(size(TC,1)*0.15));
cap_frames = sort(cap_frames);
cap = TC_sim_static(cap_frames,:);
cap_mean_sim_static =mean(cap,1);

idx= kmeans(cap,nclust,'Distance','correlation');
cap_sim_static = zeros(size(TC,2),nclust);
for i = 1:nclust
    cap_sim_static(:,i) = mean(cap(find(idx==i),:),1);
end
idx_sim_static = idx;

% % ARR
[temp, cap_frames] = sort(TC_sim_ARR(:,myROI),'descend');
cap_frames = cap_frames(1:floor(size(TC,1)*0.15));
cap_frames = sort(cap_frames);
cap = TC_sim_ARR(cap_frames,:);
cap_mean_sim_ARR =mean(cap,1);

idx= kmeans(cap,nclust,'Distance','correlation');
cap_sim_ARR = zeros(size(TC,2),nclust);
for i = 1:nclust
    cap_sim_ARR(:,i) = mean(cap(find(idx==i),:),1);
end

idx_sim_ARR = idx;

% % PRR
[temp, cap_frames] = sort(TC_sim_PRR(:,myROI),'descend');
cap_frames = cap_frames(1:floor(size(TC,1)*0.15));
cap_frames = sort(cap_frames);
cap = TC_sim_PRR(cap_frames,:);
cap_mean_sim_PRR =mean(cap,1);

idx= kmeans(cap,nclust,'Distance','correlation');
cap_sim_PRR = zeros(size(TC,2),nclust);
for i = 1:nclust
    cap_sim_PRR(:,i) = mean(cap(find(idx==i),:),1);
end
idx_sim_PRR = idx;

disp('cap clustering done')

%% Comparison of mean CAP for real and simulated data
myind = 1:size(TC,2);
myind(myROI)=[];

figure;
subplot(1,4,1);
plot(cap_mean_data(myind),cap_mean_sim(myind),'k.');axis square;
title('Laumann');

subplot(1,4,2);
plot(cap_mean_data(myind),cap_mean_sim_static(myind),'k.');axis square;
title('static');

subplot(1,4,3);
plot(cap_mean_data(myind),cap_mean_sim_ARR(myind),'k.');axis square;
title('ARR');

subplot(1,4,4);
plot(cap_mean_data(myind),cap_mean_sim_PRR(myind),'k.');axis square;
title('PRR');

%% Within data-type CAP module similarity
figure;
subplot(1,4,1);
imagesc(corrcoef(cap_sim),[0 1]);
axis square; colorbar;
title('Laumann');

subplot(1,4,2);
imagesc(corrcoef(cap_sim_static),[0 1]);
axis square; colorbar;
title('static');

subplot(1,4,3);
imagesc(corrcoef(cap_sim_ARR),[0 1]);
axis square; colorbar;
title('ARR');

subplot(1,4,4);
imagesc(corrcoef(cap_sim_PRR),[0 1]);
axis square; colorbar;
title('PRR');

%% Match CAP modules in real and simulated data
R = corr(cap_data,cap_sim);

C = perms(1:nclust);
C = permute(C,[2 1]);

vals = zeros(size(C,2),1);

for i = 1:size(C,2)
    vals(i) = mean(diag(R(:,C(:,i))));
end
[temp, myind] = max(vals);
sortind = C(:,myind);
R_Laumann = R;
sortind_Laumann = sortind;
maxval_Laumann = temp;

% %static
R = corr(cap_data,cap_sim_static);

C = perms(1:nclust);
C = permute(C,[2 1]);

vals = zeros(size(C,2),1);

for i = 1:size(C,2)
    vals(i) = mean(diag(R(:,C(:,i))));
end
[temp, myind] = max(vals);
sortind = C(:,myind);
R_static = R;
sortind_static = sortind;
maxval_static = temp;


% %ARR
R = corr(cap_data,cap_sim_ARR);

C = perms(1:nclust);
C = permute(C,[2 1]);

vals = zeros(size(C,2),1);

for i = 1:size(C,2)
    vals(i) = mean(diag(R(:,C(:,i))));
end
[temp, myind] = max(vals);
sortind = C(:,myind);
R_ARR = R;
sortind_ARR = sortind;
maxval_ARR = temp;

% %PRR
R = corr(cap_data,cap_sim_PRR);

C = perms(1:nclust);
C = permute(C,[2 1]);

vals = zeros(size(C,2),1);

for i = 1:size(C,2)
    vals(i) = mean(diag(R(:,C(:,i))));
end
[temp, myind] = max(vals);
sortind = C(:,myind);
R_PRR = R;
sortind_PRR = sortind;
maxval_PRR = temp;

figure;
subplot(1,4,1);
imagesc(R_Laumann(:,sortind_Laumann),[0 1]);
axis square; colorbar;
title('sorted');
set(gca,'XTick',1:8,'XTickLabel',sortind_Laumann);

subplot(1,4,2);
imagesc(R_static(:,sortind_static),[0 1]);
axis square; colorbar;
title('sorted');
set(gca,'XTick',1:8,'XTickLabel',sortind_static);

subplot(1,4,3);
imagesc(R_ARR(:,sortind_ARR),[0 1]);
axis square; colorbar;
title('sorted');
set(gca,'XTick',1:8,'XTickLabel',sortind_ARR);

subplot(1,4,4);
imagesc(R_PRR(:,sortind_PRR),[0 1]);
axis square; colorbar;
title('sorted');
set(gca,'XTick',1:8,'XTickLabel',sortind_PRR);

%% state distribution
y_data = hist(idx_data,[1:nclust]);
y_sim = hist(idx_sim,[1:nclust]);
y_sim_static = hist(idx_sim_static,[1:nclust]);
y_sim_ARR = hist(idx_sim_ARR,[1:nclust]);
y_sim_PRR = hist(idx_sim_PRR,[1:nclust]);

figure;hold on;
plot(1:nclust,y_data/sum(y_data),'k--');
plot(1:nclust,y_sim(sortind_Laumann)/sum(y_sim),'k');
plot(1:nclust,y_sim_static(sortind_static)/sum(y_sim_static),'r');
plot(1:nclust,y_sim_ARR(sortind_ARR)/sum(y_sim_ARR),'g');
plot(1:nclust,y_sim_PRR(sortind_PRR)/sum(y_sim_PRR),'b');
ylim([0,0.4]);

%% transition matrix

idx = idx_data;
Tmat = zeros(nclust,nclust);
temp = idx(1:end-1);
for i = 1:nclust
    myframes = find(temp==i);
    temp2 = hist(idx(myframes+1),[1:nclust]);
    Tmat(i,:) = temp2/sum(temp2);
end
Tmat_data = Tmat;


idx = idx_sim;
Tmat = zeros(nclust,nclust);
temp = idx(1:end-1);
for i = 1:nclust
    myframes = find(temp==i);
    temp2 = hist(idx(myframes+1),[1:nclust]);
    Tmat(i,:) = temp2/sum(temp2);
end
Tmat_sim = Tmat(sortind_Laumann,sortind_Laumann);

idx = idx_sim_static;
Tmat = zeros(nclust,nclust);
temp = idx(1:end-1);
for i = 1:nclust
    myframes = find(temp==i);
    temp2 = hist(idx(myframes+1),[1:nclust]);
    Tmat(i,:) = temp2/sum(temp2);
end
Tmat_sim_static = Tmat(sortind_static,sortind_static);

idx = idx_sim_ARR;
Tmat = zeros(nclust,nclust);
temp = idx(1:end-1);
for i = 1:nclust
    myframes = find(temp==i);
    temp2 = hist(idx(myframes+1),[1:nclust]);
    Tmat(i,:) = temp2/sum(temp2);
end
Tmat_sim_ARR = Tmat(sortind_ARR,sortind_ARR);

idx = idx_sim_PRR;
Tmat = zeros(nclust,nclust);
temp = idx(1:end-1);
for i = 1:nclust
    myframes = find(temp==i);
    temp2 = hist(idx(myframes+1),[1:nclust]);
    Tmat(i,:) = temp2/sum(temp2);
end
Tmat_sim_PRR = Tmat(sortind_PRR,sortind_PRR);
    
figure;
subplot(2,3,1);
imagesc(Tmat_data,[0 0.2]);axis square;colorbar;
title('Data');
subplot(2,3,2);
imagesc(Tmat_sim,[0 0.2]);axis square;colorbar;
title('Laumann');
subplot(2,3,3);
imagesc(Tmat_sim_static,[0 0.2]);axis square;colorbar;
title('static');
subplot(2,3,4);
imagesc(Tmat_sim_ARR,[0 0.2]);axis square;colorbar;
title('ARR');
subplot(2,3,5);
imagesc(Tmat_sim_PRR,[0 0.2]);axis square;colorbar;
title('PRR');
