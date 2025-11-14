%% This script can be used with the associated source dataset to plot the
%graph in Figure 1E, which exhibits temporal correlations in turning
%structure after "switch" evemts for freely behaving zebrafish

clear all;close all;
load figure1e_sourcedata; %load source data

%initialize variables
totsize = 20; %set the maximum number of turns in a sequence to plot
angthresh = 0; %set the absolute turn magnitude used for detecting a "switch" event

fishmean = zeros(length(concat_turns),totsize);

for i=1:length(concat_turns) %loop over all 19 fish in the dataset
    b1 = []; %initialize matrix that will be populated with signed cumulative angle
    for j=1:length(concat_turns{i}) %loop over all valid turn sequences for each fish.
        %sequences are separated because fish leave and enter the valid
        %consideration radius (1 cm from dish border) multiple times during
        %an experiment
    totcumulative = cumulativespont(concat_turns{i}{j},angthresh,totsize); %analyze sequences
    
    b1 = cat(1,b1,totcumulative);
    end
    fishmean(i,:) = nanmean(b1); %sequences that aren't at least totsize in length have NaN placeholders.
end

figure;
SEM = std(fishmean,0)/sqrt(size(fishmean,1));%sqrt(size(savA,1));%/tracktot;
meanSav = mean(fishmean,1);
fill([1:totsize totsize:-1:1],[meanSav + SEM meanSav(end:-1:1) - SEM(end:-1:1)],'b','facealpha',0.5,'edgealpha',0)
hold on;hp = plot(meanSav,'b','linewidth',2);
hb = plot([1 totsize],[0 0],'r','linewidth',2);
box off
xlabel('swim #','fontsize',20);ylabel('cumulative signed angle','fontsize',20);
set(gca,'fontsize',15)
axis([0 20 -0.05 0.7])
legend([hp hb],'real data','expected, random','location','best');
