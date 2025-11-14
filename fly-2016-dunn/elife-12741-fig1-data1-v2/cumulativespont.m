%%totcumulative = CUMULATIVESPONT(SEQ,ANGTHRESH,TOTSIZE)
%
% CUMULATIVESPONT takes SEQ, a sequenced of turn amplitudes (signed)
% left and right turns, and
%
% ANGTHRESH, a threshold in absolute turn magntiude used to find L->R or
% R->L switch events, and
%
% TOTSIZE, the number of turns in a sequence to consider
%
%and returns totcumulative, the cumulative turn history of SEQ, used to analyze
%the length of spontaneous turn "chains"

function totcumulative = cumulativespont(seq,angthresh,totsize)

turn_amp = seq; %find switch events
signed_turn_amp = sign(seq);
switches = signed_turn_amp;
switches = find(diff(switches)~=0)+1;
switches = switches(find(abs(turn_amp(switches))>angthresh));

turn_amp = signed_turn_amp; %only analyzed signed turn vector

turn_amp = [turn_amp NaN(1,totsize)];
totcumulative = zeros(length(switches),totsize);

for switchi = 1:length(switches); %for each switch, incorporate cumulative angle sequence
    if signed_turn_amp(switches(switchi)) > 0 %incorporate cumulative angle sequence with leading sign matching turn direction
        totcumulative(switchi,:) = cumsum(turn_amp(switches(switchi):switches(switchi)+totsize-1))-turn_amp(switches(switchi));
    else
        totcumulative(switchi,:) = -cumsum(turn_amp(switches(switchi):switches(switchi)+totsize-1))+turn_amp(switches(switchi));
    end
end


if isempty(switches) & length(seq) > 1; %if there is no switch event across the given sequence, then the fish
    %is the middle of a long sequence. incoporate these sequences.
    totcumulative = NaN(1,totsize);
    totcumulative(1:length(seq)) = cumsum([0 ones(1, length(seq)-1)]);
    totcumulative = totcumulative(1:totsize);
end


if ~isempty(switches) %incorporate the chain that exists *before* the first detected switch event
totcumulative(end+1,:) = NaN(1,totsize);
minsize = min([totsize switches(1)-1]);
totcumulative(end,1:minsize) = cumsum([0 ones(1, minsize-1)]);

end

end