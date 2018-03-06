function q=ReinforcementLearningGreedy(R, gamma, goalState, alpha, epsilon)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implemented by Benjamin Auzanneau % Wassim Ben Youssef

% Original Q Learning algorithm implemented by Ionnis Markis and Andrew Chalikiopoulos
% (https://github.com/mak92/Q-Learning-Algorithm-Implementation-in-MATLAB)
% Inspired by the Q-learning tutorial of Kardi Teknomo
% (http://people.revoledu.com/kardi/)
%
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
clc;
format short
format compact

% Four inputs: R, gamma, alpha, epsilon
if nargin<1,
% immediate reward matrix;
    R=csvread('RewardMatrix_version1.csv');
end
if nargin<2,
    gamma=0.8;              % discount factor
    alpha=0.5;              % learning rate
    epsilon=0.8;             % epsilon value for greedy algorithm
end
if nargin<3
    goalState=21;           
end

q=zeros(size(R));        % initialize Q as zero
q1=ones(size(R))*inf;    % initialize previous Q as big number
count=0;                 % counter
steps=0;                 % counts the number of steps to goal
B=[];                    % matrix to add results of steps and episode count
cumReward=0;             % counter to calculate accumulated reward
exploitCount=0;          % counter to count the number of time we exploit
exploreCount=0;          % counter to count the number of time we explore

for episode=1:100000     % we use a high number here because we want it to stop only when it converges
    
    state=1;        % Starting state of the agent
    
    
    while state~=goalState            % loop until reach goal state
        % select any action from this state using ?-greedy
        x=find(R(state,:)>=-999);         % find possible action of this state including when it go down
        if size(x,1)>0,
            
            r=rand; % get a uniform random number between 0-1
     
     % choose either explore or exploit
     if r>=epsilon   % exploit
         [~,qmax]=(max(q(state,x(1:end)))); % check for action with highest Q value
         if size(qmax)>1                    % If we have similar Q-values for different states (for instance at the beginning), we chose the next state randomly
            x1 = x(randperm(size(x)));
         else
            x1 = x(qmax); 
         end % set action with highest Q value as next state
         
         if epsilon>=0.5
            epsilon=epsilon*0.99999; % decrease epsilon
         else
             epsilon=epsilon*0.9999; % decrease epsilon
         end
         
         cumReward=cumReward+q(state,x1); %keep track of cumulative reward for graph
         exploitCount=exploitCount+1;
         %display('The agent exploits.');
         
     else        % explore
             x1=RandomPermutation(x);   % randomize the possible action
             x1=x1(1);                  % select an action (only the first element of random sequence)
             
             if epsilon>=0.5
                epsilon=epsilon*0.99999; % decrease epsilon
             else
                epsilon=epsilon*0.9999; % decrease epsilon
             end
             
             cumReward=cumReward+q(state,x1); %keep track of cumulative reward for graph
             exploreCount=exploreCount+1;
             %display('The agent explores.');
     
     end

        x2 = find(R(x1,:)>=-999);   % find possible steps from next step
        qMax=(max(q(x1,x2(1:end)))); % extract qmax from all possible next states
        q(state,x1)= q(state,x1)+alpha*((R(state,x1)+gamma*qMax)-q(state,x1));    % Temporal Difference Error
        state=x1;    % set state to next state
        
        end
           
        if state~=goalState     % keep track of steps taken if goal not reached
            steps=steps+1;
            
        else
            steps=steps+1;
            %episodes=episodes+1; % if goal reach increase episode counter
            A=[episode; steps; cumReward];   % create episodes, steps and cumReward matrix
            B=horzcat(B, A);    % add the new results to combined matrix 
        end
    
    end
        
    % break if convergence: small deviation on q for 1000 consecutive
    if sum(sum(abs(q1-q)))<0.00001 && sum(sum(q >0)) && epsilon<0.01
        if count>1000,
            q1=q;
            %episode  % report last episode
            break % for
        else
            q1=q;
            count=count+1; % set counter if deviation of q is small
        end
    else
        q1=q;
        count=0;  % reset counter when deviation of q from previous q is large
    end
    fprintf('Episode %i completed. The agent required %i steps to reach the goal.The cumulative reward gained is %i.\n', episode, steps, cumReward);
    steps=0;    % reset steps counter to 0
    cumReward=0;    % reset cumReward counter to 0
end

% row 4 in matrix is cumReward/steps taken per episode
%B(4,:) = (B(3,:)./B(2,:));
B(4,:) = rdivide(B(3,:),B(2,:));

%episodes vs cumReward taken averaged against steps taken
%plot(B(1,1 : 5 : end),B(3,1 : 5 : end));

%create a plot of episodes vs steps taken and episodes vs cumReward taken averaged against steps taken
figure % new figure
[combinedGraph] = plotyy(B(1,1 : 5 : end),B(2,1 : 5 : end), B(1,1 : 5 : end),B(4,1 : 5 : end));


title('Q-Learning Performance')
xlabel('Episodes')
ylabel(combinedGraph(1),'Steps') % left y-axis
ylabel(combinedGraph(2),'Cumulative Reward/step') % right y-axis

% create a plot of episodes vs cumReward/steps
figure
plot(B(1,1 : 5 : end),B(4,1 : 5 : end));
title('Cumulative Rewards vs Episodes')
xlabel('Episode')
ylabel('Cumulative Reward')
%yticks(linspace(0,200,50));
% create a plot of episodes vs steps
figure
plot(B(1, 1 : 5 : end),B(2, 1 : 5 : end));
title('Steps vs Episodes')
xlabel('Episode')
ylabel('Steps')


%normalize q
g=max(max(q));
if g>0, 
    q=100*q/g;
end

% display the shortest path to the goal
Optimal=[];
state=1;
Optimal=horzcat(Optimal,state);

while state~=goalState
    
         [~,optimal]=(max(q(state,:)));
         state = optimal;
         Optimal=horzcat(Optimal,state);         
end

display('Shortest path:')
display(Optimal);

display(exploitCount);
display(exploreCount);

% display de q-matrix
imagesc(q);           
colormap(flipud(gray));
textStrings = num2str(q(:),'%0.2f');  %# Create strings from the matrix values
textStrings = strtrim(cellstr(textStrings));  %# Remove any space padding
[x,y] = meshgrid(1:25);   %# Create x and y coordinates for the strings
hStrings = text(x(:),y(:),textStrings(:),...      %# Plot the strings
                'HorizontalAlignment','center');
midValue = mean(get(gca,'CLim'));  %# Get the middle value of the color range
textColors = repmat(q(:) > midValue,1,3);  %# Choose white or black for the
                                             %#   text color of the strings so
                                             %#   they can be easily seen over
                                             %#   the background color
set(hStrings,{'Color'},num2cell(textColors,2));  %# Change the text colors

set(gca,'XTick',1:25,...                         %# Change the axes tick marks
        'XTickLabel',{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'},...  %#   and tick labels
        'YTick',1:25,...
        'YTickLabel',{'1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25'},...
        'TickLength',[0 0]);