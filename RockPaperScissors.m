%% Simple example of Rock Paper Scissors optimization (using softmax)

%% Unconstrained parameters
a_theta = sym('a_theta',[3 1]);
b_theta = sym('b_theta',[3 1]);

%% Softmax so probabilities sum to 1 and are > 0
a = exp(a_theta)/sum(exp(a_theta));
b = exp(b_theta)/sum(exp(b_theta));

%% Rock > Scissors > Paper > Rock
% Rock=1, Paper=2, Scissors=3
a_loss = -(a(1)*b(3)+a(2)*b(1)+a(3)*b(2));
b_loss = -(b(1)*a(3)+b(2)*a(1)+b(3)*a(2));

%% Regularization
% Set to 0 to disable regularization
reg = 0.025;
if reg>0
    a_loss = a_loss + reg*sum(a_theta.^2);
    b_loss = b_loss + reg*sum(b_theta.^2);
end

%% Gradient descent updates
lr=0.1;
a_diff = [diff(a_loss, a_theta(1)); ...
    diff(a_loss, a_theta(2)); ...
    diff(a_loss, a_theta(3))];
b_diff = [diff(b_loss, b_theta(1)); ...
    diff(b_loss, b_theta(2)); ...
    diff(b_loss, b_theta(3))];
a_theta_update = a_theta - lr*a_diff;
b_theta_update = b_theta - lr*b_diff;
% Print equations
% disp(a_theta_update);
% disp(b_theta_update);

%% Random initial parameters
a_theta_t = rand(3,1);
b_theta_t = rand(3,1);

%% Train model
nb_epoch = 2000;
nb_batches = 1;
logs = zeros(nb_epoch, 6);
for epoch=1:nb_epoch
    % Retrieve current parameters
    a_t = vpa(subs(a, a_theta, a_theta_t));
    b_t = vpa(subs(b, b_theta, b_theta_t));
    % Save parameters to log
    logs(epoch,:) = [a_t.' b_t.'];
    % Display parameters in terminal
    % disp([a_t b_t]);
    % Perform nb_batches steps of updates
    for batch=1:nb_batches
        % Calculate next theta
        a_theta_next = vpa(subs(subs(a_theta_update,a_theta, a_theta_t), b_theta, b_theta_t));
        b_theta_next = vpa(subs(subs(b_theta_update,a_theta, a_theta_t), b_theta, b_theta_t));
        % Store next theta
        a_theta_t = a_theta_next;
        b_theta_t = b_theta_next;    
    end
end

%% Write results
csvwrite('RockPaperScissors.csv',logs);

%% Graph results
figure;
hold on;
plot(logs(:,1), '-r');
plot(logs(:,2), '--r');
plot(logs(:,3), ':r');
plot(logs(:,4), '-b');
plot(logs(:,5), '--b');
plot(logs(:,6), ':b');
legend('A - Rock', 'A - Paper', 'A - Scissors', ...
    'B - Rock', 'B - Paper', 'B - Scissors');
print('RockPaperScissors-Graph.png','-dpng');