%% Simple example of Rock Paper Scissors optimization (using barrier)

%% Constrained parameters
a_theta = sym('a_theta',[2 1]);
b_theta = sym('b_theta',[2 1]);

%% scissors = 1-rock-paper
a = [a_theta; 1.0-sum(a_theta)];
b = [b_theta; 1.0-sum(b_theta)];

%% rock > scissors > paper > rock
% rock=1, paper=2, scissors=3
a_loss = -(a(1)*b(3)+a(2)*b(1)+a(3)*b(2));
b_loss = -(b(1)*a(3)+b(2)*a(1)+b(3)*a(2));

%% barrier
w1 = 0.01;
w2 = 0.01;
a_loss = a_loss - w1*sum(log(w2*a));
b_loss = b_loss - w1*sum(log(w2*b));

%% gradient descent updates
lr=0.03;
a_diff = [diff(a_loss, a_theta(1)); diff(a_loss, a_theta(2))];
b_diff = [diff(b_loss, b_theta(1)); diff(b_loss, b_theta(2))];
a_theta_update = a_theta - lr*a_diff;
b_theta_update = b_theta - lr*b_diff;

%% random initial parameters
% random value
a_theta_t = rand(3,1);
b_theta_t = rand(3,1);
% softmax so sum to 1 and > 0
a_theta_t = exp(a_theta_t)/sum(exp(a_theta_t));
b_theta_t = exp(b_theta_t)/sum(exp(b_theta_t));
% only need first two values
a_theta_t = a_theta_t(1:2,1);
b_theta_t = b_theta_t(1:2,1);

%% train model
nb_epoch = 100;
nb_batches = 10;
path='rock-paper-scissors-barrier-output';
for epoch=1:nb_epoch
    % Retrieve current parameters
    a_t = vpa(subs(a, a_theta, a_theta_t));
    b_t = vpa(subs(b, b_theta, b_theta_t));
    % Display parameters in terminal
    disp([a_t b_t]);
    % Draw graph of parameters
    RPSGraph(path, epoch, a_t, b_t);
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
