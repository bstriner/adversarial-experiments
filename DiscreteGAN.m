%% Simple GAN for a single 2D point

%% Model parameters
% number of points
k = 3;
% generator
theta_g = sym('theta_g',[k 2]);
% discriminator
h = 32; % hidden layer size
theta_d_W_1 = sym('theta_d_W_1', [2 h]);
theta_d_b_1 = sym('theta_d_b_1', [1 h]);
theta_d_W_2 = sym('theta_d_W_2', [h 1]);
theta_d_b_2 = sym('theta_d_b_2', [1 1]);
theta_d = {theta_d_W_1;theta_d_b_1;theta_d_W_2;theta_d_b_2};
all_params = [{theta_g}; theta_d];
params_flat = FlattenConcat(all_params);

%% Real data
x_real = zeros(k, 2);
for i = 1:k
    angle = i*2*pi/k;
    x_real(i,:)=[sin(angle) cos(angle)];
end

%% Model objectives
% y = sigmoid(dot(sigmoid(dot(x,W1)+b1),W2)+b2)
x = sym('x', [k 2]);
hidden = sigmoid(mtimes(x, theta_d_W_1)+repmat(theta_d_b_1, k, 1));
y = sigmoid(mtimes(hidden, theta_d_W_2)+repmat(theta_d_b_2, k, 1));
y_real = subs(y, x, x_real);
y_fake = subs(y, x, theta_g);
% losses
loss_d = sum(log(y_fake) - log(y_real));
loss_g = sum(- log(y_fake));
% regularization
%reg_weight = 1e-2;
%loss_d = loss_d + reg_weight*sum(theta_d_W.^2);

%% Model updates
lr_g = 3e-3;
lr_d = 1e-1;
diff_g = DiffMatrix(loss_g, theta_g);
theta_g_next = theta_g - lr_g * diff_g;
theta_d_next = cell(size(theta_d));
for i = 1:size(theta_d,1)
    param = theta_d{i};
    theta_d_next{i} = param - lr_d * DiffMatrix(loss_d, param);
end
updates = [{theta_g_next};theta_d_next];
updates_flat = FlattenConcat(updates);

%% Random initialization
theta_g_t = rand(k, 2)*2-1;
theta_d_W_1_t = rand(2, h)*2-1;
theta_d_b_1_t = rand(1, h)*2-1;
theta_d_W_2_t = rand(h, 1)*2-1;
theta_d_b_2_t = rand(1, 1)*2-1;
theta_d_t = {theta_d_W_1_t; theta_d_b_1_t; theta_d_W_2_t; theta_d_b_2_t};
all_params_t = [{theta_g_t}; theta_d_t];
params_flat_t = FlattenConcat(all_params_t);

%% Train model
nb_epoch = 600;
nb_batch = 10;
for epoch = 1:nb_epoch
    % Display parameters
    % disp(theta_g_t);
    GANGraph2('discrete-gan', epoch, ...
        x_real, all_params_t{1}, all_params_t{2}, all_params_t{3}, ...
        all_params_t{4}, all_params_t{5});
    for batch = 1:nb_batch
        all_params_t = SubsCellCell(updates, all_params, all_params_t);
        % params_flat_t = subs(updates_flat, params_flat, params_flat_t);
    end
end