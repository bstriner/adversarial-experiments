%% Simple GAN for a single 2D point

%% Model parameters
% generator
theta_g = sym('theta_g',[2 1]);
% discriminator
theta_d_W = sym('theta_d_W',[2 1]);
theta_d_b = sym('theta_d_b');

%% Real data
x_real = [0; 0];

%% Model objectives
% y = sigmoid(dot(x,W)+b)
% y_real = y(x_real)
y_real = sigmoid(mtimes(transpose(x_real), theta_d_W)+theta_d_b);
% y_fake = y(theta_g)
y_fake = sigmoid(mtimes(transpose(theta_g), theta_d_W)+theta_d_b);
% losses
loss_d = log(y_fake) - log(y_real);
loss_g = - log(y_fake);
% regularization
reg_weight = 1e-2;
loss_d = loss_d + reg_weight*sum(theta_d_W.^2);

%% Model updates
lr_g = 5e-2;
lr_d = 1e-1;
diff_g = [diff(loss_g, theta_g(1));diff(loss_g, theta_g(2))];
theta_g_next = theta_g - lr_g * diff_g;
diff_d_W = [diff(loss_d, theta_d_W(1));diff(loss_d, theta_d_W(2))];
theta_d_W_next = theta_d_W - lr_d * diff_d_W;
diff_d_b = diff(loss_d, theta_d_b);
theta_d_b_next = theta_d_b - lr_d * diff_d_b;

%% Random initialization
theta_g_t = rand(2,1);
theta_d_W_t = rand(2, 1);
theta_d_b_t = rand();

%% Train model
nb_epoch = 600;
nb_batch = 1;
nb_batch_d = 10;
for epoch = 1:nb_epoch
    % Display parameters
    disp(theta_g_t);
    GANGraph('simple-gan', epoch, ...
        x_real, theta_g_t, theta_d_W_t, theta_d_b_t);
    for batch = 1:nb_batch
        % Train descriminator nb_batch_d times
        for batch_d = 1:nb_batch_d
            % Calculate updates
            theta_d_W_update = double(vpa(subs(subs(subs(theta_d_W_next, ...
                theta_g, theta_g_t), ...
                theta_d_W, theta_d_W_t), ...
                theta_d_b, theta_d_b_t)));
            theta_d_b_update = double(vpa(subs(subs(subs(theta_d_b_next, ...
                theta_g, theta_g_t), ...
                theta_d_W, theta_d_W_t), ...
                theta_d_b, theta_d_b_t)));
            % Perform updates
            theta_d_W_t = theta_d_W_update;
            theta_d_b_t = theta_d_b_update;
        end
        % Train generator once
        theta_g_update = double(vpa(subs(subs(subs(theta_g_next, ...
            theta_g, theta_g_t), ...
            theta_d_W, theta_d_W_t), ...
            theta_d_b, theta_d_b_t)));
        theta_g_t = theta_g_update;
    end
end