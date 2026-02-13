time = 1:T;

% --- 1. Simulated data ---
figure('Name', 'Simulated Data');
for i = 1:N

    subplot(N, 1, i);
    plot(time, Y(:,i), 'r', 'LineWidth', 1.25);
    title(sprintf('y_{%d}', i));
    xlim([1 T]); 
    grid on;
end

% --- 2. Auxiliary TV parameters ---
figure('Name', 'Auxiliary Parameters');

subplot(2,1,1);
plot(time, a, 'LineWidth', 1.25);
title('a_t');
legend(arrayfun(@(i) sprintf('a_{%d}', i), 1:N, 'UniformOutput', false));
xlim([1 T]); 
grid on;

subplot(2,1,2);
plot(time, c, 'LineWidth', 1.25);
title('c_t');
legend(arrayfun(@(i) sprintf('c_{%d}', i), 1:N, 'UniformOutput', false));
xlim([1 T]); 
grid on;

% --- 3. TV coefficients Phi_t ---
figure('Name', 'TV Coefficients');
k = 0;
for i = 1:N
    for j = 1:N

        k = k + 1;
        subplot(N, N, k);
        phi_ij = squeeze(Phi_tv(i, j, :));
        plot(time, phi_ij, 'r', 'LineWidth', 1.25); hold on;
        yline(Phi_bar(i, j), 'k-.', 'LineWidth', 1);
        title(sprintf('\\phi_{%d%d}', i, j));
        xlim([1 T]); 
        grid on;
    end
end

% --- 4. Max eigenvalue over time ---
figure('Name', 'Max Eigenvalue');
plot(time, max_eigt, 'r', 'LineWidth', 1.25); 
hold on;
xlim([1 T]); 
grid on;