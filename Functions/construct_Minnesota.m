function [M, C_l, C_c, ps_lam12] = construct_Minnesota(scales,P,lambda)

% lambda(1): constant
% lambda(2): lagged (own)
% lambda(3): lagged (cross)
% lambda(4): contemporaneous
% lambda(5): lag decay

N = length(scales);

M = sparse(N,N*P+1);

temp_l = zeros(N);
C_c    = NaN(N,N-1);
for i = 1:N

    for j = 1:N

        sc_ij = scales(i)/scales(j);

        if i == j

           temp_l(i,j) = sc_ij*lambda(2);
        else
           temp_l(i,j) = sc_ij*lambda(3);
        end

        if i > j

           C_c(i,j) = lambda(4)*sc_ij;
        end
    end
end

C_c = C_c';

C_l = zeros(N,N*P);
for i = 1:P

    C_l(:,(i-1)*N+1:i*N) = temp_l/(i^lambda(5));
end

C_l = [lambda(1)*scales, C_l]';

sq = 1:N^2*P+N;

blk   = reshape(sq,N*P+1,N)';

ps_l1 = zeros(N*P,1);
for i = 1:P

    ps   = (i-1)*N+1:i*N;
    temp = blk(:,ps+1);

    ps_l1(ps) = diag(temp);
end

ps_l1 = sort(ps_l1);
ps_l2 = setdiff(vec(blk(:,2:end)'),ps_l1);

ps_lam12{1} = ps_l1;
ps_lam12{2} = ps_l2;
