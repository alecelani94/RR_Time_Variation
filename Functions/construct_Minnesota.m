function [C_l, C_l_nolam, C_c, C_c_nolam] = construct_Minnesota(scales,P,lambda)

% lambda(1): constant
% lambda(2): lagged (own)
% lambda(3): lagged (cross)
% lambda(4): contemporaneous
% lambda(5): lag decay

N = length(scales);

C_l       = zeros(N,N*P);
C_l_nolam = zeros(N,N*P);
for p = 1:P

    s     = (p-1)*N;
    decay = p^lambda(5);
    for i = 1:N

        for j = 1:N

            sc_ij = scales(i)/scales(j);

            if i == j

                C_l(i,s+j) = sc_ij * lambda(2) / decay;
            else
                C_l(i,s+j) = sc_ij * lambda(3) / decay;
            end
            C_l_nolam(i,s+j) = sc_ij / decay;
        end
    end
end

C_c       = NaN(N,N);
C_c_nolam = NaN(N);
for i = 2:N
    for j = 1:i-1

        sc_ij = scales(i) / scales(j);

        C_c(i,j)       = lambda(4) * sc_ij;
        C_c_nolam(i,j) = sc_ij;
    end
end

C_l       = [lambda(1)*scales, C_l];
C_l_nolam = [lambda(1)*scales, C_l_nolam];
