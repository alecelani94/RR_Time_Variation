function beta = draw_mean_LRM(y,X,P,Qmu,Q)

% draw mean of a linear regression model
% with general error covariance matrix:

% y = X*beta + e,
% e ~ N(0,P^(-1)),     P error precision
% beta ~ N(mu,Q^(-1)), Qmu = Q*mu prior precision * mean, Q prior precision

N = size(X,2);

if isempty(P)

   XX = X'*X;
   Xy = X'*y;
else

   XX = X'*P*X;
   Xy = X'*P*y;
end

if isempty(Q) % diffuse

   vv = chol( XX, 'lower');
   mm = vv'\(vv\ Xy );

else % informative

    vv = chol( Q + XX, 'lower');

    if isempty(Qmu) % shrink towards 0

       mm = vv'\(vv\ Xy );
    else % shrink towards Qmu

       mm = vv'\(vv\ (Qmu + Xy) );
    end

end

beta = mm + vv'\randn(N,1); % draw
