function [h0, g] = draw_SV_Nocenter(y,h0,g,B,iVh_0,iVh,p_gm,m_gm,s_gm)

h = h0 + B*g;

T = length(h);

% sample S from a 10-point distrete distribution
liketemp = normpdf(repmat(y,1,10),repmat(h,1,10)+ ...
           repmat(m_gm,T,1), repmat(sqrt(s_gm),T,1));

q = repmat(p_gm,T,1).*liketemp;
q = q./repmat(sum(q,2),1,10);
S = 10 - sum(repmat(rand(T,1),1,10)<cumsum(q,2),2)+1;

d   = m_gm(S)';
iVe = sparse(1:T,1:T,1./s_gm(S)');

h0 = draw_mean_LRM(y-d-B*g,ones(T,1),iVe,[],iVh_0);
g  = draw_mean_LRM(y-d-h0,B,iVe,[],iVh);
