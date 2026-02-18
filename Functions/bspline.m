function [B, dx] = bspline(x, xl, xr, ndx, bdeg)

    % BSPLINE  Uniform B-spline basis (degree = bdeg) over [xl, xr]
    %
    % Inputs:
    %   x    : column vector of evaluation points
    %  xl   : left boundary of the domain
    %  xr   : right boundary of the domain
    %  ndx  : number of equal subintervals in [xl, xr]
    %  bdeg : spline degree (0 = piecewise constant, 1 = linear, etc.)
    %
    % Outputs:
    %  B    : basis matrix, size length(x) x (ndx + bdeg)
    %  dx   : uniform knot spacing (interval width)

    dx = (xr - xl) / ndx;         % unifo, m knot spacing over [xl, xr]
    t = xl + dx * (-bdeg:ndx-1);  % knot locations: start bdeg steps before xl

    T = (0 * x + 1) * t;
    X = x * (0 * t + 1);
    P = (X - T) / dx;
    B = (T <= X) & (X < (T + dx));
    r = [2:length(t) 1];
    for k = 1:bdeg
        B = (P .* B + (k + 1 - P) .* B(:, r)) / k;
    end
end