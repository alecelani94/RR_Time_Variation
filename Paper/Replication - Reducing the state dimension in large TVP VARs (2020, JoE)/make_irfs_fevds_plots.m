s_IR = ir_tvpvar( s_B, s_A, 40, 1:175 );
s_FEVD = cumsum( s_IR .^ 2, 3 );
s_FEVD = bsxfun( @rdivide, s_FEVD, sum( s_FEVD, 2 ) );
tper = 1954.5 + .25 * ( p + 2 ):.25:2008.75 - .25 * 40;
valid = valid <= 1;
irfs2nonnews
irfs2news
irfsH0_by_shock
irfsH40_by_shock
fevds2nonnews
fevds2news
fevdsH0_by_shock
fevdsH40_by_shock