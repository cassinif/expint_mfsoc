% This function performs the time integration of the forward equation by means
% of the exponential Euler method. For more details, see [ACCC24].
%
% [ACCC24] G. Albi, M. Caliari, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

function rho = forward_exp(rho0,u,opts)
  rho = NaN(opts.n,opts.ts+1);
  rho(:,1) = rho0;
  tolkry = opts.tau^2/100;
  mkry = [];
  for i = 1:opts.ts
    un = u(:,opts.ts+2-i);
    rhon = rho(:,i);
    srhon = opts.s(rhon);
    % Assemble linear part
    AA = opts.sigma^2/2*opts.D2f;
    % Assemble nonlinear part
    bb = -(opts.D1t*opts.Mp*rhon).*rhon-...
         (opts.Mp*rhon).*(opts.D1f*rhon)-...
         (opts.D1t*un).*(rhon.*srhon)-...
         un.*srhon.*(opts.D1f*rhon)-...
         un.*rhon.*(opts.D1t*srhon);
    bcl = -(2/(opts.sigma^2))*(opts.Mp(1,:)*rhon+srhon(1)*un(1))^2*rhon(1)-...
           (2/opts.h)*(opts.Mp(1,:)*rhon+srhon(1)*un(1))*rhon(1);
    bcr = -(2/(opts.sigma^2))*(opts.Mp(opts.n,:)*rhon+...
           srhon(opts.n)*un(opts.n))^2*rhon(opts.n)+...
           (2/opts.h)*(opts.Mp(opts.n,:)*rhon+srhon(opts.n)*un(opts.n))*rhon(opts.n);
    bb(1) = bb(1) + bcl;
    bb(opts.n) = bb(opts.n) + bcr;
    % Do integration step
    [rho(:,i+1),mkry] = kiops(opts.tau,AA,[rhon,bb],tolkry,mkry,[],[],false);
  end
end
