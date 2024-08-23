% This function performs the time integration of the backward equation by means
% of the exponential Euler--Magnus method. For more details, see [ACCC24].
%
% [ACCC24] G. Albi, M. Caliari, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

function psi = backward_exp(psi0,rho,u,opts)
  psi = NaN(opts.n,opts.ts+1);
  psi(:,1) = psi0;
  tolkry = opts.tau^2/100;
  mkry = [];
  for i = 1:opts.ts
    rhon = rho(:,opts.ts+2-i);
    psin = psi(:,i);
    un = u(:,i);
    % Assemble linear part
    AA = opts.sigma^2/2*opts.D2b+...
         spdiags(opts.Mp*rhon+(opts.s(rhon)+rhon.*opts.dsdrho(rhon)).*un,...
         0,opts.n,opts.n)*opts.D1b+opts.Mq*spdiags(rhon,0,opts.n,opts.n)*opts.D1b;
    % Assemble nonlinear part
    bb = (opts.gamma/2)*abs(un).^2+1/2*opts.dEdrho(rhon);
    % Do integration step
    [psi(:,i+1),mkry] = kiops(opts.tau,AA,[psin,bb],tolkry,mkry,[],[],false);
  end
end
