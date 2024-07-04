function psi = backward_exp_sone(psi0,rho,u,o)
  psi = NaN(o.n,o.ts+1);
  psi(:,1) = psi0;
  tolkry = o.tau^2/100;
  mkry = [];
  for i = 1:o.ts
    rhon = rho(:,o.ts+2-i);
    psin = psi(:,i);
    un = u(:,i);
    AA = o.sigma^2/2*o.D2b+...
         spdiags(o.Mp*rhon+un,0,o.n,o.n)*o.D1b+...
         o.Mq*spdiags(rhon,0,o.n,o.n)*o.D1b;
    bb = (o.gamma/2)*abs(un).^2+...
         1/2*o.dEdrho(rhon);
    [psi(:,i+1),mkry] = kiops(o.tau,AA,[psin,bb],tolkry,mkry,[],[],false);
  end
end
