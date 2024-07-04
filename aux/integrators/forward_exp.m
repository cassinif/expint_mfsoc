function rho = forward_exp(rho0,u,o)
  rho = NaN(o.n,o.ts+1);
  rho(:,1) = rho0;
  tolkry = o.tau^2/100;
  mkry = [];
  for i = 1:o.ts
    un = u(:,o.ts+2-i);
    rhon = rho(:,i);
    srhon = o.s(rhon);
    AA = o.sigma^2/2*o.D2f;
    bb = -(o.D1t*o.Mp*rhon).*rhon-...
         (o.Mp*rhon).*(o.D1f*rhon)-...
         (o.D1t*un).*(rhon.*srhon)-...
         un.*srhon.*(o.D1f*rhon)-...
         un.*rhon.*(o.D1t*srhon);
    bcl = -(2/(o.sigma^2))*(o.Mp(1,:)*rhon+srhon(1)*un(1))^2*rhon(1)-(2/o.h)*(o.Mp(1,:)*rhon+srhon(1)*un(1))*rhon(1);
    bcr = -(2/(o.sigma^2))*(o.Mp(o.n,:)*rhon+srhon(o.n)*un(o.n))^2*rhon(o.n)+(2/o.h)*(o.Mp(o.n,:)*rhon+srhon(o.n)*un(o.n))*rhon(o.n);
    bb = bb+[bcl;zeros(o.n-2,1);bcr];
    [rho(:,i+1),mkry] = kiops(o.tau,AA,[rhon,bb],tolkry,mkry,[],[],false);
  end
end
