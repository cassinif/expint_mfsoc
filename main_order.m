clear all
close all

% This is the main file to test the order of convergence of the exponential
% integrators as presented in the examples of the manuscript [ACCC24].
% The experiment is selected by the string encoded in the variable ex.
% In particular, the admitted values are:
% - opsz: Control in opinion dynamics (Sznajd model)
% - cdfe: Crowd dynamics (fast exit scenario)
%
% [ACCC24] G. Albi, M. Caliari, F. Cassini, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

addpath(genpath('aux'))

ex = 'cdfe'; % String to choose the example. See above for the admitted values.

switch ex
  case 'opsz'
    disp('--- Order check in opinion dynamics example ---')

    o.n = 200;
    nfix = 30;
    deltafix = 0.1;
    tsrif = 40000;
    tsrange = 300:100:700;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 0;
    o.sigma = sqrt(2*1e-2);
    o.gamma = 0.5;
    T = 4;
    xd = -0.5;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    o.s = @(rho) ones(o.n,1);
    o.dsdrho = @(rho) zeros(o.n,1);
    betasn = -1;
    Pfun = @(x,y) betasn*(1-x.^2);
    rhoplus = @(x,a,b) max(-(x/b).^2+a,0);
    rho0fun = @(x) rhoplus(x+0.75,0.05,0.5) + rhoplus(x-0.5,0.15,1);
    psi0 = zeros(o.n,1);

  case 'cdfe'
    disp('--- Order check in crowd dynamics example ---')

    o.n = 200;
    nfix = 20;
    deltafix = 0.1;
    tsrif = 5000;
    tsrange = 300:100:700;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 10;
    o.sigma = sqrt(2*2e-2);
    o.gamma = 1;
    T = 2;
    Efun = @(x,rho) rho;
    dEdrhofun = @(x,rho) ones(o.n,1);
    o.s = @(rho) 1-rho;
    o.dsdrho = @(rho) -ones(o.n,1);
    Pfun = @(x,y) 0;
    rho0fun = @(x) 0.9*exp(-100*(x+0.4).^2)+0.65*exp(-150*x.^2);
    psi0 = zeros(o.n,1);

  otherwise
    error('Example not known')

end

x = linspace(xl,xr,o.n).';
o.h = (xr-xl)/(o.n-1);

D1 = spdiags(ones(o.n,1)*[-1,0,1]/(2*o.h),-1:1,o.n,o.n);
D2 = spdiags(ones(o.n,1)*[1,-2,1]/(o.h^2),-1:1,o.n,o.n);

% Matrices for forward integration
o.D1f = D1;
o.D2f = D2;
o.D1f(1,1) = 2*o.beta/(o.sigma^2);
o.D1f(1,2) = 0;
o.D1f(o.n,o.n-1) = 0;
o.D1f(o.n,o.n) = -2*o.beta/(o.sigma^2);
o.D2f(1,1) = (-2-4*o.h*o.beta/(o.sigma^2))/(o.h^2);
o.D2f(1,2) = 2/(o.h^2);
o.D2f(o.n,o.n-1) = 2/(o.h^2);
o.D2f(o.n,o.n) =(-2-4*o.h*o.beta/(o.sigma^2))/(o.h^2);

% Matrices for backward integration
o.D1b = D1;
o.D2b = D2;
o.D1b(1,1) = 2*o.beta/(o.sigma^2);
o.D1b(1,2) = 0;
o.D1b(o.n,o.n-1) = 0;
o.D1b(o.n,o.n) = -2*o.beta/(o.sigma^2);
o.D2b(1,1) = (-2-4*o.h*o.beta/(o.sigma^2))/(o.h^2);
o.D2b(1,2) = 2/(o.h^2);
o.D2b(o.n,o.n-1) = 2/(o.h^2);
o.D2b(o.n,o.n) =(-2-4*o.h*o.beta/(o.sigma^2))/(o.h^2);

% Differentiation matrices
o.D1t = D1;
o.D2t = D2;
o.D1t(1,1:3) = [-3,4,-1]/(2*o.h);
o.D1t(o.n,o.n-2:o.n) = [1,-4,3]/(2*o.h);
o.D2t(1,1:4) = [2,-5,4,-1]/(o.h^2);
o.D2t(o.n,o.n-3:o.n) = [-1,4,-5,2]/(o.h^2);

[X,Y] = ndgrid(x);
w = o.h*[1/2,ones(1,o.n-2),1/2];
W = repmat(w,o.n,1);
o.Mp = W.*Pfun(X,Y).*(Y-X);
o.Mq = W.*Pfun(Y,X).*(X-Y);

o.ts = tsrif;
o.tau = T/o.ts;

rho0 = rho0fun(x);

if strcmp(ex,'opsz')
  norm_rho0 = trapz(x,rho0);
  rho0 = rho0/norm_rho0;
end

E = @(rho) Efun(x,rho);
o.dEdrho = @(rho) dEdrhofun(x,rho);

rho = repmat(rho0,1,o.ts+1);
psi = repmat(psi0,1,o.ts+1);
u = -(o.s(fliplr(rho))/o.gamma).*(o.D1t*psi);

trange = linspace(0,T,o.ts+1);

disp('Computing reference solution...')
niter = 0;
while niter < nfix
  niter = niter + 1;

  disp(sprintf('Iteration %i out of %i',niter,nfix))
  if strcmp(ex,'opsz')
    rho = forward_exp_sone(rho0,u,o);
    psi = backward_exp_sone(psi0,rho,u,o);
  else
    rho = forward_exp(rho0,u,o);
    psi = backward_exp(psi0,rho,u,o);
  end

  u = u-deltafix*(o.gamma*u+o.s(fliplr(rho)).*(o.D1t*psi));
  %Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho));
end
disp('Reference solution computed!')

rhoref = rho(:,end);
psiref = psi(:,end);

counter = 0;
for ts = tsrange
  disp(sprintf('Simulation with m = %i time steps',ts))
  counter = counter + 1;
  o.ts = ts;
  o.tau = T/o.ts;

  rho = repmat(rho0,1,o.ts+1);
  psi = repmat(psi0,1,o.ts+1);
  u = -(o.s(fliplr(rho))/o.gamma).*(o.D1t*psi);

  trange = linspace(0,T,o.ts+1);

  niter = 0;
  while niter < nfix
    niter = niter + 1;
    disp(sprintf('Iteration %i out of %i',niter,nfix))
    if strcmp(ex,'opsz')
      rho = forward_exp_sone(rho0,u,o);
      psi = backward_exp_sone(psi0,rho,u,o);
    else
      rho = forward_exp(rho0,u,o);
      psi = backward_exp(psi0,rho,u,o);
    end
    u = u-deltafix*(o.gamma*u+o.s(fliplr(rho)).*(o.D1t*psi));
    %Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho));
  end

  err_rho(counter) = norm(rho(:,end)-rhoref,inf)/norm(rhoref,inf);
  err_psi(counter) = norm(psi(:,end)-psiref,inf)/norm(psiref,inf);

end

figure;
subplot(1,2,1)
loglog(tsrange,err_rho,'xr',tsrange,err_rho(end)*(tsrange/tsrange(end)).^(-1),'--k')
ylim([1e-3,1e-2])
legend('Error of rho(T)')
xlabel('m')
ylabel('Relative error')
subplot(1,2,2)
loglog(tsrange,err_psi,'ob',tsrange,err_psi(end)*(tsrange/tsrange(end)).^(-1),'--k')
ylim([1e-3,1e-2])
legend('Error of psi(0)')
xlabel('m')
ylabel('Relative error')

rmpath(genpath('aux'))
