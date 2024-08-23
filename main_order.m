clear all
close all

% This is the main file to test the order of convergence of the exponential
% integrators as presented in the examples of the manuscript [ACCC24].
% The experiment is selected by the string encoded in the variable ex.
% In particular, the admitted values are:
% - opsz: Control in opinion dynamics (Sznajd model)
% - cdfe: Crowd dynamics (fast exit scenario)
%
% For insights about the parameters/functions involved, see the script main.m.
%
% [ACCC24] G. Albi, M. Caliari, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

addpath(genpath('auxiliary'))

ex = 'cdfe'; % String to choose the example. See above for the admitted values.

switch ex
  case 'opsz'
    disp('--- Order check in opinion dynamics example ---')

    opts.n = 200;
    nfix = 30; % number of gradient descent iterations
    deltafix = 0.1; % step size gradient descent 
    tsrif = 40000; % number of time steps for the reference solution
    tsrange = 300:100:700; % number of time steps for computing errors

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 0;
    opts.sigma = sqrt(2e-2);
    opts.gamma = 0.5;
    T = 4;
    xd = -0.5;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    opts.s = @(rho) ones(opts.n,1);
    opts.dsdrho = @(rho) zeros(opts.n,1);
    betasn = -1;
    Pfun = @(x,y) betasn*(1-x.^2);
    rhoplus = @(x,a,b) max(-(x/b).^2+a,0);
    rho0fun = @(x) rhoplus(x+0.75,0.05,0.5) + rhoplus(x-0.5,0.15,1);
    psiT = zeros(opts.n,1);

  case 'cdfe'
    disp('--- Order check in crowd dynamics example ---')

    opts.n = 200;
    nfix = 20; % number of gradient descent iterations
    deltafix = 0.1; % step size gradient descent
    tsrif = 5000; % number of time steps for the reference solution
    tsrange = 300:100:700; % number of time steps for computing errors

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 10;
    opts.sigma = sqrt(4e-2);
    opts.gamma = 1;
    T = 2;
    Efun = @(x,rho) rho;
    dEdrhofun = @(x,rho) ones(opts.n,1);
    opts.s = @(rho) 1-rho;
    opts.dsdrho = @(rho) -ones(opts.n,1);
    Pfun = @(x,y) 0;
    rho0fun = @(x) 0.9*exp(-100*(x+0.4).^2)+0.65*exp(-150*x.^2);
    psiT = zeros(opts.n,1);

  otherwise
    error('Example not known')

end

x = linspace(xl,xr,opts.n).';
opts.h = (xr-xl)/(opts.n-1);

D1 = spdiags(ones(opts.n,1)*[-1,0,1]/(2*opts.h),-1:1,opts.n,opts.n);
D2 = spdiags(ones(opts.n,1)*[1,-2,1]/(opts.h^2),-1:1,opts.n,opts.n);

% Matrices for forward integration
opts.D1f = D1;
opts.D2f = D2;
opts.D1f(1,1) = 2*opts.beta/(opts.sigma^2);
opts.D1f(1,2) = 0;
opts.D1f(opts.n,opts.n-1) = 0;
opts.D1f(opts.n,opts.n) = -2*opts.beta/(opts.sigma^2);
opts.D2f(1,1) = (-2-4*opts.h*opts.beta/(opts.sigma^2))/(opts.h^2);
opts.D2f(1,2) = 2/(opts.h^2);
opts.D2f(opts.n,opts.n-1) = 2/(opts.h^2);
opts.D2f(opts.n,opts.n) =(-2-4*opts.h*opts.beta/(opts.sigma^2))/(opts.h^2);

% Matrices for backward integration
opts.D1b = D1;
opts.D2b = D2;
opts.D1b(1,1) = 2*opts.beta/(opts.sigma^2);
opts.D1b(1,2) = 0;
opts.D1b(opts.n,opts.n-1) = 0;
opts.D1b(opts.n,opts.n) = -2*opts.beta/(opts.sigma^2);
opts.D2b(1,1) = (-2-4*opts.h*opts.beta/(opts.sigma^2))/(opts.h^2);
opts.D2b(1,2) = 2/(opts.h^2);
opts.D2b(opts.n,opts.n-1) = 2/(opts.h^2);
opts.D2b(opts.n,opts.n) =(-2-4*opts.h*opts.beta/(opts.sigma^2))/(opts.h^2);

% Differentiation matrices
opts.D1t = D1;
opts.D2t = D2;
opts.D1t(1,1:3) = [-3,4,-1]/(2*opts.h);
opts.D1t(opts.n,opts.n-2:opts.n) = [1,-4,3]/(2*opts.h);
opts.D2t(1,1:4) = [2,-5,4,-1]/(opts.h^2);
opts.D2t(opts.n,opts.n-3:opts.n) = [-1,4,-5,2]/(opts.h^2);

[X,Y] = ndgrid(x);
w = opts.h*[1/2,ones(1,opts.n-2),1/2];
W = repmat(w,opts.n,1);
opts.Mp = W.*Pfun(X,Y).*(Y-X);
opts.Mq = W.*Pfun(Y,X).*(X-Y);

opts.ts = tsrif;
opts.tau = T/opts.ts;

rho0 = rho0fun(x);

if strcmp(ex,'opsz')
  norm_rho0 = trapz(x,rho0);
  rho0 = rho0/norm_rho0;
end

E = @(rho) Efun(x,rho);
opts.dEdrho = @(rho) dEdrhofun(x,rho);

rho = repmat(rho0,1,opts.ts+1);
psi = repmat(psiT,1,opts.ts+1);
u = -(opts.s(fliplr(rho))/opts.gamma).*(opts.D1t*psi);

trange = linspace(0,T,opts.ts+1);

disp('Computing reference solution...')
niter = 0;
while niter < nfix
  niter = niter + 1;

  disp(sprintf('Iteration %i out of %i',niter,nfix))
  if strcmp(ex,'opsz')
    rho = forward_exp_sone(rho0,u,opts);
    psi = backward_exp_sone(psiT,rho,u,opts);
  else
    rho = forward_exp(rho0,u,opts);
    psi = backward_exp(psiT,rho,u,opts);
  end

  u = u-deltafix*(opts.gamma*u+opts.s(fliplr(rho)).*(opts.D1t*psi));
end
disp('Reference solution computed!')

rhoref = rho(:,end);
norm_rhoref = norm(rhoref,inf);
psiref = psi(:,end);
norm_psiref = norm(psiref,inf);

counter = 0;
for ts = tsrange
  disp(sprintf('Simulation with m = %i time steps',ts))
  counter = counter + 1;
  opts.ts = ts;
  opts.tau = T/opts.ts;

  rho = repmat(rho0,1,opts.ts+1);
  psi = repmat(psiT,1,opts.ts+1);
  u = -(opts.s(fliplr(rho))/opts.gamma).*(opts.D1t*psi);

  trange = linspace(0,T,opts.ts+1);

  niter = 0;
  while niter < nfix
    niter = niter + 1;

    disp(sprintf('Iteration %i out of %i',niter,nfix))
    if strcmp(ex,'opsz')
      rho = forward_exp_sone(rho0,u,opts);
      psi = backward_exp_sone(psiT,rho,u,opts);
    else
      rho = forward_exp(rho0,u,opts);
      psi = backward_exp(psiT,rho,u,opts);
    end

    u = u-deltafix*(opts.gamma*u+opts.s(fliplr(rho)).*(opts.D1t*psi));
  end

  err_rho(counter) = norm(rho(:,end)-rhoref,inf)/norm_rhoref;
  err_psi(counter) = norm(psi(:,end)-psiref,inf)/norm_psiref;

end

% Plot errors in density and adjoint
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
drawnow

rmpath(genpath('auxiliary'))
