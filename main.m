clear all
close all

% This is the main file to reproduce the numerical examples described in
% the manuscript [ACCC24]. The specific experiment is determined by the string
% encoded in the variable ex. In particular, the admitted values are:
% - opsz: Control in opinion dynamics (Sznajd model)
% - ophk: Control in opinion dynamics (Hegselmann--Krause model)
% - cdfe: Crowd dynamics (fast exit scenario)
% - mtoc: Mass transfer via optimal control
%
% The mean field optimal control problem in [ACCC24] aims at finding a control
% u which minimizes a given functional J constrained to a PDE for the density
% function rho (forward equation). The problem is tackled numerically by a gradient
% descent approach which employs an auxiliary PDE for the adjoint variable psi
% (backward equation).
%
% The parameters/functions of the mean field optimal control problem are encoded
% in the variables as described in the following.
% - Jfunf: discretized functional
% - Efun: running cost
% - opts.gamma: penalization parameter
% - Cter: terminal cost (if nonzero)
% - Pfun: nonlocal interaction operator
% - opts.s: selective function
% - opts.sigma: Brownian motion related parameter
% - rho0: initial condition for density equation
% - beta: magnitude of boundary flux
% - opts.dsdrho: derivative of the selective function wrt rho
% - dEdrhofun: derivative of the running cost wrt rho
% - dCterdrho: derivative of the terminal cost wrt rho (if nonzero)
%
% The initial guess for the control is determined by the function uguess.
% Additional parameters for the specific example under consideration (e.g., the
% target point for the Sznajd model) are set in relevant parts of the code.
%
% The spatial domain is discretized with opts.n equispaced nodes in the interval
% [xl,xr], and we employ second order centered finite differences.
% The relevant matrices are encoded in the variables as described in the following.
% - opts.D1f: first derivative for forward equation encapsulating boundary conditions
% - opts.D2f: second derivative for forward equation encapsulating boundary conditions
% - opts.D1b: first derivative for backward equation encapsulating boundary conditions
% - opts.D2b: second derivative for backward equation encapsulating boundary conditions
% - opts.D1t: first derivative
% - opts.D2t: second derivative
% Also, the integrals on the spatial domain are always discretized with the
% trapezoidal quadrature rule.
%
% The temporal integration of the forward-backward equations is performed up
% to the final time T with number of time steps encoded in the variable opts.ts
% (i.e., using a constant time step size). The employed exponential integrators
% are implemented in the relevant functions as described in the following.
% - auxiliary/integrators/forward_exp_sone.m: exponential Euler--Magnus for the forward
% equation with selective function equal to one
% - auxiliary/integrators/backward_exp_sone.m: exponential Euler--Magnus for the backward
% equation with selective function equal to one
% - auxiliary/integrators/forward_exp.m: exponential Euler for the forward equation
% - auxiliary/integrators/backward_exp.m: exponential Euler--Magnus for the backward equation
%
% The step size of the gradient descent (encoded in the variable deltav) is
% determined by a discretized version of the Barzilai-Borwein method with
% cut-off parameters deltamin and deltamax. The algorithm stops when the difference
% between the functional values at two consecutive iterations is below the tolerance
% specified in the variable tolsw.
%
% The code has been tested with MathWorks MATLAB(R) R2019a, R2022a, R2023a, and R2024a on
% Windows and Linux distributions.
% The execution times using MathWorks MATLAB(R) R2022a with an Intel(R) Core(TM)
% i7-10750H CPU with six physical cores and 16GB of RAM are:
% - opsz: about 40 seconds
% - ophk: about 15 seconds
% - cdfe: about 45 seconds
% - mtoc: about 75 seconds
%
% [ACCC24] G. Albi, M. Caliari, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

addpath(genpath('auxiliary'))

ex = 'opsz'; % String to choose the example. See above for the admitted values.

switch ex
  case 'opsz'
    disp('--- Control in opinion dynamics: Sznajd model ---')

    opts.n = 800;
    opts.ts = 200;

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 0;
    opts.sigma = sqrt(2e-2);
    opts.gamma = 0.5;
    T = 8;
    xd = -0.5;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    opts.s = @(rho) ones(opts.n,1);
    opts.dsdrho = @(rho) zeros(opts.n,1);
    betasz = -1;
    Pfun = @(x,y) betasz*(1-x.^2);
    rhoplus = @(x,a,b) max(-(x/b).^2+a,0);
    rho0fun = @(x) rhoplus(x+0.75,0.05,0.5) + rhoplus(x-0.5,0.15,1);
    psiT = zeros(opts.n,1);

    % Gradient descent parameters
    tolsw = 2e-3;
    deltav = 0.1;
    deltamin = 0.1;
    deltamax = 2;

    % Guess for the initial control
    uguess = @(t,x) zeros(size(x,1),size(t,2));
    % a way to define time/space dependent initial guess
    %uguess = @(t,x) 1e-6*exp(-t.^2).*(xr-x).*(x-xl).*(xd-x);

  case 'ophk'
    disp('--- Control in opinion dynamics: Hegselmann--Krause model ---')

    opts.n = 1000;
    opts.ts = 100;

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 0;
    opts.sigma = sqrt(2e-3);
    opts.gamma = 2.5;
    T = 10;
    xd = 0;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    opts.s = @(rho) ones(opts.n,1);
    opts.dsdrho = @(rho) zeros(opts.n,1);
    kappa = 0.15;
    Pfun = @(x,y) (y<=(x+kappa)).*(y>=(x-kappa));
    epsilon = 0.01;
    rho0fun = @(x) 0.5+epsilon*(1-x.^2);
    psiT = zeros(opts.n,1);

    % Gradient descent parameters
    tolsw = 2e-3;
    deltav = 0.01;
    deltamin = 0.01;
    deltamax = 0.03;

    % Guess for the initial control
    uguess = @(t,x) zeros(size(x,1),size(t,2));

  case 'cdfe'
    disp('--- Crowd dynamics: fast exit of two groups ---')

    opts.n = 1000;
    opts.ts = 250;

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 10;
    opts.sigma = sqrt(4e-2);
    opts.gamma = 1;
    T = 3;
    Efun = @(x,rho) rho;
    dEdrhofun = @(x,rho) ones(opts.n,1);
    opts.s = @(rho) 1-rho;
    opts.dsdrho = @(rho) -ones(opts.n,1);
    Pfun = @(x,y) 0;
    rho0fun = @(x) 0.9*exp(-100*(x+0.4).^2)+0.65*exp(-150*x.^2);
    psiT = zeros(opts.n,1);

    % Gradient descent parameters
    tolsw = 2e-3;
    deltav = 0.01;
    deltamin = 0.01;
    deltamax = 0.2;

    % Guess for the initial control
    uguess = @(t,x) zeros(size(x,1),size(t,2));

  case 'mtoc'
    disp('--- Mass transfer via optimal control ---')

    opts.n = 1000;
    opts.ts = 200;

    % Model parameters
    xl = -1;
    xr = 1;
    opts.beta = 0;
    opts.sigma = sqrt(2e-2);
    opts.gamma = 0.1;
    T = 3;
    mu1 = 0.5;
    sigma1 = 0.1;
    mu2 = -0.3;
    sigma2 = 0.15;
    rhobarfun = @(x) exp(-(x-mu1).^2/(2*sigma1^2)) + exp(-(x-mu2).^2/(2*sigma2^2));
    Efun = @(rho,rhobar) (rho-rhobar).^2;
    dEdrhofun = @(rho,rhobar) 2*(rho-rhobar);
    opts.s = @(rho) ones(opts.n,1);
    opts.dsdrho = @(rho) zeros(opts.n,1);
    betasz = -0.05;
    Pfun = @(x,y) betasz*(1-x.^2);
    Cter = @(rhoT,rhobar) (rhoT-rhobar).^2;
    dCterdrho = @(rhoT,rhobar) 2*(rhoT-rhobar);
    mu0 = 0;
    sigma0 = 0.1;
    rho0fun = @(x) exp(-(x-mu0).^2/(2*sigma0^2));
    psiT = zeros(opts.n,1);

    % Gradient descent parameters
    tolsw = 2e-3;
    deltav = 0.03;
    deltamin = 0.015;
    deltamax = 0.015;

    % Guess for the initial control
    uguess = @(t,x) zeros(size(x,1),size(t,2));

  otherwise
    error('Example not known')

end

% Space discretization
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

opts.tau = T/opts.ts;
trange = linspace(0,T,opts.ts+1);

rho0 = rho0fun(x);
switch ex
  case {'opsz','ophk'}
    norm_rho0 = trapz(x,rho0);
    rho0 = rho0/norm_rho0;
    E = @(rho) Efun(x,rho);
    opts.dEdrho = @(rho) dEdrhofun(x,rho);

    % Functional expression
    Jfunf = @(rho,u) 1/2*trapz(trange,trapz(x,E(rho)+opts.gamma*abs(fliplr(u)).^2.*rho));

  case 'cdfe'
    E = @(rho) Efun(x,rho);
    opts.dEdrho = @(rho) dEdrhofun(x,rho);

    % Functional expression
    Jfunf = @(rho,u) 1/2*trapz(trange,trapz(x,E(rho)+opts.gamma*abs(fliplr(u)).^2.*rho));

  case 'mtoc'
    norm_rho0 = trapz(x,rho0);
    rho0 = rho0/norm_rho0;
    rhobar = rhobarfun(x);
    norm_rhobar = trapz(x,rhobar);
    rhobar = rhobar/norm_rhobar;
    E = @(rho) Efun(rho,rhobar);
    opts.dEdrho = @(rho) dEdrhofun(rho,rhobar);

    % Functional expression
    Jfunf = @(rho,u) 1/2*trapz(trange,trapz(x,E(rho)+opts.gamma*abs(fliplr(u)).^2.*rho))+...
                     1/2*trapz(x,Cter(rho(:,opts.ts+1),rhobar));

  otherwise
    error('Example not known')
end

rho = repmat(rho0,1,opts.ts+1);
psi = repmat(psiT,1,opts.ts+1);

[TT,XX] = ndgrid(x,fliplr(trange));
u = uguess(TT,XX);

Jtmp = Jfunf(rho,u);

niter = 0;

disp('Doing integration (displaying functional values)...')
switch ex
  case {'opsz','ophk'}
    tic
    niter = niter + 1;

    rho = forward_exp_sone(rho0,u,opts);
    psi = backward_exp_sone(psiT,rho,u,opts);

    dJdun = opts.gamma*u+opts.D1t*psi;
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = Jfunf(rho,u)
    J_hist(niter) = Jfun;

    % Gradient descent
    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;

      rho = forward_exp_sone(rho0,u,opts); % Forward integration
      psi = backward_exp_sone(psiT,rho,u,opts); % Backward integration

      % Control update
      dJdunm1 = dJdun;
      dJdun = opts.gamma*u+opts.D1t*psi;
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/...
               trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;

      Jfun = Jfunf(rho,u)
      errsw = abs(Jfun-Jtmp);
      J_hist(niter) = Jfun;
      Jtmp = Jfun;
    end
    cpu_time=toc;

  case 'cdfe'
    tic
    niter = niter + 1;

    rho = forward_exp(rho0,u,opts);
    psi = backward_exp(psiT,rho,u,opts);

    dJdun = opts.gamma*u+opts.s(fliplr(rho)).*(opts.D1t*psi);
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = Jfunf(rho,u)
    J_hist(niter) = Jfun;

    % Gradient descent
    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;

      rho = forward_exp(rho0,u,opts); % Forward integration
      psi = backward_exp(psiT,rho,u,opts); % Backward integration

      % Control update
      dJdunm1 = dJdun;
      dJdun = opts.gamma*u+opts.s(fliplr(rho)).*(opts.D1t*psi);
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/...
               trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;

      Jfun = Jfunf(rho,u)
      errsw = abs(Jfun-Jtmp);
      J_hist(niter) = Jfun;
      Jtmp = Jfun;
    end
    cpu_time=toc;

  case 'mtoc'
    tic
    niter = niter + 1;

    rho = forward_exp_sone(rho0,u,opts);
    psiT = dCterdrho(rho(:,opts.ts+1),rhobar)/2;
    psi = backward_exp_sone(psiT,rho,u,opts);

    dJdun = opts.gamma*u+opts.D1t*psi;
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = Jfunf(rho,u)
    J_hist(niter) = Jfun;

    % Gradient descent
    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;

      rho = forward_exp_sone(rho0,u,opts); % Forward integration
      psiT = dCterdrho(rho(:,opts.ts+1),rhobar)/2;
      psi = backward_exp_sone(psiT,rho,u,opts); % Backward integration

      % Control update
      dJdunm1 = dJdun;
      dJdun = opts.gamma*u+opts.D1t*psi;
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/...
               trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;

      Jfun = Jfunf(rho,u)
      errsw = abs(Jfun-Jtmp);
      J_hist(niter) = Jfun;
      Jtmp = Jfun;
    end
    cpu_time=toc;

  otherwise
    error('Example not known')
end
disp('Integration done!')

disp(sprintf('Number of iterations: %i',niter))
disp(sprintf('Functional value: %.4f',Jfun))
disp(sprintf('Wall clock time: %.2f',cpu_time))

viridis = color_viridis(); % colormap

switch ex
  case {'opsz','ophk'}
    % Plot density at initial and final times and target
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b',[xd,xd],[0 max(rho(:,end))],'-g');
    ylim([0 max(rho(:,end))])
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density','x_d - Target')
    drawnow

    % Plot density and control
    figure;
    ax1=subplot(1,2,1);
    p1=pcolor(x,trange,rho');
    p1.EdgeColor='none';
    colormap(ax1,viridis)
    colorbar
    xlabel('x')
    ylabel('t')
    title('rho(t,x)')
    ax2=subplot(1,2,2);
    p2=pcolor(x,trange,fliplr(u)');
    p2.EdgeColor='none';
    colormap(ax2,copper)
    colorbar
    xlabel('x')
    ylabel('t')
    title('u(t,x)')
    drawnow

  case 'cdfe'
    % Plot density at initial and final times
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b');
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density')
    drawnow

    % Plot density and control
    figure;
    ax1=subplot(1,2,1);
    p1=pcolor(x,trange,rho');
    p1.EdgeColor='none';
    colormap(ax1,viridis)
    colorbar
    xlabel('x')
    ylabel('t')
    title('rho(t,x)')
    ax2=subplot(1,2,2);
    p2=pcolor(x,trange,fliplr(u)');
    p2.EdgeColor='none';
    colormap(ax2,copper)
    colorbar
    xlabel('x')
    ylabel('t')
    title('u(t,x)')
    drawnow

  case 'mtoc'
    % Plot density at initial and final times and target
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b',x,rhobar,'-g');
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density', 'rhobar(x) -- Target density')
    drawnow

    % Plot density and control
    figure;
    ax1=subplot(1,2,1);
    p1=pcolor(x,trange,rho');
    p1.EdgeColor='none';
    colormap(ax1,viridis)
    colorbar
    xlabel('x')
    ylabel('t')
    title('rho(t,x)')
    ax2=subplot(1,2,2);
    p2=pcolor(x,trange,fliplr(u)');
    p2.EdgeColor='none';
    colormap(ax2,copper)
    colorbar
    xlabel('x')
    ylabel('t')
    title('u(t,x)')
    drawnow

  otherwise
    error('Example not known')
end

% Plot functional value decrease
figure;
semilogy(1:niter,J_hist,'x')
hold on
semilogy(0:niter,J_hist(end)*ones(1,niter+1),'-r')
title('Decrease of functional J')
xlabel('l')
ylabel('J(u^l)')
legend('Functional value',sprintf('Reached value: %.4f',J_hist(end)))
drawnow

rmpath(genpath('auxiliary'))
