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
% [ACCC24] G. Albi, M. Caliari, F. Cassini, E. Calzola, and F. Cassini.
%          Exponential integrators for mean-field selective optimal control
%          problems. arXiv preprint arXiv:2302.00127, 2024.

addpath(genpath('aux'))

ex = 'opsz'; % String to choose the example. See above for the admitted values.

switch ex
  case 'opsz'
    disp('--- Control in opinion dynamics: Sznajd model ---')

    o.n = 1000;
    tolsw = 2e-3;
    o.ts = 200;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 0;
    o.sigma = sqrt(2*1e-2);
    o.gamma = 0.5;
    T = 8;
    xd = -0.5;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    o.s = @(rho) ones(o.n,1);
    o.dsdrho = @(rho) zeros(o.n,1);
    betasz = -1;
    Pfun = @(x,y) betasz*(1-x.^2);
    rhoplus = @(x,a,b) max(-(x/b).^2+a,0);
    rho0fun = @(x) rhoplus(x+0.75,0.05,0.5) + rhoplus(x-0.5,0.15,1);
    psi0 = zeros(o.n,1);

    deltav = 0.1;
    deltamin = 0.2;
    deltamax = 2;

  case 'ophk'
    disp('--- Control in opinion dynamics: Hegselmann--Krause model ---')

    o.n = 1000;
    tolsw = 2e-3;
    o.ts = 100;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 0;
    o.sigma = sqrt(2*1e-3);
    o.gamma = 2.5;
    T = 10;
    xd = 0;
    Efun = @(x,rho) (x-xd).^2.*rho;
    dEdrhofun = @(x,rho) (x-xd).^2;
    o.s = @(rho) ones(o.n,1);
    o.dsdrho = @(rho) zeros(o.n,1);
    kappa = 0.15;
    Pfun = @(x,y) (y<=(x+kappa)).*(y>=(x-kappa));
    epsilon = 0.01;
    rho0fun = @(x) 0.5+epsilon*(1-x.^2);
    psi0 = zeros(o.n,1);

    deltav = 0.01;
    deltamin = 0.01;
    deltamax = 0.03;

  case 'cdfe'
    disp('--- Crowd dynamics: fast exit of two groups ---')

    o.n = 1000;
    tolsw = 2e-3;
    o.ts = 250;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 10;
    o.sigma = sqrt(2*2e-2);
    o.gamma = 1;
    T = 3;
    Efun = @(x,rho) rho;
    dEdrhofun = @(x,rho) ones(o.n,1);
    o.s = @(rho) 1-rho;
    o.dsdrho = @(rho) -ones(o.n,1);
    Pfun = @(x,y) 0;
    rho0fun = @(x) 0.9*exp(-100*(x+0.4).^2)+0.65*exp(-150*x.^2);
    psi0 = zeros(o.n,1);

    deltav = 0.01;
    deltamin = 0.01;
    deltamax = 0.2;

  case 'mtoc'
    disp('--- Mass transfer via optimal control ---')

    o.n = 1000;
    tolsw = 2e-3;
    o.ts = 200;

    % Model parameters
    xl = -1;
    xr = 1;
    o.beta = 0;
    o.sigma = sqrt(2*1e-2);
    o.gamma = 0.1;
    T = 3;
    mu1 = 0.5;
    sigma1 = 0.1;
    mu2 = -0.3;
    sigma2 = 0.15;
    rhobarfun = @(x) exp(-(x-mu1).^2/(2*sigma1^2)) + exp(-(x-mu2).^2/(2*sigma2^2));
    Efun = @(rho,rhobar) abs(rho-rhobar).^2;
    dEdrhofun = @(rho,rhobar) 2*(rho-rhobar);
    o.s = @(rho) ones(o.n,1);
    o.dsdrho = @(rho) zeros(o.n,1);
    betasz = -0.05;
    Pfun = @(x,y) betasz*(1-x.^2);
    mu0 = 0;
    sigma0 = 0.1;
    rho0fun = @(x) exp(-(x-mu0).^2/(2*sigma0^2));
    psi0 = zeros(o.n,1);

    deltav = 0.03;
    deltamin = 0.015;
    deltamax = 0.015;

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

o.tau = T/o.ts;
rho0 = rho0fun(x);
switch ex
  case {'opsz','ophk'}
    norm_rho0 = trapz(x,rho0);
    rho0 = rho0/norm_rho0;
    E = @(rho) Efun(x,rho);
    o.dEdrho = @(rho) dEdrhofun(x,rho);
  case 'cdfe'
    E = @(rho) Efun(x,rho);
    o.dEdrho = @(rho) dEdrhofun(x,rho);
  case 'mtoc'
    norm_rho0 = trapz(x,rho0);
    rho0 = rho0/norm_rho0;
    rhobar = rhobarfun(x);
    norm_rhobar = trapz(x,rhobar);
    rhobar = rhobar/norm_rhobar;
    E = @(rho) Efun(rho,rhobar);
    o.dEdrho = @(rho) dEdrhofun(rho,rhobar);
  otherwise
    error('Example not known')
end

rho = repmat(rho0,1,o.ts+1);
psi = repmat(psi0,1,o.ts+1);
u = -(o.s(fliplr(rho))/o.gamma).*(o.D1t*psi);

trange = linspace(0,T,o.ts+1);
Jtmp = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho));

niter = 0;

disp('Doing integration (displaying functional values)...')
switch ex
  case {'opsz','ophk'}
    tic
    niter = niter + 1;
    rho = forward_exp_sone(rho0,u,o);
    psi = backward_exp_sone(psi0,rho,u,o);
    dJdun = o.gamma*u+o.D1t*psi;
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))
    J_hist(niter) = Jfun;

    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;
      rho = forward_exp_sone(rho0,u,o);
      psi = backward_exp_sone(psi0,rho,u,o);
      dJdunm1 = dJdun;
      dJdun = o.gamma*u+o.D1t*psi;
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;
      Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))
      errsw = abs(Jfun-Jtmp);
      J_hist(niter) = Jfun;
      Jtmp = Jfun;
    end
    cpu_time=toc;
  case 'cdfe'
    tic
    niter = niter + 1;
    rho = forward_exp(rho0,u,o);
    psi = backward_exp(psi0,rho,u,o);
    dJdun = o.gamma*u+o.s(fliplr(rho)).*(o.D1t*psi);
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))
    J_hist(niter) = Jfun;

    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;
      rho = forward_exp(rho0,u,o);
      psi = backward_exp(psi0,rho,u,o);
      dJdunm1 = dJdun;
      dJdun = o.gamma*u+o.s(fliplr(rho)).*(o.D1t*psi);
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;
      Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))
      errsw = abs(Jfun-Jtmp);
      J_hist(niter) = Jfun;
      Jtmp = Jfun;
    end
    cpu_time=toc;
  case 'mtoc'
    tic
    niter = niter + 1;
    rho = forward_exp_sone(rho0,u,o);
    psi0 = rho(:,o.ts+1)-rhobar;
    psi = backward_exp_sone(psi0,rho,u,o);
    dJdun = o.gamma*u+o.D1t*psi;
    unm1 = u;
    u = u-deltav*dJdun;

    Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))+...
           1/2*trapz(x,abs(rho(:,o.ts+1)-rhobar).^2)
    J_hist(niter) = Jfun;

    errsw = 2*tolsw;
    while errsw > tolsw
      niter = niter + 1;
      rho = forward_exp_sone(rho0,u,o);
      psi0 = rho(:,o.ts+1)-rhobar;
      psi = backward_exp_sone(psi0,rho,u,o);
      dJdunm1 = dJdun;
      dJdun = o.gamma*u+o.D1t*psi;
      deltav = abs(trapz(trange,fliplr(trapz(x,(u-unm1).*(dJdun-dJdunm1)))))/trapz(trange,fliplr(trapz(x,(dJdun-dJdunm1).^2)));
      deltav = max(deltamin,min(deltav,deltamax));
      unm1 = u;
      u = u-deltav*dJdun;
      Jfun = 1/2*trapz(trange,trapz(x,E(rho)+o.gamma*abs(fliplr(u)).^2.*rho))+...
             1/2*trapz(x,abs(rho(:,o.ts+1)-rhobar).^2)
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

viridis = color_viridis();

switch ex
  case {'opsz','ophk'}
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b',[xd,xd],[0 max(rho(:,end))],'-g');
    ylim([0 max(rho(:,end))])
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density','x_d - Target')
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
  case 'cdfe'
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b');
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density')
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
  case 'mtoc'
    figure;
    plot(x,rho(:,1),'*r',x,rho(:,end),'+b',x,rhobar,'-g');
    xlabel('x')
    ylabel('rho(t,x)')
    legend('rho_0(x) - Initial density','rho(T,x) - Final density', 'rhobar(x) -- Target density')
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
  otherwise
    error('Example not known')
end

figure;
semilogy(1:niter,J_hist,'x')
hold on
semilogy(0:niter,J_hist(end)*ones(1,niter+1),'-r')
title('Decrease of functional J')
xlabel('l')
ylabel('J(u^l)')
legend('Functional value',sprintf('Reached value: %.4f',J_hist(end)))

rmpath(genpath('aux'))
