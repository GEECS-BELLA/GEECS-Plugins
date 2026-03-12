%% Intialize %%
clear all; close all;
% Load Simulation DATA (TOPAS Beam Divergence @ each EBT3A Layer)
load('DIV.mat')
% Load Simulation DATA (TOPAS v SRIM as well as Eloss due to scatter TOPAS)
load('SIM.mat')
%% Define Constants %%
c = 299792458; % m/s
me = 0.5109989461; % Mass electron in MeV/c^2
G = 0.5109989461; % me*c^2 in MeV;
K = 0.307075; % 4*pi*Na*re^2*G in Mev*cm^2/mol
zi = 1; % Charge # of incident particle
M = 938.27231; % Incident particle mass (rest mass) MeV/c^2
Eo = 938.27231; % Proton Rest mass (MeV)
%% Define Z, A, X (Density), and L (Length) for stack configuration %%
Zea = 7.46; Aea = 461.381; Xea = 0.90138; Lea = 0.028;
Zes = 6.71; Aes = 144.21; Xes = 1.26909; Les = 0.125;
Zcu = 29; Acu = 63.546; Xcu = 8.96; Lcu = 0.3;
Zal = 13; Aal = 26.981539; Xal = 2.7; Lal = 0.025;
Zni = 28; Ani = 58.6934; Xni = 8.9; Lni = 0.11;
%% Density Effect Correction (Copper, Aluminum, and Nickel) %%
C_cu = 4.4190; C_al = 4.2395; C_ni = 4.3115;
X0_cu = -0.0254; X0_al = 0.1708; X0_ni = -0.0566;
X1_cu = 3.2792; X1_al = 3.0127; X1_ni = 3.1851;
m_cu = 2.9044; m_al = 3.6345; m_ni = 2.8430;
a_cu = 0.14339; a_al = 0.08024; a_ni = 0.16496;
Delta0_cu = 0.08; Delta0_al = 0.12; Delta0_ni = 0.10;
Gc = 2*(log(10));
% Density correction; Need to solve for EBT3 % ???
% Setting EBT3S and EBT3A to Cu for now
C_es = C_cu; C_ea = C_cu;
X0_es = X0_cu; X0_ea = X0_cu;
X1_es = X1_cu; X1_ea = X1_cu;
m_es = m_cu; m_ea = m_cu;
a_es = a_cu; a_ea = a_cu;
Delta0_es = Delta0_cu; Delta0_ea = Delta0_cu;
%% Create Stack %%
Al = Lal; Ni = Lni; Cu = Lcu; EBS = Les; EBA = Lea;
%
Stack = [Al, Ni, EBS, EBA, EBS, Cu, EBS, EBA, EBS, Cu, Cu, EBS, EBA, EBS, Cu, Cu, EBS, EBA, EBS, Cu, Cu,...
    EBS, EBA, EBS, Cu, EBS, EBA, EBS, Cu, EBS, EBA, EBS, Cu, EBS, EBA, EBS, Cu, EBS, EBA, EBS];
%
Lf = sum(Stack,'all');
PL = 0.005; % Propagation Length
StackLen = (0:PL:Lf);
T = length(StackLen); T1 = length(Stack);
% Solve for material cutoff locations
Cut = zeros(1,T1);
for j = 1:T1
    Cut(j) = round(Stack(j)/PL);
end
%% Stack coefficients (X (Density), A, Z, and I)
Xi = [Xal, Xni, Xes, Xea, Xes, Xcu, Xes, Xea, Xes, Xcu, Xcu, Xes, Xea, Xes, Xcu, Xcu, Xes, Xea, Xes, Xcu,...
    Xcu, Xes, Xea, Xes, Xcu, Xes, Xea, Xes, Xcu, Xes, Xea, Xes, Xcu, Xes, Xea, Xes, Xcu, Xes, Xea, Xes];
Aeff = [Aal, Ani, Aes, Aea, Aes, Acu, Aes, Aea, Aes, Acu, Acu, Aes, Aea, Aes, Acu, Acu, Aes, Aea, Aes, Acu,...
    Acu, Aes, Aea, Aes, Acu, Aes, Aea, Aes, Acu, Aes, Aea, Aes, Acu, Aes, Aea, Aes, Acu, Aes, Aea, Aes];
Zeff = [Zal, Zni, Zes, Zea, Zes, Zcu, Zes, Zea, Zes, Zcu, Zcu, Zes, Zea, Zes, Zcu, Zcu, Zes, Zea, Zes, Zcu,...
    Zcu, Zes, Zea, Zes, Zcu, Zes, Zea, Zes, Zcu, Zes, Zea, Zes, Zcu, Zes, Zea, Zes, Zcu, Zes, Zea, Zes];
Ieff = 10.*Zeff; % Bloch Estimation (eV)
% Assign coeffient values along depth of stack using cutoff matrix
u = repelem(Xi,Cut); Den = reshape(u,[],1); Density = Den(1:T);
v = repelem(Aeff,Cut); AeffS = reshape(v,1,[]); AStack = AeffS(1:T);
v1 = repelem(Zeff,Cut); ZeffS = reshape(v1,1,[]); ZStack = ZeffS(1:T);
v2 = repelem(Ieff,Cut); IeffS = reshape(v2,1,[]); IStack = IeffS(1:T);
%% Density correction coefficients (Stack)
Ci = [C_al, C_ni, C_es, C_ea, C_es, C_cu, C_es, C_ea, C_es, C_cu, C_cu, C_es, C_ea, C_es, C_cu,...
    C_cu, C_es, C_ea, C_es, C_cu, C_cu, C_es, C_ea, C_es, C_cu, C_es, C_ea, C_es, C_cu, C_es, C_ea,...
    C_es, C_cu, C_es, C_ea, C_es, C_cu, C_es, C_ea, C_es];
X0i = [X0_al, X0_ni, X0_es, X0_ea, X0_es, X0_cu, X0_es, X0_ea, X0_es, X0_cu, X0_cu, X0_es, X0_ea,...
    X0_es, X0_cu, X0_cu, X0_es, X0_ea, X0_es, X0_cu, X0_cu, X0_es, X0_ea, X0_es, X0_cu, X0_es,...
    X0_ea, X0_es, X0_cu, X0_es, X0_ea, X0_es, X0_cu, X0_es, X0_ea, X0_es, X0_cu, X0_es, X0_ea, X0_es];
X1i = [X1_al, X1_ni, X1_es, X1_ea, X1_es, X1_cu, X1_es, X1_ea, X1_es, X1_cu, X1_cu, X1_es, X1_ea,...
    X1_es, X1_cu, X1_cu, X1_es, X1_ea, X1_es, X1_cu, X1_cu, X1_es, X1_ea, X1_es, X1_cu, X1_es,...
    X1_ea, X1_es, X1_cu, X1_es, X1_ea, X1_es, X1_cu, X1_es, X1_ea, X1_es, X1_cu, X1_es, X1_ea, X1_es];
mi = [m_al, m_ni, m_es, m_ea, m_es, m_cu, m_es, m_ea, m_es, m_cu, m_cu, m_es, m_ea,...
    m_es, m_cu, m_cu, m_es, m_ea, m_es, m_cu, m_cu, m_es, m_ea, m_es, m_cu, m_es,...
    m_ea, m_es, m_cu, m_es, m_ea, m_es, m_cu, m_es, m_ea, m_es, m_cu, m_es, m_ea, m_es];
ai = [a_al, a_ni, a_es, a_ea, a_es, a_cu, a_es, a_ea, a_es, a_cu, a_cu, a_es, a_ea,...
    a_es, a_cu, a_cu, a_es, a_ea, a_es, a_cu, a_cu, a_es, a_ea, a_es, a_cu, a_es,...
    a_ea, a_es, a_cu, a_es, a_ea, a_es, a_cu, a_es, a_ea, a_es, a_cu, a_es, a_ea, a_es];
Delta0i = [Delta0_al, Delta0_ni, Delta0_es, Delta0_ea, Delta0_es, Delta0_cu, Delta0_es, Delta0_ea,...
    Delta0_es, Delta0_cu, Delta0_cu, Delta0_es, Delta0_ea, Delta0_es, Delta0_cu, Delta0_cu, Delta0_es,...
    Delta0_ea, Delta0_es, Delta0_cu, Delta0_cu, Delta0_es, Delta0_ea, Delta0_es, Delta0_cu, Delta0_es,...
    Delta0_ea, Delta0_es, Delta0_cu, Delta0_es, Delta0_ea, Delta0_es, Delta0_cu, Delta0_es, Delta0_ea,...
    Delta0_es, Delta0_cu, Delta0_es, Delta0_ea, Delta0_es];
% Assign coeffient values along depth of stack using cutoff matrix
w = repelem(Ci,Cut); CiS = reshape(w,1,[]); CiStack = CiS(1:T);
w1 = repelem(X0i,Cut); X0iS = reshape(w1,1,[]); X0iStack = X0iS(1:T);
w2 = repelem(X1i,Cut); X1iS = reshape(w2,1,[]); X1iStack = X1iS(1:T);
w3 = repelem(mi,Cut); miS = reshape(w3,1,[]); miStack = miS(1:T);
w4 = repelem(ai,Cut); aiS = reshape(w4,1,[]); aiStack = aiS(1:T);
w5 = repelem(Delta0i,Cut); Delta0iS = reshape(w5,1,[]); Delta0iStack = Delta0iS(1:T);
%% Calculate Bethe Block %%
%Inital Particle Energy
Ei = 50; EStep = Ei;
% Define Material specific lengths along stack
Dal = (0:PL:Lal); Dni = (0:PL:Lni); Des = (PL:PL:Les); Dea = (PL:PL:Lea); Dcu = (PL:PL:Lcu);
Li = [Dal(:); Dni(:); Des(:); Dea(:); Des(:); Dcu(:); Des(:); Dea(:); Des(:);Dcu(:);...
    Dcu(:); Des(:); Dea(:); Des(:); Dcu(:); Dcu(:); Des(:); Dea(:); Des(:); Dcu(:);...
    Dcu(:); Des(:); Dea(:); Des(:); Dcu(:); Des(:); Dea(:); Des(:); Dcu(:); Des(:);...
    Dea(:); Des(:); Dcu(:); Des(:); Dea(:); Des(:); Dcu(:); Des(:); Dea(:); Des(:);0;0;0;0];
% Preallocate matricies for Loop
PDelta = zeros(1,T); PWmax = zeros(1,T); PBetheB = zeros(1,T); LSP = zeros(1,T);
EDep = zeros(1,T); Gamma = zeros(1,T); Beta = zeros(1,T); x = zeros(1,T);
%
% Solve Bethe Bloch Equation for Stack starting @ 50 MeV
for m = 1:T
        Gamma(m) = (EStep(m)+Eo)/Eo;
        Beta(m) = sqrt(1-(1/Gamma(m))^2);
        % Density correction
        x(m) = log(Beta(m)*Gamma(m));
        PDelta(m) = DeltaCalc(CiStack(m), X0iStack(m), X1iStack(m), miStack(m), aiStack(m), Delta0iStack(m), Gc, x(m));
        % Solve Bethe Block
        PWmax(m) = Wmax(Beta(m), Gamma(m), G, me, M);
        PBetheB(m) = BetheB(K, zi, ZStack(m), AStack(m), G, Beta(m), Gamma(m), PWmax(m), IStack(m), PDelta(m));
        % Define Energy Spectrum
        LSP(m) = PBetheB(m).*Density(m).*0.1; % stopping power [Mev/mm]
        EDep(m) = LSP(m).*Li(m);    % [MeV]
        EStep(m+1) = EStep(m) + EDep(m); %%%%%%%%%%%%%%%%%%%%%%%%%
end

%%
figure(101)
subplot(1,3,1)
plot(10*LSP(1:50),'.')
subplot(1,3,2)
plot(10*EDep(1:50),'.')
subplot(1,3,3)
plot(10*EStep(1:50),'.')

%% Small Angle Scatter %%
% Define XO (Radiation length) for each Material
XO_al = 24.01; XO_ni = 12.68; XO_cu = 12.86; XO_es = 41.92; XO_ea = 49.45;% g/cm^2
%XO_es = 42.85; XO_ea = 36.42;% g/cm^2 (For Replacement Material Estimate, Currently Analytical Estimate)
% Radiation Length (Stack)
XOi = [XO_al, XO_ni, XO_es, XO_ea, XO_es, XO_cu, XO_es, XO_ea, XO_es, XO_cu, XO_cu, XO_es, XO_ea,...
    XO_es, XO_cu, XO_cu, XO_es, XO_ea, XO_es, XO_cu, XO_cu, XO_es, XO_ea, XO_es, XO_cu, XO_es,...
    XO_ea, XO_es, XO_cu, XO_es, XO_ea, XO_es, XO_cu, XO_es, XO_ea, XO_es, XO_cu, XO_es, XO_ea, XO_es];
% Assign XO values along depth of stack using cutoff matrix
u1 = repelem(XOi,Cut); RadL = reshape(u1,1,[]);
RadLen = RadL(1:T);
% Determine Relative density along depth of stack
Yal = zeros(1,T); Yni = zeros(1,T); Yes = zeros(1,T); Yea = zeros(1,T); Ycu = zeros(1,T);
% Find Total Ammount of each Material along depth of stack
for f = 1:T
    Yal(f) = sum(Density(f) == Xal);
    Yni(f) = sum(Density(f) == Xni);
    Yes(f) = sum(Density(f) == Xes);
    Yea(f) = sum(Density(f) == Xea);
    Ycu(f) = sum(Density(f) == Xcu);
end
b = Yal; b1 = Yni; b2 = Yes;  b3 = Yea; b4 = Ycu;
for i=2:T
    b(i) = b(i-1) + Yal(i);
    b1(i) = b1(i-1) + Yni(i);
    b2(i) = b2(i-1) + Yes(i);
    b3(i) = b3(i-1) + Yea(i);
    b4(i) = b4(i-1) + Ycu(i);
end
RelDen_al = ((b*PL)*Xal)./(StackLen+0.000001);
RelDen_ni = ((b1*PL)*Xni)./(StackLen+0.000001);
RelDen_es = ((b2*PL)*Xes)./(StackLen+0.000001);
RelDen_ea = ((b3*PL)*Xea)./(StackLen+0.000001);
RelDen_cu = ((b4*PL)*Xcu)./(StackLen+0.000001);
RelDen = RelDen_al + RelDen_ni + RelDen_es + RelDen_ea + RelDen_cu;
%Solve for relative XO value along depth of stack
LHS = StackLen.*RelDen;
RHS_al = ((b*PL)*Xal)./XO_al;
RHS_ni = ((b1*PL)*Xni)./XO_ni;
RHS_es = ((b2*PL)*Xes)./XO_es;
RHS_ea = ((b3*PL)*Xea)./XO_ea;
RHS_cu = ((b4*PL)*Xcu)./XO_cu;
RHS = RHS_al + RHS_ni + RHS_es + RHS_ea + RHS_cu;
XO = LHS./RHS;
% Solve small angle scatter estimation
E1 = EStep(2:T+1) + Eo; % Total Energy (U + KE)
z1 = rand(100,T,'single'); % independent Gaussian random variables (z1, z2) with mean zero and variance one
z2 = rand(100,T,'single'); % independent Gaussian random variables (z1, z2) with mean zero and variance one
p = sqrt(E1.^2 - (Eo/c^2)^2)/c; % Momentum
Xz = (LHS*0.1); % Mass per unit area g/cm^2
%
Theta0 = Scatter(XO,Xz,zi,Beta,p,c);
% Scale to Beam diamter 10 mm (Topas) and solve for % increase
% MAX value
Y_Plane = YPlane(1,1,Xz*10,Theta0);
YPlot = Y_Plane+10;
G2 = (YPlot-10)*100/10;
% MC
Y_Plane1 = YPlane(z1,z2,Xz*10,Theta0);
YPlot1 = Y_Plane1+10;
G3 = (YPlot1-10)*100/10;
%% Plot %%
% Bethe Bolch Model %
Acc = Beta.*Gamma; % 0.1<=Beta*Gamma<=1000 for Bethe Bloch model accuracy
for n = 1:T
    if Acc(n) <= 1 && Acc(n) >= 1000
        disp('Warning: The Model is not Accurate')
    end
end
%
DresPlot(StackLen,-LSP,Xdist,ESpec);
%DresPlot(StackLen,-LSP,Xu,ESpecT);
%
% Small Angle Scatter Model %
DivPlot(StackLen,G3*2,StackLen,G2*2,XT,YT);
%% Functions %%
% Bethe Bloch Model Functions %
function BBWmax = Wmax(Beta, Gamma, G, me, M)
    BBWmax = (2*G.*Beta.^2.*Gamma.^2)/(1+((2*Gamma*me)/M)+(me/M)^2);
end

function BetheBlochCalc = BetheB(K, zi, Zeff, Aeff, G, Beta, Gamma, PWmax, Ieff, PDelta)
    % First Expression
    A1 = K*(zi^2)*(Zeff/Aeff);
    A2 = 1/(Beta^2);
    A3 = A1*A2;
    % Second Expression
    A4 = (2*G*Beta^2*Gamma^2*PWmax)/(Ieff^2);
    A5 = Beta^2 - (PDelta/2);
    % Bethe Block
    BetheBlochCalc = A3*((0.5*log(A4)) - A5);
end

function Delta = DeltaCalc(C, X0, X1, m, a, Delta0, Gc, x)
    if x>=X1
        Delta = Gc*x-C;
    elseif X0<=x & x<X1
        Delta = Gc*x-C + a*(X1 - x)^m;
    elseif x<X0
        Delta = Delta0*10.^(2*(x-X0));
    end
end

% Small Angle Scatter Functions %
function SAScatter = Scatter(X0,Xi,z,Beta,p,c)
    SAScatter = ((13.6*z)./(Beta.*c.*p)).*(sqrt(Xi./X0)).*(1+0.038*log((Xi.*z^2)/(X0.*Beta.^2)));
end

function Yp = YPlane(z1,z2,Xi,T0)
    Yp = z1.*Xi.*T0/sqrt(12) + z2.*(Xi.*T0)/2;
end

% Plot Funciton %
function plotDres = DresPlot(D1, Y1, D2, Y2)
figure1 = figure;
set(figure1, 'Position', [100 300 1500 500]);
set(figure1, 'visible', 'on');
axes1 = axes('Parent', figure1);
hold(axes1,'on');
G = 50;
ylim([0 G])
xlim([0 6])
% Aluminum
rectangle('Parent',axes1,'Position',[0 0 .025 G],'FaceColor',[0.6 0.6 0.5]);
% Nickel
rectangle('Parent',axes1,'Position',[0.025 0 .11 G],'FaceColor',[0.7 0.5 0]);
% RCF (Active)
rectangle('Parent',axes1,'Position',[0.26 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[0.838 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[1.716 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[2.594 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[3.472 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[4.05 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[4.628 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[5.206 0 0.028 G],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[5.784 0 0.028 G],'FaceColor',[1 0 0]);
% Copper
rectangle('Parent',axes1,'Position',[0.413 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[0.991 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[1.291 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[1.869 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[2.169 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[2.747 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[3.047 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[3.625 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[4.203 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[4.781 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[5.359 0 0.3 G],'FaceColor',[0.5 0.8 0.5]);
plot(D1, Y1, 'b','LineWidth',1)
hold on
plot(D2, Y2, 'k-.','LineWidth',1)
hold off
xlabel('Depth [mm]')
ylabel('Dose [MeV/mm]')
title('Analytical Model (Bethe-Bloch)')
legend('ML: Linear Stopping Power (MeV/mm)','SRIM: Energy (MeV/mm)', 'Location', 'Southoutside','NumColumns',1);
end

function plotDiv = DivPlot(D1, Y1, D2, Y2, D3, Y3)
figure1 = figure;
set(figure1, 'Position', [100 300 1500 500]);
set(figure1, 'visible', 'on');
axes1 = axes('Parent', figure1);
hold(axes1,'on');
G1 = 25;
ylim([0 G1])
xlim([0 6])
% Aluminum
rectangle('Parent',axes1,'Position',[0 0 .025 G1],'FaceColor',[0.6 0.6 0.5]);
% Nickel
rectangle('Parent',axes1,'Position',[0.025 0 .11 G1],'FaceColor',[0.7 0.5 0]);
% RCF (Active)
rectangle('Parent',axes1,'Position',[0.26 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[0.838 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[1.716 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[2.594 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[3.472 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[4.05 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[4.628 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[5.206 0 0.028 G1],'FaceColor',[1 0 0]);
rectangle('Parent',axes1,'Position',[5.784 0 0.028 G1],'FaceColor',[1 0 0]);
% Copper
rectangle('Parent',axes1,'Position',[0.413 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[0.991 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[1.291 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[1.869 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[2.169 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[2.747 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[3.047 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[3.625 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[4.203 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[4.781 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
rectangle('Parent',axes1,'Position',[5.359 0 0.3 G1],'FaceColor',[0.5 0.8 0.5]);
plot(D1, Y1, '-.','Color',[0,0.7,0.9],'LineWidth',1)
hold on
plot(D2, Y2, 'k--','LineWidth',1.5)
plot(D3, Y3, 'k-o','LineWidth',1)
hold off
xlabel('Depth [mm]')
ylabel('Beam Divergence (% increase)')
title('Analytical Model (Small Angle Scatter)')
legend('ML: Small Angle Scatter Est.','ML: Small Angle Scatter Est (Max)','TOPAS: Small Angle Scatter at EBT3 Active', 'Location', 'Southoutside','NumColumns',1);
end
