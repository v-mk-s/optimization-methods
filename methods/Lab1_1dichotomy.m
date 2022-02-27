% dichotomy var 12
clc;
figure;
a = 0;
b = 1;
x = linspace(a,b,1000);
e1 = 1e-3;
dx1 = e1 / 5;
n1 = 0;
s1 = 0;
i = 1;
subplot(3,2,1)
hold on;
grid on;
while b-a > e1
plot([a b],[2*i 2*i],'b','LineWidth',1);
t = (b + a) / 2;
 x1 = t - dx1;
 x2 = t + dx1;
 i = i + 1;
 if f(x1) < f(x2)
     b = x2;
 else
     a = x1;
 end
 s1=s1+2;
 n1=n1+1;
 end
legend('Minimum at 1e-3');
min1 = sprintf('%.3f',t);
f1 = f(t);
f1 = sprintf('%.3f',f1);

subplot(3,2,2)
xlabel('Ox') 
ylabel('Oy')
grid on;
hold on;
y=f(x);
plot(x,y);
plot (t,f(t),'og');
plot([t t],[f(t) f(t)],'b','LineWidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = 0;
b = 1;
x = linspace(a,b,1000);
e2 = 1e-7;
dx1 = e2 / 5;
n2 = 0;
s2 = 0;
i = 1;
subplot(3,2,3)
hold on;
grid on;
while b-a > e2
plot([a b],[2*i 2*i],'b','LineWidth',1);
t = (b + a) / 2;
 x1 = t - dx1;
 x2 = t + dx1;
 i = i + 1;
 if f(x1) < f(x2)
     b = x2;
 else
     a = x1;
 end
 s2=s2+2;
 n2=n2+1;
 end
legend('Minimum at 1e-7');
min2 = sprintf('%.7f',t);
f2 = f(t);
f2 = sprintf('%.7f',f2);

subplot(3,2,4)
xlabel('Ox') 
ylabel('Oy')
grid on;
hold on;
y=f(x);
plot(x,y);
plot (t,f(t),'og');
plot([t t],[f(t) f(t)],'b','LineWidth',1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a = 0;
b = 1;
x = linspace(a,b,1000);
e3 = 1e-17;
dx1 = e3 / 5;
n3 = 0;
s3 = 0;
i = 1;
subplot(3,2,5)
hold on;
grid on;
while b-a > e3
plot([a b],[2*i 2*i],'b','LineWidth',1);
t = (b + a) / 2;
 x1 = t - dx1;
 x2 = t + dx1;
 i = i + 1;
 if f(x1) < f(x2)
     b = x2;
 else
     a = x1;
 end
 s3=s3+2;
 n3=n3+1;
 end
legend('Minimum at 1e-17');
min3 = sprintf('%.17f',t);
f3 = f(t);
f3 = sprintf('%.17f',f3);

subplot(3,2,6)
xlabel('Ox') 
ylabel('Oy')
grid on;
hold on;
y=f(x);
plot(x,y);
plot (t,f(t),'og');
plot([t t],[f(t) f(t)],'b','LineWidth',1);
f3 = f(t);
f3 = sprintf('%.17f',f3);



% table
n = {n1;n2;n3};
s = {s1;s2;s3};
e = {e1;e2;e3};
min = {min1;min2;min3};
funcvalue = {f1;f2;f3};
T=table(e, min, funcvalue, n, s);
disp(T);


function func = f(x)
R1 = exp((x.^4 + 2*x.^3 - 5.*x + 6.)/5.);
R2 = cosh(1./(-15.*x.^3 + 10.*x + 5*sqrt(10.)));
func = R1+R2-3.;
end














% eps =1e-2; % eps 
% a=0; % [a,b]
% b=1;
% delta=0.01; % выбираем сами
% count_iter=0;
% count_func = 0;
% 
% x = linspace(a,b); % разбитие отрезка на 
% grid on;
% hold on;
% y=f(x);
% plot(x,y);
% i=1;
% aa= [0 0];
% % plot(x,y,'LineWidth',4);
% r=(b-a)*delta;
% 
% while b-a >eps
%  count_iter=count_iter+1;
% 
%  x1 = (b+a)/2-r;
%  x2 = (b+a)/2+r;
%  aa(i,1) = a;
%  aa(i,2) = b;
%  i = i +1;
%  if f(x1)>f(x2)
%      a=x1;
%  else
%      b=x2;
%  end
%  count_func=count_func+2;
% end
% t=(b+a)/2;
% disp('Экстремум') % Maxima and minima
% disp('Extrema') % Maxima and minima
% disp(t)
% disp(f(t))
% disp('Количество итераций:');
% disp('Count of iterations:');
% disp(count_iter);
% disp('Количество вычисленных значений функции:');
% disp('Number of function values evaluated:');
% disp(count_func);
% y=f(t);
% 
% 
% [n,m] =size(aa);
% plot([t t],[f(t) f(t)+1e-2],'b','LineWidth',4);
% figure
% 
% hold on;
% for i=1:n
% plot([aa(i,1) aa(i,2)],[2*i 2*i],'b','LineWidth',4);
% end
% for i=1:n
% disp('Первая граница (First board):');
% disp(aa(i,1));
% disp('Вторая граница (Second board:');
% disp(aa(i,2));
% end
% 
% function VarF = f(x)
% R1 = exp((x.^4 + 2*x.^3 - 5.*x + 6.)/5.);
% R2 = cosh(1./(-15.*x.^3 + 10.*x + 5*sqrt(10.)));
% VarF = R1+R2-3.;
% 
% %R1 := Exp((Degree(x,4)+2*Degree(x,3)-5*x+6)/5);
% %R2 := Coh(1/(-15*Degree(x,3)+10*x+5*Sqrt(10)));
% %VarF := R1+R2-3;
% end
















