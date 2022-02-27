% Golden-section search var 12
clc;
figure;
a = 0;
b = 1;
e1 = 1e-3;
goldnumber = (1+sqrt(5))./2;
x1 = a + (b - a) / (goldnumber + 1);
x2 = b - (b - a) / (goldnumber + 1);
f1 = f(x1);
f2 = f(x2);
n1 = 0;
s1 = 2;
i = 1;
subplot(1,3,1);
hold on;
grid on;   

while b - a > e1
 plot([a b],[2*i 2*i],'b','LineWidth',1);
 i = i + 1;
if (f1<f2)
    b=x2;
    x2 = x1;
    f2 = f1;
    x1 = a +(b - a)/(goldnumber + 1);
    f1=f(x1);
    s1 = s1 + 1;
    n1 = n1 + 1;
else
a = x1;
x1 = x2;
f1 = f2;
x2 = b - (b - a) / (goldnumber + 1);
f2 = f(x2); 
s1 = s1 + 1;
end
end
x = (a + b) / 2;
legend('Minimum at 1e-3');
min1 = sprintf('%.3f',x);
f1 = f(x);
f1 = sprintf('%.3f',f1);

t= -1:0.00001:0;
% figure
% xlabel('Ox') 
% ylabel('Oy')
% hold on;
% grid on;
% plot(t,f(t));
% plot(x,f(x),'*r');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e2 = 1e-7;
goldnumber = (1+sqrt(5))./2;
x1 = a + (b - a) / (goldnumber + 1);
x2 = b - (b - a) / (goldnumber + 1);
f1 = f(x1);
f2 = f(x2);
n2 = 0;
s2 = 2;
i = 1;
subplot(1,3,2);
hold on;
grid on;
while (b - a > e2)
plot([a b],[2*i 2*i],'b','LineWidth',1);
 i = i + 1;
if (f1 < f2)
    b=x2;
    x2 = x1;
    f2 = f1;
    x1 = a +(b - a)/(goldnumber + 1);
    f1=f(x1);
    s2 = s2 + 1;
    n2 = n2 + 1;
else
a = x1;
x1 = x2;
f1 = f2;
x2 = b - (b - a) / (goldnumber + 1);
f2 = f(x2); 
s2 = s2 + 1;
end
end
x = (a + b) / 2;
legend('Minimum at 1e-7');
min2 = sprintf('%.7f',x);
f2 = f(x);
f2 = sprintf('%.7f',f2);


t= -1:0.00001:0;
% figure
% hold on;
% grid on;
% xlabel('Ox') 
% ylabel('Oy')
% plot(t,f(t));
% plot(x,f(x),'*r');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
e3 = 1e-17;
goldnumber = (1+sqrt(5))./2;
x1 = a + (b - a) / (goldnumber + 1);
x2 = b - (b - a) / (goldnumber + 1);
f1 = f(x1);
f2 = f(x2);
n3 = 0;
s3 = 2;
i = 1;
subplot(1,3,3);
hold on;
grid on;

while (b - a > e3)
plot([a b],[2*i 2*i],'b','LineWidth',1);
 i = i + 1;
if (f1 < f2)
    b=x2;
    x2 = x1;
    f2 = f1;
    x1 = a +(b - a)/(goldnumber + 1);
    f1=f(x1);
    s3 = s3 + 1;
    n3 = n3 + 1;
else
  plot([a b],[2*i 2*i],'b','LineWidth',1);
a = x1;
x1 = x2;
f1 = f2;
x2 = b - (b - a) / (goldnumber + 1);
f2 = f(x2); 
s3 = s3 + 1;
if n3 > 20 %50     
    break;
end
%disp(a);
%disp(b);
end
end
x = (a + b) / 2;
legend('Minimum at 1e-17');
min3 = sprintf('%.17f',x);
min = {min1;min2;min3};
f3 = f(x);
f3 = sprintf('%.17f',f3);
funcvalue = {f1;f2;f3};

%преобразовать в строки
% n1 = sprintf('%.17f',n1);
% n2 = sprintf('%.17f',n2);
% n3 = sprintf('%.17f',n3);
% 
% s1 = sprintf('%.17f',s1);
% s2 = sprintf('%.17f',s2);
% s3 = sprintf('%.17f',s3);
% 
% e1 = sprintf('%.17f',e1);
% e2 = sprintf('%.17f',e2);
% e3 = sprintf('%.17f',e3);
%end


n = {n1;n2;n3};
s = {s1;s2;s3};
e = {e1;e2;e3};
T=table(e, min, funcvalue, n, s);
disp(T);

t= -1:0.00001:0;
% figure
% hold on;
% grid on;
% xlabel('Ox') 
% ylabel('Oy')
% plot(t,f(t));
% plot(x,f(x),'*r');

function func = f(x)
R1 = exp((x.^4 + 2*x.^3 - 5.*x + 6.)/5.);
R2 = cosh(1./(-15.*x.^3 + 10.*x + 5*sqrt(10.)));
func = R1+R2-3.;
end


























