

syms a b;
c = 2*a^2+b;
disp(c);

h = c;
h = subs(h, b, 5);
h = subs(h, a, 3);
disp(h);

d= diff(c, a);
disp(d);

test = subs(d, a, 3);
disp(test);

x = sym('x', [2 2]);
disp(x);

y = x* x.'+3;
disp(y);

z = subs(y, x(1,1), 4);
disp(z);

xt = [[2,1];[3,3]];
z = double(subs(y, x, xt));
disp(z);
