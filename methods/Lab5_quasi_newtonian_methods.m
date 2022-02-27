%s
% # Лабораторная работа №5: Квазиньютоновские методы
% Изучение квазиньютоновских методов:
% 1. метод ДФП (Давидона-Флетчера-Пауэлла)
% 2. метод БПГШ (Бройдена-Пауэлла-Гольдфарба-Шенно), в некоторых источниках
% именуемый методом БФШ (Бройдена-Флетчера-Шенно)
% 3. метод Пауэлла
% 4. метод Мак-Кормика
%e
%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
clc % очистка рабочего поля

choose_function = 3; % 1 - квадратичная (0,0),
% 2 - Розенброка (1, 1), 3 - Химмельблаy (3, 2), (-3.77,-3.28),
% (3.58, -1.84) 
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% %change - замена для других функций

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% точности вычислений
e={'1e-2';'1e-3'}; % '1e-3';'1e-7' %c
e1 = str2double(e(1));
e2 = str2double(e(2));

% параметры методов
kappa0 = 1; % начальный коэффициент сходимости %c
nu = 0.8; % [0.5, 0.8] для метода градиентного спуска с дроблением шага 
omega = 0.5; % вектор спуска, задаваемый антиградиентом
kappa_max = 5.0; % для золотого сечения максимальная граница поиска %c
alpha = 1; % для фунции Розенброка

X1 = [-1.0; -2.0]; % 1я точка для исследования
X2 = [0.5; 0.7]; % 2я точка для исследования
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

count_f_xy=0; % количество вычисленых значений функции f_xy
count_new_dots=0; % количество вычисленных новых точек (x, y)
count_grad_f_xy=0; % количество вычисленных градиентов

window_offset = 20; % левый нижний угол
window_offset_size = 300; % размер окна

% preallocated
% norm_w = zeros(1,26,'double');

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
% выбор функции и eps
% целевая функция через символьные переменные
switch choose_function
    case 1
        % квадратичная функция
        f_xy=@(x,y) (x.^2./2)+(y.^2.); %change
        %f_xy=@(x,y) (x.^2.)+(y.^2.); % 1 шаг %change
        %f_xy=@(x,y) (x.^2.-y).^2+(x-1).^2; % Аттетков функция %change

        %e={'1e-3';'1e-7'}; % '1e-3';'1e-8' %change
        e1 = str2double(e(1));
        e2 = str2double(e(2));

        kappa0 = 1; % начальный коэффициент сходимости
        nu = 0.8; % [0.5, 0.8] для метода градиентного спуска с дроблением шага 
        omega = 0.5; % вектор спуска, задаваемый антиградиентом
        kappa_max = 5; % для золотого сечения максимальное kappa

        %X1 = [-1.0; -2.0]; % 1я точка для исследования
        %X2 = [0.5; 0.7]; % 2я точка для исследования
    case 2
        alpha = 2;
        % функция Розенброка %change
        f_xy = @(x, y) alpha*(x.^2 - y).^2 + (x - 1).^2;%change
        %f_xy=@(x,y) (x.^2.-y).^2+(x-1).^2; % Аттетков функция

        %e={'1e-3';'1e-7'}; % '1e-3';'1e-8' %change
        e1 = str2double(e(1));
        e2 = str2double(e(2));

        kappa0 = 1; % начальный коэффициент сходимости
        nu = 0.8; % [0.5, 0.8] для метода градиентного спуска с дроблением шага 
        omega = 0.5; % вектор спуска, задаваемый антиградиентом
        kappa_max = 2.5; % для золотого сечения максимальная граница

        %X1 = [-1.0; -2.0]; % 1я точка для исследования
        %X2 = [0.5; 0.7]; % 2я точка для исследования
    case 3
        % функция Химмельблау %change
        f_xy=@(x,y) (x.^2+y-11).^2 + (x+y.^2-7).^2; %change

        %e={'1e-3';'1e-7'}; % '1e-3';'1e-8' %change
        e1 = str2double(e(1));
        e2 = str2double(e(2));

        kappa0 = 1; % начальный коэффициент сходимости
        nu = 0.8; % [0.5, 0.8] для метода градиентного спуска с дроблением шага 
        omega = 0.5; % вектор спуска, задаваемый антиградиентом
        kappa_max = 5; % для золотого сечения максимальное kappa

        %X1 = [-1.0; -2.0]; % 1я точка для исследования
        %X2 = [0.5; 0.7]; % 2я точка для исследования
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
case_num = 1;
method_num = 1;

% main ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
fprintf('Лабораторная работа №5: Квазиньютоновские методы\n'); %с
fprintf('Изучение методов второго порядка:\n'); %с
fprintf('1. метод ДФП (Давидона-Флетчера-Пауэлла)\n'); %с
fprintf('2. метод БПГШ (Бройдена-Пауэлла-Гольдфарба-Шенно)\n'); %с
fprintf('3. метод Пауэлла\n'); %с
fprintf('4. метод Мак-Кормика\n'); %с

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        fprintf('-------------------- Квадратичная функция -----------------\n');
    case 2
        fprintf('-------------------- функция Розенброка -------------------\n');
    case 3
        fprintf('-------------------- Функция Химмельблау ------------------\n');
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
fprintf('Целевая функция:   f(x, y) = %s\n\n', f_xy(sym('x'), sym('y')));
fprintf('-----------------------------------------------------------\n\n');

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
if choose_function == 2
    alpha = 1; %======================================================= %change
    f_xy = @(x, y) alpha*(x.^2 - y).^2 + (x - 1).^2; %change
    fprintf(strcat('alpha = ', num2str(alpha), '\n')); %change
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% не двигать окно, иначе графики едут
full_window_size = get(0, 'ScreenSize');
full_window_size(3) = 1920;
full_window_size(4) = 1080;

%%%%%%%%%%%%%%%%%%%%%%%%%% Метод 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
method_num = 1;
kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
fprintf('--------- Метод ДФП (Давидона-Флетчера-Пауэлла): --------------\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод ДФП (квадратичная функция)', 'NumberTitle', 'off'); %c
    case 2
        figure('Position', [7*window_offset 7*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод ДФП (функция Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод ДФП (функция Химмельблау)', 'NumberTitle', 'off'); %c
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

tiledlayout(2,2); % creates a tiled chart layout for displaying multiple plots in the current figure.   

% 1я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c
         
%  2я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c
      

%%%%%%%%%%%%%%%%% Метод 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
method_num = 2;
kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
fprintf('- Метод БПГШ (Бройдена-Пауэлла-Гольдфарба-Шенно): -\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод БПГШ (квадратичная ф.)', ...
            'NumberTitle', 'off'); %c
    case 2
        figure('Position', [6*window_offset 6*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод БПГШ (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод БПГШ (на пр. спуска с дробл. шага) (ф. Химмельблау)', ...
            'NumberTitle', 'off'); %c
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

tiledlayout(2,2);

% 1я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c
         
%  2я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c

%%%%%%%%%%%%%%%%% Метод 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
method_num = 3;
kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
fprintf('- Метод Пауэлла: -\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [window_offset window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод Пауэлла (квадратичная ф.)', ...
            'NumberTitle', 'off'); %c
    case 2
        figure('Position', [5*window_offset 5*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод Пауэлла (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [window_offset window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод Пауэлла (на пр. спуска с дробл. шага) (ф. Химмельблау)', ...
            'NumberTitle', 'off'); %c
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

tiledlayout(2,2);

% 1я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c
         
%  2я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c

%%%%%%%%%%%%%%%%% Метод 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
method_num = 4;
kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
fprintf('----------------- Метод Мак-Кормика: -------------------\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [0 0 full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод Мак-Кормика (квадратичная ф.)', ...
            'NumberTitle', 'off'); %c
    case 2
        figure('Position', [4*window_offset 4*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод Мак-Кормика (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [0 0 full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод Мак-Кормика (на пр. спуска с дробл. шага) (ф. Химмельблау)', ...
            'NumberTitle', 'off'); %c
    otherwise
        disp('ERROR!')
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

tiledlayout(2,2);

% 1я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c
         
%  2я точка +++++++++++++++++++++++++
method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, method_num); %c


% только для Розенброка
%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
if choose_function == 2
    fprintf('-----------------------------------------------------------\n\n');
    fprintf('-------------------- функция Розенброка -------------------\n');


    %======================================================= %c
    alpha = 0.5; %c
    kappa_max = 5; %c
    f_xy = @(x, y) alpha*(x.^2 - y).^2 + (x - 1).^2; %change
    fprintf(strcat('alpha = ', num2str(alpha), '\n')); %change

    fprintf('Целевая функция:   f(x, y) = %s\n\n', f_xy(sym('x'), sym('y')));
    fprintf('-----------------------------------------------------------\n\n');
    %%%%%%%%%%%%%%%%%%%%%%%%%% Метод 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    method_num = 1;
    kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
    fprintf('--------- Метод ДФП (Давидона-Флетчера-Пауэлла): --------------\n\n'); %c

    figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод ДФП (функция Розенброка, alpha = ', num2str(alpha), ')'), ...
        'NumberTitle', 'off'); %c
    
    tiledlayout(2,2); % creates a tiled chart layout for displaying multiple plots in the current figure.   
    
    % 1я точка +++++++++++++++++++++++++
    % 1я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
             
    %  2я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
          
    
    %%%%%%%%%%%%%%%%% Метод 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    method_num = 2;
    kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
    fprintf('- Метод БПГШ (Бройдена-Пауэлла-Гольдфарба-Шенно): -\n\n'); %c
    
    figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод БПГШ (ф. Розенброка, alpha = ', ...
        num2str(alpha), ')'), 'NumberTitle', 'off'); %c
    
    tiledlayout(2,2);
    
    % 1я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
             
    %  2я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c

    %%%%%%%%%%%%%%%%% Метод 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    method_num = 3;
    kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
    fprintf('----------------- Метод Пауэлла: -----------------\n\n'); %c
    
    figure('Position', [window_offset window_offset full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод Пауэлла (ф. Розенброка, alpha = ', ...
        num2str(alpha), ')'), 'NumberTitle', 'off'); %c
    
    tiledlayout(2,2);
    
    % 1я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
             
    %  2я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c

    %%%%%%%%%%%%%%%%% Метод 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    method_num = 4;
    kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
    fprintf('------------------ Метод Мак-Кормика: -------------\n\n'); %c
    
    figure('Position', [0 0 full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод Мак-Кормика (ф. Розенброка, alpha = ', ...
        num2str(alpha), ')'), 'NumberTitle', 'off'); %c
    
    tiledlayout(2,2);
    
    % 1я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X1, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
             
    %  2я точка +++++++++++++++++++++++++
    method_evaluate_print(f_xy, X2, e, e1, e2, kappa0, kappa_max, ...
        nu, omega, method_num); %c
end



%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% функции отрисовки и методы ++++++++++++++++++++++++++++++++++++++++++++++ 

% вычисление методa %c
function method_evaluate_print(f_xy, X0, e, e1, e2, kappa0, kappa_max, ...
    nu, omega, case_num)
    % X0 - начальная точка
    x1=sprintf('(%.3f, %.3f)', X0(1), X0(2)); % начальная точка (x, y) с точность е1
    x2=sprintf('(%.8f, %.8f)', X0(1), X0(2)); % начальная точка (x, y) с точность е2
    % !! fprintf('Начальное приближение:  X0 = (%1.0f, %1.0f)^T\n', X0(1), X0(2)'); 

    % e1 ++++++++++++++++++
    tStart = tic;           % pair 1: tic
    [X, norm_w, x_k, y_k,count_f_xy,count_new_dots, ...
        count_grad_f_xy] = methods(f_xy, X0, e1, kappa0, ...
        kappa_max, nu, omega, case_num); %с

    time_needed_1 = toc(tStart); % время выполнения

    n_1=count_f_xy;
    k_1=count_new_dots;
    ng_1=count_grad_f_xy;
    xmin_1=X;
    f_1=f_xy(xmin_1(1),xmin_1(2));
    xmin_1=sprintf('(%.3f, %.3f)', xmin_1(1), xmin_1(2));
    f_1=sprintf('%.3f', f_1);      
    
    nexttile % связана с tiledlayout, переход к след графику на форме    
    draw_surf_coutour_and_steps(f_xy, X0, x_k, y_k, norm_w,e1, time_needed_1); 
    

    [x_min_end, x_max_end, y_min_end, y_max_end] = axis_adjustment(x_k, y_k);   
    axis([x_min_end x_max_end y_min_end y_max_end]);
    
    
    
    % e2 ++++++++++++++++++
    tStart = tic;           % pair 1: tic
    [X, norm_w, x_k, y_k,count_f_xy,count_new_dots, ...
        count_grad_f_xy] = methods(f_xy, X0, e2, kappa0, ...
        kappa_max, nu, omega, case_num); %с
    
    time_needed_2 = toc(tStart);

    n_2=count_f_xy;
    k_2=count_new_dots;
    ng_2=count_grad_f_xy;
    xmin_2=X;
    f_2=f_xy(xmin_2(1),xmin_2(2));
    xmin_2=sprintf('(%.5f, %.5f)',xmin_2(1),xmin_2(2));
    f_2=sprintf('%.5f',f_2);
    
    nexttile
    draw_surf_coutour_and_steps(f_xy, X0, x_k, y_k, norm_w,e2, time_needed_2); 

    [x_min_end, x_max_end, y_min_end, y_max_end] = axis_adjustment(x_k, y_k);   
    axis([x_min_end x_max_end y_min_end y_max_end]);
    
    % !! улучшить вывод таблицы, обрезать незначащие нули
    % таблица к точке
    x={num2str(xmin_1,'%.4f'); num2str(xmin_2,'%.4f')};
    f_x={num2str(f_1,'%.4f'); num2str(f_2,'%.4f')};
    count_f_xy=[n_1;n_2];
    count_new_dots=[k_1;k_2];
    count_grad_f_xy=[ng_1;ng_2];
    point={num2str(x1,'%.4f');num2str(x2,'%.4f')};

    % вывод таблицы
    T=table(e,x,f_x,count_new_dots,count_f_xy,count_grad_f_xy, point, ...
        'VariableNames',{'точность', 'конечная точка', 'экстремум', ...
        'кол-во новых точек', 'функция выч.', ...
        'градиент выч.', 'начальная точка'});
    disp(T);

end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% методы %c
function  [X, norm_w, x_k, y_k,count_f_xy,count_new_dots, ...
        count_grad_f_xy] = methods(f, X0, eps, kappa0, ...
        kappa_max, nu, omega, case_num)
    kappa_max = 25;
    
    count_f_xy=0; %n
    count_new_dots=0; %k
    count_grad_f_xy=0; %ng

    syms x y kappa
    Arg = [x; y];
    f = f(x, y);
    antigrad_f = -gradient(f);  % антиградиент

    % Hessian matrix
    %H = [diff(f, x, {2}),     diff(diff(f, x), y);
    %     diff(diff(f, y), x),     diff(f, y, {2})];
    
    X = X0;
    kappa_k = kappa0; %#ok<NASGU> 
    A = eye(2,2); % I_2
    %f_k = double(subs(f, Arg, X));
    %count_f_xy=count_f_xy+1;

    W = double(subs(antigrad_f, Arg, X));
    count_grad_f_xy=count_grad_f_xy+1;
    norm_w = norm(W);
    x_k = X(1);
    y_k = X(2);

    X_prev = X;
    W_prev = W;
    while norm(W) > eps  
        count_new_dots=count_new_dots+1;
        %disp(norm(W));

        p = A*W;
        norm_p = norm(p);
        p = p/norm_p;

        phi = subs(f, Arg, X + kappa.*p);
        phi = @(kappa_k)double(subs(phi, kappa, kappa_k));

        % !! выбирать a,b аккуратно для каждой функции %change
        [kappa_k,n1] = method_golden_section_search(phi, 0, kappa_max, eps*1e-3); 
        count_f_xy=count_f_xy+n1;

        X = X + kappa_k.*p;
        W = double(subs(antigrad_f, Arg, X));
        count_grad_f_xy=count_grad_f_xy+1;

        norm_w = [norm_w norm(W)]; %#ok<AGROW> 
        x_k = [x_k X(1)]; %#ok<AGROW> 
        y_k = [y_k X(2)];   %#ok<AGROW> 
        %disp(X);
        %disp(norm(W));
        %pause(0.2);
        dX = X - X_prev; 
        dW = W - W_prev; 
     
        switch case_num
            case 1 % delta_A по методу ДФП
                dA = -( dX*(dX.')/dot(dW, dX) + ...
                    A*dW*(dW.')*(A.')/dot(A*dW, dW));
            case 2 % delta_A по методу БПГШ
                r = A*dW/dot(A*dW, dW) - dX/dot(dW, dX);     
                dA = -(dX*(dX.')/dot(dW, dX) + ...
                    A*dW*(dW.')*(A.')/dot(A*dW, dW)) + dot(A*dW, dW)*r*(r.');  
            case 3 % delta_A по методу Пауэлла
                dX_wave = dX+A*dW;
                % 0.1 чтобы убрать вырожденную матрицу
                dA = -(dX_wave*(dX_wave.'))/(dot(dW, dX_wave+1e-7));
                %disp(X);
                %disp(norm(W));
            case 4 % delta_A по методу Мак-Кормика %!!bug
                dA = -(dX*(dX.')/dot(dW, dX) + ...
                    A*dW*(dX.')/dot((dX.'), dW));  
                %disp(norm(W));
        end 

        A = A + dA;

        X_prev = X;
        W_prev = W;
    end
end

% выбор гиперпараметров
function kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_number)
    switch method_number
        case 1 % Метод Ньютона
            switch case_num % подбор параметров
                case 1
                    switch alpha
                        case 1
                            kappa_max = 0.1;
                        case 10
                            kappa_max = 1;
                        case 100
                            kappa_max = 1;
                        case 1000
                            kappa_max = 1;
                    end
                case 2
                    kappa_max = 2;
            end
        case 2 % Модификация: Дробление
            switch case_num
                case 1
                    switch alpha
                        case 1
                            kappa_max = 0.1;
                        case 10
                            kappa_max = 1;
                        case 100
                            kappa_max = 1;
                        case 1000
                            kappa_max = 1;
                    end
                case 2
                    kappa_max = 1;
            end
    end
end

% метод золотого сечения
function [x,n1] = method_golden_section_search(f, a, b, eps)
    n1=2;
        
    goldnumber = (1+sqrt(5))./2;
    a_k=a; % левая граница
    b_k=b; % правая граница
    len_k = b_k - a_k; 

    x_k_left = a_k + len_k/ (goldnumber+1);
    x_k_right = b_k - len_k/ (goldnumber+1);
    f_left = f(x_k_left);
    f_right = f(x_k_right);

    while len_k > eps
        if f_left > f_right
            a_k = x_k_left;
            x_k_left = x_k_right;
            f_left = f_right;
            len_k = b_k - a_k;

            x_k_right = b_k - (b_k - a_k)/(goldnumber+1);
            f_right = f(x_k_right);
        else
            b_k = x_k_right;
            x_k_right = x_k_left;
            f_right = f_left;
            len_k = b_k - a_k;

            x_k_left = a_k + (b_k - a_k)/(goldnumber+1);
            f_left = f(x_k_left);
        end

        n1=n1+1;
    end
    x = (a_k + b_k) / 2;
end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% подгон осей автоматический
function [x_min_end, x_max_end, y_min_end, y_max_end] = axis_adjustment(x_k, y_k)
    % params
    border_multiplier = 0.08;
    border_addition = 0.08;
    
    min_x_k = min(x_k);
    max_x_k = max(x_k);
    len_x_k = max_x_k-min_x_k;
    border_x = border_multiplier*len_x_k+border_addition;

    min_y_k = min(y_k);
    max_y_k = max(y_k);
    len_y_k = max_y_k-min_y_k;
    border_y = border_multiplier*len_y_k+border_addition;

    % прямоугольник, который нужно вывести
    x_min_end = min_x_k-border_x;
    x_max_end = max_x_k+border_x;
    y_min_end = min_y_k-border_y;
    y_max_end = max_y_k+border_y;

    curr_width = x_max_end-x_min_end;
    curr_hight = y_max_end-y_min_end;

    % размеры окна вывода графика
    plot_width = 608;
    plot_hight = 262;

    % прямоугольник подгоняем по форме
    magnification_ratio = 0; %#ok<NASGU> 
    if (plot_width/plot_hight > curr_width/curr_hight)
        magnification_ratio = plot_hight/curr_hight; % коэффициент растяжения
        curr_width_new = plot_width/magnification_ratio;
        %magnification_ratio = curr_width_new/curr_width;

        addition = (curr_width_new-curr_width)/2;

        x_min_end = x_min_end-addition;
        x_max_end = x_max_end+addition;
    else
        magnification_ratio = plot_width/curr_width; % коэффициент растяжения
        curr_hight_new = plot_hight/magnification_ratio;
        %magnification_ratio = curr_hight_new/curr_hight;

        addition = (curr_hight_new-curr_hight)/2;

        y_min_end = y_min_end-addition;
        y_max_end = y_max_end+addition;
    end
    
    %plot

end


% функция отрисовки графика
function draw_surf_coutour_and_steps(f, X0, x_k, y_k, ~, e, time_needed)
    x_max = max(abs(X0(1)), abs(X0(2)));
    border_5 = 4; %c
    [X, Y] = meshgrid(-2*x_max-border_5:0.01:2*x_max+border_5, ...
        -2*x_max-border_5:0.01:2*x_max+border_5);

    Z = f(X, Y);

    v = f(x_k, y_k);
    % !! v насчитать не для всех, выводить через одну от растояния зависит
    contour(X, Y, Z, v, '-', 'Color', 'b'); %, 'ShowText','on');
    hold on

    plot(x_k, y_k, '-', 'Color', 'green');
    plot(x_k, y_k, '.', 'Color', 'r', 'MarkerSize', 8);
    grid on;
    xlabel('x');
    ylabel('y');
    title(['точность ', num2str(e), ...
        '; Start point: (',num2str(X0(1)),', ',num2str(X0(2)),'); ', ...
            'вычислено за ', sprintf('%.4f', time_needed), 'с '])

    %hold off

end





