%s
% # Лабораторная работа №6: Методы симплексного поиска:
% Изучение методов симплексного поиска:
% 1. метод симплексного поиска при помощи регулярного симплекса 
% (метод Спендлея-Хекста-Химсфорта)
% 2. метод Нелдера-Мида (простейшая схема метода симплексного поиска
% при помощи нерегулярного симплекса)
%e
%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
clc % очистка рабочего поля

choose_function = 1; % 1 - квадратичная (0,0),
% 2 - Розенброка (1, 1), 3 - Химмельблаy (3, 2), (-3.77,-3.28),
% (3.58, -1.84) 
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

% %change - замена для других функций

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% точности вычислений
e={'1e-3';'1e-5'}; % '1e-3';'1e-7' %c
e1 = str2double(e(1));
e2 = str2double(e(2));

% параметры методов
kappa0 = 1; % начальный коэффициент сходимости %c
nu = 0.8; % [0.5, 0.8] для метода градиентного спуска с дроблением шага 
omega = 0.5; % вектор спуска, задаваемый антиградиентом
kappa_max = 5.0; % для золотого сечения максимальная граница поиска %c
alpha = 1; % для фунции Розенброка
%global dim; %#ok<GVMIS> 
%dim = 2; % поиск по n координатам

X1 = [-1.0; -2.2]; % 1я точка для исследования
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
        alpha = 1;
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
fprintf('Лабораторная работа №6: Методы симплексного поиска\n'); %с
fprintf('Изучение методов симплексного поиска:\n'); %с
fprintf('1. метод симплексного поиска при помощи регулярного симплекса\n'); %с
fprintf('2. метод Нелдера-Мида (нерегулярный симплекс)\n'); %с

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

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
if choose_function == 2
    alpha = 1; %======================================================= %change
    f_xy = @(x, y) alpha*(x.^2 - y).^2 + (x - 1).^2; %change
    fprintf(strcat('alpha = ', num2str(alpha), '\n')); %change
end
%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

%xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   
fprintf('Целевая функция:   f(x, y) = %s\n\n', f_xy(sym('x'), sym('y')));
fprintf('-----------------------------------------------------------\n\n');



% не двигать окно, иначе графики едут
full_window_size = get(0, 'ScreenSize');
full_window_size(3) = 1920;
full_window_size(4) = 1080;

%%%%%%%%%%%%%%%%%%%%%%%%%% Метод 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
method_num = 1;
kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
fprintf('--------- Метод регулярного симплекса: ------------\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод регулярного симплекса (квадратичная ф.)', 'NumberTitle', 'off'); %c
    case 2
        figure('Position', [7*window_offset 7*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод регулярного симплекса (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод регулярного симплекса (ф. Химмельблау)', 'NumberTitle', 'off'); %c
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
fprintf('- Метод нерегулярного симплекса: -\n\n'); %c

%oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
switch choose_function
    case 1
        figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод нерегулярного симплекса (квадратичная ф.)', ...
            'NumberTitle', 'off'); %c
    case 2
        figure('Position', [6*window_offset 6*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            strcat('Метод нерегулярного симплекса (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
            'NumberTitle', 'off'); %c
    case 3
        figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
            full_window_size(4)-window_offset_size], 'Name', ...
            'Метод нерегулярного симплекса (ф. Химмельблау)', ...
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
    alpha = 2; %c
    kappa_max = 5; %c
    f_xy = @(x, y) alpha*(x.^2 - y).^2 + (x - 1).^2; %change
    fprintf(strcat('alpha = ', num2str(alpha), '\n')); %change

    fprintf('Целевая функция:   f(x, y) = %s\n\n', f_xy(sym('x'), sym('y')));
    fprintf('-----------------------------------------------------------\n\n');
    %%%%%%%%%%%%%%%%%%%%%%%%%% Метод 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    method_num = 1;
    kappa_max = kappa_adjustment(kappa_max, alpha, case_num, method_num);
    fprintf('--------- Метод регулярного симплекса: --------\n\n'); %c

    figure('Position', [3*window_offset 3*window_offset full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод регулярного симплекса (ф. Розенброка, alpha = ', num2str(alpha), ')'), ...
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
    fprintf('- Метод нерегулярного симплекса: -\n\n'); %c
    
    figure('Position', [2*window_offset 2*window_offset full_window_size(3)-window_offset_size ...
        full_window_size(4)-window_offset_size], 'Name', ...
        strcat('Метод нерегулярного симплекса (ф. Розенброка, alpha = ', ...
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
    [X, x_k, y_k,count_f_xy,count_new_dots, ...
        x_k_add, y_k_add] = methods(f_xy, X0, e1, kappa0, ...
        kappa_max, nu, omega, case_num); %с

    time_needed_1 = toc(tStart); % время выполнения

    n_1=count_f_xy;
    k_1=count_new_dots;
    xmin_1=X;
    f_1=f_xy(xmin_1(1),xmin_1(2));
    xmin_1=sprintf('(%.3f, %.3f)', xmin_1(1), xmin_1(2));
    f_1=sprintf('%.3f', f_1);      
    
    nexttile % связана с tiledlayout, переход к след графику на форме    
    draw_surf_coutour_and_steps(f_xy, X0, x_k, y_k,e1, ...
        time_needed_1, case_num, x_k_add, y_k_add); 
    

    [x_min_end, x_max_end, y_min_end, y_max_end] = axis_adjustment(x_k, y_k);   
    axis([x_min_end x_max_end y_min_end y_max_end]);
    
    
    
    % e2 ++++++++++++++++++
    tStart = tic;           % pair 1: tic
    [X, x_k, y_k,count_f_xy,count_new_dots, x_k_add, y_k_add ...
        ] = methods(f_xy, X0, e2, kappa0, ...
        kappa_max, nu, omega, case_num); %с
    
    time_needed_2 = toc(tStart);

    n_2=count_f_xy;
    k_2=count_new_dots;
    xmin_2=X;
    f_2=f_xy(xmin_2(1),xmin_2(2));
    xmin_2=sprintf('(%.5f, %.5f)',xmin_2(1),xmin_2(2));
    f_2=sprintf('%.5f',f_2);
    
    nexttile
    draw_surf_coutour_and_steps(f_xy, X0, x_k, y_k,e2, ...
        time_needed_2, case_num, x_k_add, y_k_add); 

    [x_min_end, x_max_end, y_min_end, y_max_end] = axis_adjustment(x_k, y_k);   
    axis([x_min_end x_max_end y_min_end y_max_end]);
    
    % !! улучшить вывод таблицы, обрезать незначащие нули
    % таблица к точке
    x={num2str(xmin_1,'%.4f'); num2str(xmin_2,'%.4f')};
    f_x={num2str(f_1,'%.4f'); num2str(f_2,'%.4f')};
    count_f_xy=[n_1;n_2];
    count_new_dots=[k_1;k_2];
    point={num2str(x1,'%.4f');num2str(x2,'%.4f')};

    % вывод таблицы
    T=table(e,x,f_x,count_new_dots,count_f_xy, point, ...
        'VariableNames',{'точность', 'конечная точка', 'экстремум', ...
        'кол-во симплексов', 'функция выч.', ...
        'начальная точка'});
    disp(T);

end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

% методы %c
function  [X, x_k, y_k,count_f_xy,count_new_dots, x_k_add, ...
    y_k_add] = methods(f, ...
    X0, eps, kappa0, kappa_max, nu, omega, case_num)
    dim = int16(2); % поиск по n координатам
    kappa_max = 25;
    L = 2.;

    % нерегулярный симплекс
    alpha = 1;
    beta = 2;
    gamma = 0.5;
%     alpha = 2;
%     betta = 2.5;
%     gamma = 0.25;
    % регулярный симплекс
    delta = 0.5; 


    
    count_f_xy=dim; %n
    count_new_dots=0; %k

    syms x y kappa
    Arg = [x; y];
    f = f(x, y);
    
    X = X0;
    kappa_k = kappa0; %#ok<NASGU> 


    %x_k = X(1);
    %y_k = X(2);

    x_k_add = [];
    y_k_add = [];

    x_k = [];
    y_k = [];

    % координаты вершин симплекса
    X_simplex = zeros(dim, dim+1);
    for i = 1:dim+1
        for j = 1:dim
            if (j < i-1)
                X_simplex(j, i) = X(j);
            elseif (j == i-1)
                X_simplex(j, i) = X(j)+L*sqrt(double(j)/((2.*(double(j)+1.))));
            else
                X_simplex(j, i) = X(j)-L/sqrt((2.*double(j)*(double(j)+1.)));
            end
        end
    end
    
    x_k_add = X_simplex(1, :);
    y_k_add = X_simplex(2, :);

    x_k = [x_k X_simplex(1, :)]; %#ok<AGROW> 
    y_k = [y_k X_simplex(2, :)]; %#ok<AGROW> 

    switch case_num
            case 1 % регулярный симплекс 
            Xc = [0; 0];
            for i = 1:dim+1
                Xc = Xc + X_simplex(:, i);
            end
            Xc = Xc/double(dim+1);         
            Fc = double(subs(f, Arg, Xc));

            F_sum = 0;
            for i = 1:dim+1
                F_sum = F_sum + (Fc-double(subs(f, Arg, X_simplex(:, i)))).^2;
            end
            F_sum = F_sum/double(dim+1);         
            
            eps2 = eps*eps;
            while ((L > eps*0.1) || (F_sum > eps2))
        %     while(norm(double(subs(f, Arg, X_simplex(:, 1)))-...
        %             double(subs(f, Arg, X_simplex(:, 2)))) > eps*0.1)
                count_new_dots=count_new_dots+1;
        %         if (count_new_dots >= 4)
        %             break;
        %         end
        
                f_value = zeros(2, dim+1);
                % вычисление функций и сортировка
                for i = 1:dim+1
                    f_value(1, i) = double(subs(f, Arg, X_simplex(:, i)));
                    f_value(2, i) = i;
                end
                count_f_xy=count_f_xy+1;
        
                % сортировка
                for i = 1:dim+1
                    for j = i:dim+1
                        if (f_value(1, i) > f_value(1, j))
                            temp = f_value(1, i);
                            f_value(1, i) = f_value(1, j);
                            f_value(1, j) = temp;
            
                            temp = f_value(2, i);
                            f_value(2, i) = f_value(2, j);
                            f_value(2, j) = temp;
                        end
                    end
                end
               
                % координата максимального значения
                x_coord_max = int32(f_value(2, 3));
                X_sum = [0; 0];
                for i = 1:dim+1
                    if (i ~= x_coord_max)
                        X_sum = X_sum + X_simplex(:, i);
                    end
                end
                X_C = X_sum/double(dim);
                %6.1
                X_new_k = 2*X_C - X_simplex(:, x_coord_max);
                x_coord_min = int32(f_value(2, 1));

                count_f_xy=count_f_xy+1;
                if (double(subs(f, Arg, X_new_k)) > double(subs(f, Arg, X_simplex(:, x_coord_max))))
                    for i = 1:dim+1
                        L = L*delta;
                        if (i ~= x_coord_min)
                            X_simplex(:, i) = X_simplex(:, x_coord_min)+ ...
                                delta*(X_simplex(:, i)-X_simplex(:, x_coord_min));
                        end
                    end
                else
                    X_simplex(:, x_coord_max) = X_new_k;
                end
        
                x_k_add = [x_k_add; X_simplex(1, :)]; %#ok<AGROW> 
                y_k_add = [y_k_add; X_simplex(2, :)]; %#ok<AGROW> 
                x_k = [x_k X_simplex(1, :)]; %#ok<AGROW> 
                y_k = [y_k X_simplex(2, :)]; %#ok<AGROW> 
        
                %x_k = [x_k X_simplex(1, x_coord_min)]; %#ok<AGROW> 
                %y_k = [y_k X_simplex(2, x_coord_min)]; %#ok<AGROW> 
        
                %disp(L);
                %value1 = double(subs(f, Arg, X_simplex(:, x_coord_min)));
                %value1= norm(double(subs(f, Arg, X_simplex(:, 1)))-...
                %    double(subs(f, Arg, X_simplex(:, 2))));
                %disp(value1);
        
                %draw_surf_coutour_and_steps(f, X0, x_k, y_k, eps, 1, ...
                %    case_num, x_k_add, y_k_add)

                Xc = [0; 0];
                for i = 1:dim+1
                    Xc = Xc + X_simplex(:, i);
                end
                Xc = Xc/double(dim+1);         
                Fc = double(subs(f, Arg, Xc));
    
                F_sum = 0;
                for i = 1:dim+1
                    F_sum = F_sum + (Fc-double(subs(f, Arg, X_simplex(:, i)))).^2;
                end
                F_sum = F_sum/double(dim+1); 
            end
            X = X_simplex(:, x_coord_min);
        case 2 % нерегулярный симплекс
            Xc = [0; 0];
            for i = 1:dim+1
                Xc = Xc + X_simplex(:, i);
            end
            Xc = Xc/double(dim+1);         
            Fc = double(subs(f, Arg, Xc));

            F_sum = 0;
            for i = 1:dim+1
                F_sum = F_sum + (Fc-double(subs(f, Arg, X_simplex(:, i)))).^2;
            end
            F_sum = F_sum/double(dim+1);         
            
            eps2 = eps*eps;
            while ((L > eps*0.1) || (F_sum > eps2))
        %     while(norm(double(subs(f, Arg, X_simplex(:, 1)))-...
        %             double(subs(f, Arg, X_simplex(:, 2)))) > eps*0.1)
                count_new_dots=count_new_dots+1;
        %         if (count_new_dots >= 4)
        %             break;
        %         end
        
                f_value = zeros(2, dim+1);
                % вычисление функций и сортировка
                for i = 1:dim+1
                    f_value(1, i) = double(subs(f, Arg, X_simplex(:, i)));
                    f_value(2, i) = i;
                end
                count_f_xy=count_f_xy+1;
        
                % сортировка
                for i = 1:dim+1
                    for j = i:dim+1
                        if (f_value(1, i) > f_value(1, j))
                            temp = f_value(1, i);
                            f_value(1, i) = f_value(1, j);
                            f_value(1, j) = temp;
            
                            temp = f_value(2, i);
                            f_value(2, i) = f_value(2, j);
                            f_value(2, j) = temp;
                        end
                    end
                end
               
                % координата максимального значения
                x_coord_max = int32(f_value(2, 3));
                X_sum = [0; 0];
                for i = 1:dim+1
                    if (i ~= x_coord_max)
                        X_sum = X_sum + X_simplex(:, i);
                    end
                end
                X_C = X_sum/double(dim);
                %6.3 
                X_new_k = X_C + alpha*(X_C - X_simplex(:, x_coord_max));
                x_coord_min = int32(f_value(2, 1));

                % f(x_k+1_n+1) < f(x_k+1_1) 
                if (double(subs(f, Arg, X_new_k)) < double(subs(f, Arg, ...
                        X_simplex(:, x_coord_min))))
                    count_f_xy=count_f_xy+1;

                    X_new_k_star1 = X_C + beta*(X_new_k - X_C);
                    X_wave = X_simplex(:, x_coord_max);
                    X_new_k_star2 = X_C + gamma*(X_wave - X_C);

                    count_f_xy=count_f_xy+2;
                    if (double(subs(f, Arg, X_new_k_star1)) < double(subs(f, Arg, ...
                            X_new_k)))
                        X_simplex(:, x_coord_max) = X_new_k_star1;
                    elseif (double(subs(f, Arg, X_new_k_star2)) < double(subs(f, Arg, ...
                            X_new_k_star1)))
                        X_simplex(:, x_coord_max) = X_new_k_star2;
                    else
                        X_simplex(:, x_coord_max) = X_new_k;
                    end
                else
                    for i = 1:dim+1
                        L = L*delta;
                        if (i ~= x_coord_min)
                            X_simplex(:, i) = X_simplex(:, x_coord_min)+ ...
                                delta*(X_simplex(:, i)-X_simplex(:, x_coord_min));
                        end
                    end
                end

        
                x_k_add = [x_k_add; X_simplex(1, :)]; %#ok<AGROW> 
                y_k_add = [y_k_add; X_simplex(2, :)]; %#ok<AGROW> 
                x_k = [x_k X_simplex(1, :)]; %#ok<AGROW> 
                y_k = [y_k X_simplex(2, :)]; %#ok<AGROW> 
        
                %x_k = [x_k X_simplex(1, x_coord_min)]; %#ok<AGROW> 
                %y_k = [y_k X_simplex(2, x_coord_min)]; %#ok<AGROW> 
        
                %disp(L);
                %value1 = double(subs(f, Arg, X_simplex(:, x_coord_min)));
                %value1= norm(double(subs(f, Arg, X_simplex(:, 1)))-...
                %    double(subs(f, Arg, X_simplex(:, 2))));
                %disp(value1);
        
                %draw_surf_coutour_and_steps(f, X0, x_k, y_k, eps, 1, ...
                %    case_num, x_k_add, y_k_add)

                Xc = [0; 0];
                for i = 1:dim+1
                    Xc = Xc + X_simplex(:, i);
                end
                Xc = Xc/double(dim+1);         
                Fc = double(subs(f, Arg, Xc));
    
                F_sum = 0;
                for i = 1:dim+1
                    F_sum = F_sum + (Fc-double(subs(f, Arg, X_simplex(:, i)))).^2;
                end
                F_sum = F_sum/double(dim+1); 
            end
            
            if (double(subs(f, Arg, X_simplex(:, x_coord_max))) > double(subs(f, Arg, ...
                    X_simplex(:, x_coord_min))))
                X = X_simplex(:, x_coord_min);
            else
                X = X_simplex(:, x_coord_max);
            end
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
function draw_surf_coutour_and_steps(f, X0, x_k, y_k, e, time_needed, ...
    case_num, x_k_add, y_k_add)
    x_max = max(abs(X0(1)), abs(X0(2)));
    border_5 = 4; %c
    [X, Y] = meshgrid(-2*x_max-border_5:0.01:2*x_max+border_5, ...
        -2*x_max-border_5:0.01:2*x_max+border_5);

    Z = f(X, Y);

    v = f(x_k, y_k);
    % !! v насчитать не для всех, выводить через одну от растояния зависит
    contour(X, Y, Z, v, '-', 'Color', 'b'); %, 'ShowText','on');
    hold on

    %plot(x_k, y_k, '-', 'Color', 'green'); % отрисовка промежуточных шагов

    % отрисовка треугольников
    [numRows,~] = size(x_k_add);
    for i=1:numRows
        plot([x_k_add(i, :) x_k_add(i, 1)], [y_k_add(i, :) y_k_add(i, 1)], ...
            '-', 'Color', 'black', 'LineWidth', 0.5); % отрисовка шагов
    end % 'LineWidth', 2

    plot(x_k, y_k, '.', 'Color', 'r', 'MarkerSize', 8);
    plot(X0(1), X0(2), '.', 'Color', 'green', 'MarkerSize', 20);

    grid on;
    xlabel('x');
    ylabel('y');
    title(['точность ', num2str(e), ...
        '; Start point: (',num2str(X0(1)),', ',num2str(X0(2)),'); ', ...
            'вычислено за ', sprintf('%.4f', time_needed), 'с '])

    %hold off

end





