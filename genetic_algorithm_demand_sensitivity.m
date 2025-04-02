function genetic_algorithm_demand_sensitivity()

    % Initialization Parameters
    num_generations = 100;
    population_size = 50;
    crossover_rate = 0.8;
    mutation_rate = 0.1;

    % Cost Parameters
    C1i = [4967, 4967]; % Setup cost (Rp/batch)
    C2i = [256, 323]; % Inventory cost per item i (Rp/ton)
    C3i = [73667, 85945]; % Production cost per item i (Rp/ton)
    C4i = [6250, 6250] * 12; % Solid waste disposal cost per year (Rp/year)
    C5i = [4500 * 60, 0]; % Liquid waste sales revenue per year (Rp/year)

    % Production per year
    Pi = [4228, 3624]; % Production of item i (kg/year)

    % Warehouse Capacity per month
    current_warehouse_volume = 3 * 3 * 2.7; % m^3
    warehouse_capacity = current_warehouse_volume * 2; % m^3

    % Product space requirement
    vi = [0.02, 0.02]; % Assumed space requirement per product (m^3/kg)

    % Define demand variations for sensitivity analysis
    demand_variations = [3000:500:6000]; % Example range of demands

    % Preallocate storage for best fitness values
    best_fitness_values = zeros(length(demand_variations), 1);

    % Loop over each demand variation
    for d = 1:length(demand_variations)
        Di = [demand_variations(d), demand_variations(d)]; % Current demand for analysis

        % Population Initialization
        population = initialize_population(population_size, numel(Di) * 2); % T_i and T_2i

        best_fitness = inf;

        for generation = 1:num_generations
            % Objective Function Evaluation
            fitness = evaluate_population(population, Di, Pi, C1i, C2i, C3i, C4i, C5i, warehouse_capacity, vi);

            % Selection
            selected_population = selection(population, fitness);

            % Crossover
            new_population = crossover(selected_population, crossover_rate);

            % Mutation
            mutated_population = mutation(new_population, mutation_rate);

            % Re-evaluation
            fitness = evaluate_population(mutated_population, Di, Pi, C1i, C2i, C3i, C4i, C5i, warehouse_capacity, vi);

            % Update Population
            population = mutated_population;

            % Store Best Fitness
            current_best_fitness = min(fitness);
            if current_best_fitness < best_fitness
                best_fitness = current_best_fitness;
            end
        end

        % Store best fitness for current demand variation
        best_fitness_values(d) = best_fitness;

        fprintf('Demand: %d, Best Fitness = %f\n', demand_variations(d), best_fitness);
    end

    % Plotting the results
    figure;
    plot(demand_variations, best_fitness_values, '-o');
    xlabel('Demand (kg/year)');
    ylabel('Best Fitness Value');
    title('Sensitivity Analysis: Best Fitness Value vs Demand');
    grid on;

end

% Helper functions

function population = initialize_population(population_size, num_variables)
    % Initialize random population
    population = rand(population_size, num_variables);
end

function fitness = evaluate_population(population, Di, Pi, C1i, C2i, C3i, C4i, C5i, warehouse_capacity, vi)
    % Evaluate the objective function for each individual in the population
    num_individuals = size(population, 1);
    fitness = zeros(num_individuals, 1);

    for i = 1:num_individuals
        T = population(i, 1:end/2);
        T2 = population(i, end/2+1:end);
        fitness(i) = calculate_objective_function(T, T2, Di, Pi, C1i, C2i, C3i, C4i, C5i, warehouse_capacity, vi);
    end
end

function objective_value = calculate_objective_function(T, T2, Di, Pi, C1i, C2i, C3i, C4i, C5i, warehouse_capacity, vi)
    % Calculate the objective function value based on T and T2
    n = numel(Di);
    objective_value = 0;

    for i = 1:n
        term1 = C1i(i) / T(i);
        term2 = (C2i(i) + Di(i) * T2(i)^2) / (2 * T(i));
        term3 = (C3i(i) * Di(i) + C4i(i) - C5i(i));
        term4 = (C4i(i) - vi(i) * Pi(i) * (T2(i) * Di(i)) / (Pi(i) - Di(i) + eps));

        objective_value = objective_value + term1 + term2 + term3 + term4;
    end

    % Ensure warehouse capacity is not exceeded
    if sum(T .* vi .* Di) > warehouse_capacity
        objective_value = inf; % Infinite penalty if capacity is exceeded; 
    end
end

function selected_population = selection(population, fitness)
    % Select individuals based on their fitness using roulette wheel selection
    num_individuals = size(population, 1);
    selected_population = zeros(size(population));

    % Avoid division by zero if all fitness are zero
    total_fitness = sum(fitness);
    if total_fitness == 0
        normalized_fitness = ones(num_individuals, 1) / num_individuals; % Equal probability if all fitness are zero
    else
        normalized_fitness = fitness / total_fitness;
    end

    cumulative_fitness = cumsum(normalized_fitness);

    for i = 1:num_individuals
        r = rand;
        selected_index = find(cumulative_fitness >= r, 1, 'first');
        if ~isempty(selected_index)
            selected_population(i, :) = population(selected_index, :);
        else
            % Debugging output
            disp('Warning: No selection made, choosing random individual instead.');
            selected_population(i, :) = population(randi(num_individuals), :);
        end
    end
end

function new_population = crossover(population, crossover_rate)
    % Perform crossover to generate new individuals
    num_individuals = size(population, 1);
    new_population = zeros(size(population));

    for i = 1:2:num_individuals
        if rand < crossover_rate
            parent1 = population(i, :);
            parent2 = population(i+1, :);

            % Single-point crossover
            crossover_point = randi([1, numel(parent1)-1]);
            new_population(i, :) = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
            new_population(i+1, :) = [parent2(1:crossover_point), parent1(crossover_point+1:end)];
        else
            new_population(i, :) = population(i, :);
            new_population(i+1, :) = population(i+1, :);
        end
    end
end

function mutated_population = mutation(population, mutation_rate)
    % Perform mutation on the population
    num_individuals = size(population, 1);
    num_variables = size(population, 2);
    mutated_population = population;

    for i = 1:num_individuals
        for j = 1:num_variables
            if rand < mutation_rate
                % Mutation by adding a random value
                mutated_population(i, j) = population(i, j) + randn * 0.1;
            end
        end
    end
end
