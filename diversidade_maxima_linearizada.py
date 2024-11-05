import os
import numpy as np
from pyomo.environ import *
import time  

def read_instance(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    N = int(lines[0].strip().split()[0])  
    M = int(lines[0].strip().split()[1])  
    
    d_ij = np.zeros((N, N))
    
    for line in lines[1:]:
        i, j, dist = line.strip().split()
        i, j, dist = int(i), int(j), float(dist)
        d_ij[i, j] = dist
        d_ij[j, i] = dist  
    
    return N, M, d_ij


folder_path = 'instancias'  
if not os.path.exists(folder_path):
    raise FileNotFoundError(f"O caminho '{folder_path}' não existe.")

files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

custom_results = []

for file in files:
    file_path = os.path.join(folder_path, file)
    N, M, d_ij = read_instance(file_path)
    
    model = ConcreteModel()

    nodes = range(1, N + 1)
    Q = [(i, j) for i in nodes for j in nodes if i < j]

    # Definindo variáveis
    model.x = Var(nodes, within=Binary) 
    model.y = Var(Q, within=Binary) 

    model.obj = Objective(expr=sum(d_ij[i-1][j-1] * model.y[i, j] for (i, j) in Q), sense=maximize)

    model.sum_x = Constraint(expr=sum(model.x[i] for i in nodes) == M)

    def y_constraint_1(model, i, j):
        return model.x[i] + model.x[j] - model.y[i, j] <= 1
    model.y_constraint_1 = Constraint(Q, rule=y_constraint_1)

    def y_constraint_2(model, i, j):
        return -model.x[i] + model.y[i, j] <= 0
    model.y_constraint_2 = Constraint(Q, rule=y_constraint_2)

    def y_constraint_3(model, i, j):
        return -model.x[j] + model.y[i, j] <= 0
    model.y_constraint_3 = Constraint(Q, rule=y_constraint_3)

    start_time = time.time()

    solver = SolverFactory('glpk')
    solver_results = solver.solve(model)

    execution_time = time.time() - start_time

    if solver_results.solver.termination_condition == TerminationCondition.optimal:
        optimal_value = model.obj()
        custom_results.append((file, optimal_value, execution_time))
    else:
        custom_results.append((file, None, execution_time))

for file, optimal_value, execution_time in custom_results:
    if optimal_value is not None:
        print(f"Resultados para o arquivo: {file}")
        print(f"Valor ótimo: {optimal_value:.2f}")
        print(f"Tempo de execução: {execution_time:.2f} segundos")
    else:
        print(f"Resultados para o arquivo: {file}")
        print("Solução não encontrada ou modelo inviável.")
        print(f"Tempo de execução: {execution_time:.2f} segundos")
