import numpy as np
import random
from deap import base, creator, tools, algorithms

# Dados: posições e níveis de urgência (perigo)
np.random.seed(43)
n_victims = 10
positions = np.random.rand(n_victims, 2) * 100  # Coordenadas (x, y)
danger_levels = np.random.randint(1, 6, n_victims)  # Condições de perigo (1 a 5)


# Função de avaliação (fitness)
# Considera que resgata todas as vitimas de uma vez
# se for ter que alterar por perigo de vitima, altera-se a distancia de cada nó.
# a distancia será a diferença euclidiana da origem até a vitima.(no final a distancia seria a mesma)
def evaluate(individuo):
    total_distance = 0
    urgency_score = 0
    for i in range(len(individuo) - 1):
        v1, v2 = individuo[i], individuo[i + 1]
        total_distance += np.linalg.norm(
            positions[v1] - positions[v2]
        )  # Distância euclidiana
        urgency_score += danger_levels[v1] / (i + 1)  # Penalizar atrasos no atendimento

    # Adicionar a distância de retorno ao ponto inicial. não tem, queremos apenas a sequencia de resgate.
    # total_distance += np.linalg.norm(positions[individuo[-1]] - positions[individuo[0]])

    # Adicionar a última vítima à avaliação de urgência
    # urgency_score += danger_levels[individuo[-1]] / len(individuo)

    return total_distance, urgency_score


# Configuração do DEAP
creator.create("FitnessMin", base.Fitness, weights=(-0.7, 0.3))  # Minimização
creator.create("EstrIndividuos", list, fitness=creator.FitnessMin)

# Armazena as premissas do problema
toolbox = base.Toolbox()
toolbox.register("Genes", np.random.permutation, n_victims)  # Geração inicial
toolbox.register("individuos", tools.initIterate, creator.EstrIndividuos, toolbox.Genes)

toolbox.register("populacao", tools.initRepeat, list, toolbox.individuos)
# povo = toolbox.populacao(n = 50)

# Operadores genéticos
toolbox.register("mate", tools.cxPartialyMatched)  # Cruzamento para permutações
toolbox.register(
    "mutate", tools.mutShuffleIndexes, indpb=0.1
)  # Mutação por troca -> 10%
toolbox.register("select", tools.selTournament, tournsize=2)  # Torneio para seleção
toolbox.register("evaluate", evaluate)


# Algoritmo Genético com DEAP
def genetic_algorithm(n_generations=100, population_size=50, cxpb=0.7, mutpb=0.2):
    population = toolbox.populacao(n=population_size)
    hall_of_fame = tools.HallOfFame(1)  # Armazena o melhor indivíduo

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=cxpb,
        mutpb=mutpb,
        ngen=n_generations,
        stats=stats,
        halloffame=hall_of_fame,
        verbose=True,
    )

    return hall_of_fame[0], hall_of_fame[0].fitness.values[0]


# Executar o algoritmo genético
best_route, best_score = genetic_algorithm()

# Mostrar resultados
print("\nMelhor ordem de visitação:", best_route)
print("Custo total (fitness):", best_score)

# Visualizar posições e rota
import matplotlib.pyplot as plt

positions = np.array(positions)
best_route_positions = positions[best_route]

plt.figure(figsize=(10, 6))
plt.scatter(positions[:, 0], positions[:, 1], c="red", label="Vítimas")
plt.plot(
    np.append(best_route_positions[:, 0], best_route_positions[0, 0]),
    np.append(best_route_positions[:, 1], best_route_positions[0, 1]),
    c="blue",
    label="Melhor Rota",
)
for i, (x, y) in enumerate(positions):
    plt.text(x, y, f"{i}", fontsize=12, ha="center", va="center")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Melhor Rota para Resgate")
plt.show()
