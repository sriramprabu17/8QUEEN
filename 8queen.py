
import random
import numpy as np
from numpy.random import choice
import pandas as pd

class MyGa():
    
    boardSize=8
    PopSize = 10
     
    
    def smartLocate(self,*args):
        self.Pop = [1,2,3,4,5,6,7,8]
        for x in range(0,self.boardSize-1):
            #print ("x",x)
            rn =np.random.randint(self.boardSize - x)
            #print ("rn",rn)
            r= x-(rn)
            #print ("r",r)
            a = self.Pop[r]
            self.Pop[r] =self.Pop[x]
            self.Pop[x] = a
        return self.Pop
        #print(self.Pop)
    
    def Pop(self,*args):
        self.initPop =  np.zeros(shape=(self.PopSize,self.boardSize))
        #print(self.initPop)
        for x in range(0,self.PopSize):
            self.initPop[x, :] = self.smartLocate()
        #print(self.initPop[0][1])
        self.dim2to3()
        return self.initPop
    
    def dim2to3(self):

        self.population = np.zeros(shape=(self.PopSize, 8, 8))

        x = 0
        for pop in self.initPop:
            pop = np.uint8(pop)
            i = 0
            for j in pop:
                self.population[x, i, j-1] = 1
                i = i + 1
            x = x + 1
      #  print(self.population)
        
    def fit(self,population):
        self.Pop2D =  np.zeros(shape=(self.PopSize,self.boardSize,self.boardSize))
        self.fitScore = 0 
        self.fitPop =  np.zeros(shape=(self.PopSize,1))
        total_num_attacks = self.queensdiagonal(self.population)
        population_fitness = np.copy(
            total_num_attacks)  # Direct assignment makes both variables refer to the same array. Use numpy.copy() for creating a new independent copy.

        for solution_idx in range(population.shape[0]):
            if population_fitness[solution_idx] == 0:
                population_fitness[solution_idx] = float("inf")
            else:
                population_fitness[solution_idx] = 1.0 / population_fitness[solution_idx]
        #print(population_fitness)
        a = self.rank_selection(population_fitness,2)    
        print (a)
        return population_fitness, total_num_attacks     
        


    def queensdiagonal(self, population):
         # For a given queen, how many queens sharing the same column? This is how the fitness value is calculated.
    
         total_num_attacks = np.zeros(population.shape[0])  # Number of attacks for all solutions (diagonal only).
    
         for solution_idx in range(population.shape[0]):
             ga_solution = population[solution_idx, :]
             
             #print("Population",solution_idx,ga_solution)
             # Padding zeros around the solution board for being able to index the boundaries (leftmost/rightmost columns & top/bottom rows). # This is by adding 2 rows (1 at the top and another at the bottom) and adding 2 columns (1 left and another right).
             temp = np.zeros(shape=(10, 10))
             # Copying the solution board inside the badded array.
             temp[1:9, 1:9] = ga_solution
    
             # Returning the indices (rows and columns) of the 8 queeens.
             row_indices, col_indices = np.where(ga_solution == 1)
             # Adding 1 to the indices because the badded array is 1 row/column far from the original array.
             row_indices = row_indices + 1
             col_indices = col_indices + 1
             #print("a",row_indices,col_indices)
    
             total = 0  # total number of attacking pairs diagonally for each solution.
    
             for element_idx in range(8):
                 x = row_indices[element_idx]
                 y = col_indices[element_idx]
    
                 mat_bottom_right = temp[x:, y:]
                 total = total + self.diagonal_attacks(mat_bottom_right)
                 #print("xy",solution_idx,temp[x:, y:],x,y)
    
                 mat_bottom_left = temp[x:, y:0:-1]
                 #print("xy-1",solution_idx,temp[x:, y:0:-1],x,y)
                 total = total + self.diagonal_attacks(mat_bottom_left)
    
                 mat_top_right = temp[x:0:-1, y:]
                 #print("x-1y",solution_idx,temp[x:0:-1, y:],x,y)
                 total = total + self.diagonal_attacks(mat_top_right)
    
                 mat_top_left = temp[x:0:-1, y:0:-1]
                 #print("x-1y-1",solution_idx,temp[x:0:-1, y:0:-1],x,y)
                 total = total + self.diagonal_attacks(mat_top_left)
    
             # Dividing the total by 2 because it counts the solution as attacking itself diagonally.
             total_num_attacks[solution_idx] = total_num_attacks[solution_idx] + total / 2
    
         return total_num_attacks
    
    def diagonal_attacks(self, mat):
         if (mat.shape[0] < 2 or mat.shape[1] < 2):
             # print("LESS than 2x2.")
             return 0
         num_attacks = mat.diagonal().sum() - 1
         return num_attacks
     
    def rank_selection(self, fitness, num_parents):

        """
        Selects the parents using the rank selection technique. Later, these parents will mate to produce the offspring.
        It accepts 2 parameters:
            -fitness: The fitness values of the solutions in the current population.
            -num_parents: The number of parents to be selected.
        It returns an array of the selected parents.
        """
        

        fitness_sorted = sorted(range(len(fitness)), key=lambda k: fitness[k])
         
        fitness_sorted.reverse()
        
        # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
        parents =  np.zeros(shape=(num_parents,self.boardSize,self.boardSize))
        
        for parent_num in range(num_parents):
            
            parents[parent_num,:] = self.population[fitness_sorted[parent_num],:]
        print(parents.shape)
        return parents
    
    def single_point_crossover(self, parents, offspring_size):

        """
        Applies the single-point crossover. It selects a point randomly at which crossover takes place between the pairs of parents.
        It accepts 2 parameters:
            -parents: The parents to mate for producing the offspring.
            -offspring_size: The size of the offspring to produce.
        It returns an array the produced offspring.
        """

        offspring = numpy.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = numpy.random.randint(low=0, high=parents.shape[1], size=1)[0]

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k % parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1) % parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def swap_mutation(self, offspring):

        """
        Applies the swap mutation which interchanges the values of 2 randomly selected genes.
        It accepts a single parameter:
            -offspring: The offspring to mutate.
        It returns an array of the mutated offspring.
        """

        for idx in range(offspring.shape[0]):
            mutation_gene1 = numpy.random.randint(low=0, high=offspring.shape[1]/2, size=1)[0]
            mutation_gene2 = mutation_gene1 + int(offspring.shape[1]/2)

            temp = offspring[idx, mutation_gene1]
            offspring[idx, mutation_gene1] = offspring[idx, mutation_gene2]
            offspring[idx, mutation_gene2] = temp
        return offspring
    yy
    def best_solution(self):

        """
        Returns information about the best solution found by the genetic algorithm. Can only be called after completing at least 1 generation.
        If no generation is completed (at least 1), an exception is raised. Otherwise, the following is returned:
            -best_solution: Best solution in the current population.
            -best_solution_fitness: Fitness value of the best solution.
            -best_match_idx: Index of the best solution in the current population.
        """
        
        if self.generations_completed < 1:
            raise RuntimeError("The best_solution() method can only be called after completing at least 1 generation but {generations_completed} is completed.".format(generations_completed=self.generations_completed))

#        if self.run_completed == False:
#            raise ValueError("Warning calling the best_solution() method: \nThe run() method is not yet called and thus the GA did not evolve the solutions. Thus, the best solution is retireved from the initial random population without being evolved.\n")

        # Getting the best solution after finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = self.cal_pop_fitness()
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0][0]

        best_solution = self.population[best_match_idx, :]
        best_solution_fitness = fitness[best_match_idx]

        return best_solution, best_solution_fitness, best_match_idx
MyGa1= MyGa()
a= MyGa1.Pop()
b = MyGa1.fit(a)
#print(b)
#MyGa1.rank_selection(b,2)


