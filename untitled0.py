import kivy.app
import kivy.uix.gridlayout
import kivy.uix.boxlayout
import kivy.uix.button
import kivy.uix.textinput
import kivy.uix.label
import numpy
from kivy.config import Config

class BuzzleApp(kivy.app.App):
    pop_created = 0 # 0 means a population is not yet created.

    def initialize_population(self, *args):
        self.num_solutions = numpy.uint8(self.num_solutions_TextInput.text)

        self.reset_board_text()

        self.population_1D_vector = numpy.zeros(shape=(self.num_solutions,8))  # Each solution is represented as a row in this array. If there are 5 rows, then there are 5 solutions.

        # Creating the initial population RANDOMLY as a set of 1D vectors.
        for solution_idx in range(self.num_solutions):
            initial_queens_y_indices = numpy.random.rand(8) * 8
            initial_queens_y_indices = initial_queens_y_indices.astype(numpy.uint8)
            self.population_1D_vector[solution_idx, :] = initial_queens_y_indices

        self.vector_to_matrix()

        self.pop_created = 1  # indicates that the initial population is created in order to enable drawing solutions on GUI.
        self.num_attacks_Label.text = "Initial Population Created."

    def vector_to_matrix(self):
        # Converts the 1D vector solutions into a 2D matrix solutions representing the board, where 1 means a queen exists. The matrix form of the solutions makes calculating the fitness value much easier.

        self.population = numpy.zeros(shape=(self.num_solutions, 8, 8))

        solution_idx = 0
        for current_solution in self.population_1D_vector:
            current_solution = numpy.uint8(current_solution)
            row_idx = 0
            for col_idx in current_solution:
                self.population[solution_idx, row_idx, col_idx] = 1
                row_idx = row_idx + 1
            solution_idx = solution_idx + 1

    def reset_board_text(self):
        # Reset board on GUI.
        for row_idx in range(self.all_widgets.shape[0]):
            for col_idx in range(self.all_widgets.shape[1]):
                self.all_widgets[row_idx, col_idx].text = "[color=ffffff]" + str(row_idx) + ", " + str(
                    col_idx) + "[/color]"
                with self.all_widgets[row_idx, col_idx].canvas.before:
                    kivy.graphics.Color(0, 0, 0, 1)  # green; colors range from 0-1 not 0-255
                    self.rect = kivy.graphics.Rectangle(size=self.all_widgets[row_idx, col_idx].size,
                                                        pos=self.all_widgets[row_idx, col_idx].pos)

    def update_board_UI(self, *args):
        if (self.pop_created == 0):
            print("No Population Created Yet. Create the initial Population by Pressing the \"Initial Population\" Button in Order to Call the f() Method At First.")
            self.num_attacks_Label.text = "Press \"Initial Population\""
            return

        self.reset_board_text()

        population_fitness, total_num_attacks = self.fitness(self.population)

        max_fitness = numpy.max(population_fitness)
        max_fitness_idx = numpy.where(population_fitness == max_fitness)[0][0]
        best_solution = self.population[max_fitness_idx, :]

        self.num_attacks_Label.text = "Max Fitness = " + str(numpy.round(max_fitness, 4)) + "\n# Attacks = " + str(
            total_num_attacks[max_fitness_idx])

        for row_idx in range(8):
            for col_idx in range(8):
                if (best_solution[row_idx, col_idx] == 1):
                    self.all_widgets[row_idx, col_idx].text = "[color=22ff22]Queen[/color]"
                    with self.all_widgets[row_idx, col_idx].canvas.before:
                        kivy.graphics.Color(0, 1, 0, 1)  # green; colors range from 0-1 not 0-255
                        self.rect = kivy.graphics.Rectangle(size=self.all_widgets[row_idx, col_idx].size,
                                                            pos=self.all_widgets[row_idx, col_idx].pos)

    def fitness(self, population):
        total_num_attacks_column = self.attacks_column(self.population)

        total_num_attacks_diagonal = self.attacks_diagonal(self.population)

        total_num_attacks = total_num_attacks_column + total_num_attacks_diagonal

        # GA fitness is increasing (higher value is favorable) but the total number of attacks (total_num_attacks) is decreasing. An increasing fitness value could be created by dividing 1.0 by the number of attacks. For example, if the number of attacks is 5.0, then the fitness is 1.0/5.0=0.2
        population_fitness = numpy.copy(
            total_num_attacks)  # Direct assignment makes both variables refer to the same array. Use numpy.copy() for creating a new independent copy.

        for solution_idx in range(population.shape[0]):
            if population_fitness[solution_idx] == 0:
                population_fitness[solution_idx] = float("inf")
            else:
                population_fitness[solution_idx] = 1.0 / population_fitness[solution_idx]

        return population_fitness, total_num_attacks

    def attacks_diagonal(self, population):
        # For a given queen, how many queens sharing the same column? This is how the fitness value is calculated.

        total_num_attacks = numpy.zeros(population.shape[0])  # Number of attacks for all solutions (diagonal only).

        for solution_idx in range(population.shape[0]):
            ga_solution = population[solution_idx, :]

            # Padding zeros around the solution board for being able to index the boundaries (leftmost/rightmost columns & top/bottom rows). # This is by adding 2 rows (1 at the top and another at the bottom) and adding 2 columns (1 left and another right).
            temp = numpy.zeros(shape=(10, 10))
            # Copying the solution board inside the badded array.
            temp[1:9, 1:9] = ga_solution

            # Returning the indices (rows and columns) of the 8 queeens.
            row_indices, col_indices = numpy.where(ga_solution == 1)
            # Adding 1 to the indices because the badded array is 1 row/column far from the original array.
            row_indices = row_indices + 1
            col_indices = col_indices + 1

            total = 0  # total number of attacking pairs diagonally for each solution.

            for element_idx in range(8):
                x = row_indices[element_idx]
                y = col_indices[element_idx]

                mat_bottom_right = temp[x:, y:]
                total = total + self.diagonal_attacks(mat_bottom_right)

                mat_bottom_left = temp[x:, y:0:-1]
                total = total + self.diagonal_attacks(mat_bottom_left)

                mat_top_right = temp[x:0:-1, y:]
                total = total + self.diagonal_attacks(mat_top_right)

                mat_top_left = temp[x:0:-1, y:0:-1]
                total = total + self.diagonal_attacks(mat_top_left)

            # Dividing the total by 2 because it counts the solution as attacking itself diagonally.
            total_num_attacks[solution_idx] = total_num_attacks[solution_idx] + total / 2

        return total_num_attacks

    def diagonal_attacks(self, mat):
        if (mat.shape[0] < 2 or mat.shape[1] < 2):
            # print("LESS than 2x2.")
            return 0
        num_attacks = mat.diagonal(). - 1
        return num_attacks

    def attacks_column(self, population):
        # For a given queen, how many queens sharing the same coulmn? This is how the fitness value is calculated.
        total_num_attacks = numpy.zeros(population.shape[0])  # Number of attacks for all solutions (column only).

        for solution_idx in range(population.shape[0]):
            ga_solution = population[solution_idx, :]

            for queen_y_pos in range(8):
                # Vertical
                col_sum = numpy.sum(ga_solution[:, queen_y_pos])
                if (col_sum == 0 or col_sum == 1):
                    col_sum = 0
                else:
                    col_sum = col_sum - 1  # To avoid regarding a queen attacking itself.

                total_num_attacks[solution_idx] = total_num_attacks[solution_idx] + col_sum

        return total_num_attacks

    def build(self):
        boxLayout = kivy.uix.boxlayout.BoxLayout(orientation="vertical")

        gridLayout = kivy.uix.gridlayout.GridLayout(rows=8, size_hint_y=9)
        boxLayout_buttons = kivy.uix.boxlayout.BoxLayout(orientation="horizontal")

        boxLayout.add_widget(gridLayout)
        boxLayout.add_widget(boxLayout_buttons)

        # Preparing the 8x8 board.
        self.all_widgets = numpy.zeros(shape=(8, 8), dtype="O")

        for row_idx in range(self.all_widgets.shape[0]):
            for col_idx in range(self.all_widgets.shape[1]):
                self.all_widgets[row_idx, col_idx] = kivy.uix.button.Button(text=str(row_idx) + ", " + str(col_idx),
                                                                            font_size=25)
                self.all_widgets[row_idx, col_idx].markup = True
                gridLayout.add_widget(self.all_widgets[row_idx, col_idx])

        # Preparing buttons inside the child BoxLayout.
        initial_button = kivy.uix.button.Button(text="Initial Population", font_size=15, size_hint_x=2)
        initial_button.bind(on_press=self.initialize_population)

        ga_solution_button = kivy.uix.button.Button(text="Show Best Solution", font_size=15, size_hint_x=2)
        ga_solution_button.bind(on_press=self.update_board_UI)

        start_ga_button = kivy.uix.button.Button(text="Start GA", font_size=15, size_hint_x=2)

        self.num_solutions_TextInput = kivy.uix.textinput.TextInput(text="8", font_size=20, size_hint_x=1)
        self.num_generations_TextInput = kivy.uix.textinput.TextInput(text="10000", font_size=20, size_hint_x=1)
        self.num_mutations_TextInput = kivy.uix.textinput.TextInput(text="5", font_size=20, size_hint_x=1)

        self.num_attacks_Label = kivy.uix.label.Label(text="# Attacks/Best Solution", font_size=15, size_hint_x=2)

        boxLayout_buttons.add_widget(initial_button)
        boxLayout_buttons.add_widget(ga_solution_button)
        boxLayout_buttons.add_widget(start_ga_button)
        boxLayout_buttons.add_widget(self.num_solutions_TextInput)
        boxLayout_buttons.add_widget(self.num_generations_TextInput)
        boxLayout_buttons.add_widget(self.num_mutations_TextInput)
        boxLayout_buttons.add_widget(self.num_attacks_Label)

        return boxLayout

Config.set('graphics', 'width', '1000')
Config.set('graphics', 'height', '600')

buzzleApp = BuzzleApp()
buzzleApp.run()