from JordanNeuralNetwork import JordanNeuralNetwork


OUTPUT_LAYER_SIZE = 1


def run():
    print("Neural network initialization.../")
    alpha = float(input("Enter alpha: "))
    input_layer_size = int(input("Enter input layer size: "))
    hidden_layer_size = int(input("Hidden layer size: "))
    max_learning_steps = int(input("Max amount of learning steps: "))
    neural_network = JordanNeuralNetwork(alpha,
                                         max_learning_steps,
                                         input_layer_size,
                                         hidden_layer_size,
                                         OUTPUT_LAYER_SIZE)
    while True:
        choice = input("1)Learning, 2)Process, Something)Break: ")
        if choice == "1":
            input_data = [int(item) for item in input("Enter sequence for learning: ").split()]
            excellent_example = int(input("Enter excellent example for sequence: "))
            expected_error = float(input("Enter expected error for learning: "))
            neural_network.learning(input_data, excellent_example, expected_error)
        elif choice == "2":
            input_data = [int(item) for item in input("Enter sequence for process: ").split()]
            print(f"Predicted result is {neural_network.process(input_data)}")
        else:
            break


if __name__ == "__main__":
    run()
