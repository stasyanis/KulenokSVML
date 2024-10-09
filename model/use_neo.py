import numpy as np
from model.neo import SingleNeuron

# Загрузка весов из файла и тестирование
new_neuron = SingleNeuron(input_size=8)
new_neuron.load_weights('model/neo_weights.txt')

# Пример использования
test_data = np.array([[0, 28.0, 0, 0, 4, 27.32, 5.7, 158]])
predictions = new_neuron.forward(test_data)
print("Предсказанные значения:", predictions, *np.where(predictions >= 0.5, 'Диабета нет', 'Диабет есть'))