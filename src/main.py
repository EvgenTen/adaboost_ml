from data import Iris, Temperature
from adaboost import Adaboost
import numpy as np
NUMBER_ITERATION=100

total_train_i = np.zeros(8)
total_test_i = np.zeros(8)
total_train_t = np.zeros(8)
total_test_t = np.zeros(8)


for num in range(NUMBER_ITERATION):
    Temperature().mix_train()
    total_train_t = np.add(total_train_t, Adaboost(Temperature().rules, Temperature().train_data).get_H(Temperature().train_data))
    total_test_t = np.add(total_test_t,  Adaboost(Temperature().rules, Temperature().train_data).get_H(Temperature().test_data))

    Iris().mix_train()
    total_train_i = np.add(total_train_i, Adaboost(Iris().rules, Iris().train_data).get_H(Iris().train_data))
    total_test_i = np.add(total_test_i, Adaboost(Iris().rules, Iris().train_data).get_H(Iris().test_data))


for i in range(8):

    print('################################# {0} #####################################'.format('Iris error'))

    print('Iris train error: {0}'.format((total_train_i / 100)[i]))
    print('Iris test error: {0}'.format((total_test_i / 100)[i]))

    print('################################################################################################')
print(' ')
for i in range(8):

    print('############################################ {0} #############################'.format('Temperature error'))

    print('Temperature train error: {0}'.format((total_train_t / 100)[i]))
    print('Temperature test error: {0}'.format((total_test_t / 100)[i]))

    print('##########################################################################################################')

