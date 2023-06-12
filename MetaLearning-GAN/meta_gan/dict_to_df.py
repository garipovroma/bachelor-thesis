import pandas as pd

accuracy_with_borders = {'DecisionTreeClassifier': '0.18628990650177002 ± 0.0', 'KNeighborsClassifier': '0.2857142984867096 ± 0.0', 'SVC': '0.228152796626091 ± 0.0', 'LogisticRegression': '0.1737310290336609 ± 0.0', 'SimpleMetaNeuralNetwork': '0.3611721694469452 ± 0.05561577044106114', 'deepset': '0.5780220031738281 ± 0.0068250628864325175', 'deepset_without_meta': '0.438513845205307 ± 0.007953953550496462', 'deepset_v6': '0.577184796333313 ± 0.0023865456713500982', 'deepset_v7': '0.5710099339485168 ± 0.008723165125880448', 'deepset_v7_without_meta': '0.44175824522972107 ± 0.01990887053448875', 'deepset_flattened': '0.5757194757461548 ± 0.008911983787996309', 'deepset_flattened_without_meta': '0.4185243248939514 ± 0.010298606807061964', 'lm_gan_discriminator': '0.5278213620185852 ± 0.0055159176356618395', 'lm_gan_discriminator_without_meta': '0.4871794879436493 ± 0.0'}
df = pd.DataFrame.from_dict(accuracy_with_borders, orient='index', columns=['accuracy'])\
    # .sort_values(by='accuracy')

approach = ['Классический'] * 5
approach += ['LM-GAN'] * 2
approach +=


print(df)