# Pandas review lab

1. **Votre environnement contient le fichier age.csv dans le dossier courant.**

   **1. A l'aide de la fonction read_csv de pandas, charger le dataframe et le stocker dans la variable df**

   **1. Afficher les informations générales à l'aide de la méthode describe du dataframe**

```python
import pandas as pd

df = pd.read_csv('age.csv')
df.describe()
```

2. **Votre environnement contient le fichier age.csv dans le dossier courant.**

   **1. A l'aide de la fonction read_csv de pandas, charger le dataframe et le stocker dans la variable df**

   **1. Selectionner la colonne 'age' du dataframe et la stocker dans la colonne age_col**

```python
import pandas as pd

df = pd.read_csv('age.csv')
age_col = df['age']
```

3. **Votre environnement contient le fichier age.csv dans le dossier courant. Il contient une colonne age.**

   **1. Charger le fichier dans un dataframe.**

   **2. Calculer l'age moyen et le stocker dans la variable average_age.**

```python
import pandas as pd

df = pd.read_csv('age.csv')
average_age = df.mean()[0]
```

4. **Expliquer rapidement (mais clairement) la difference entre un objet pandas.DataFrame et pandas.Series**

Un objet pandas.Series est un vecteur tandis qu'un objet pandas.Dataframe est une matrice

5. **Votre environnement contient le fichier age.csv. Le csv contient une colonne age. Charger le fichier data et stocker dans un nouveau dataframe over_25 les lignes dont l'age est supérieur ou égale à 25.**

```python
import pandas as pd

over_25 = pd.read_csv('age.csv')
over_25 = over_25[over_25['age'] >= 25]
```

6. **On a mis dans votre environnement le fichier  age_vs_salary.csv. Les colonnes du fichiers sont Age et Salary. Charger le fichier à l'aide de pandas et récupérer toutes les lignes dont l'age est inferieur ou égal à 20  et le salaire superieur ou égal à 2000. Vous stockerez ces lignes dans la variables selected_lines**

```python
import pandas as pd

df = pd.read_csv('age_vs_salary.csv')
selected_lines = df[(df['Age'] <= 20) & (df['Salary'] >= 2000)]
```

7. **Vous arrivez dans une petite agence immobilière travaillant à l'ancienne qui utilise encore des fichiers à plat pour garder l'historique de ses biens. Une famille nombreuse est cliente de cette dernière et vous demande une maison avec au moins trois chambres. Elle dispose d'un budget de 350000 €. Charger le dataset house_prices.csv à l'aide de pandas. Le dataset contient les colonnes suivantes : 'size', 'nb_rooms', 'price'. A l'aide d'un mask récupérer toutes les lignes du dataframe correspondant à la demande de la famille. Stocker le resultat dans la variable big_family_houses**

```python
import pandas as pd
import numpy as np

df = pd.read_csv('house_prices.csv')
m = (df['nb_rooms'] < 3) | (df['price'] > 350000)
df = df.mask(m, None)
df['price'] = df['price'].astype(np.float)
df['size'] = df['size'].astype(np.float)
big_family_houses = df.dropna()
```

8. **Charger le fichier house_size_bedrooms_orientation_garden.csv qui contient des données à propos de maisons et le stocker dans la variable df. Ce fichier contient les colonnes ['size', 'bedrooms', 'orientation', 'garden' ]. La colonne garden prend comme valeurs possibles 0 ou 1. La colonne orientation prend comme valeurs possibles : Nord, Sud, Est, Ouest. A l'aide de np.log, créer une nouvelle colonne log_size qui contient log(1 + size). Ajouter au dataframe  df,  la colonne south_garden. Elle doit valoir 1 si la maison est orientée Sud et a un jardin et 0 sinon.**

```python
import pandas as pd
import numpy as np

df = pd.read_csv('house_size_bedrooms_orientation_garden.csv')
df['log_size'] = df['size'].apply(lambda x: np.log(1 + x))
df['south_garden'] = (df['orientation'] == 'Sud').astype(np.int)
```

9. **Vous travaillez dans une banque et vous devez analyser le fichier de transaction ci-dessous.** 

   1. **Charger le fichier 'transactions.csv' dans le dataframe df. Le fichier contient les colonnes suivantes: 'account_sender_name', 'country_sender', 'account_receiver_name', 'country_receiver', 'datetime_timestamp', 'amount'.**

   2. **A l'aide d'un group by, calculer la somme des montants reçues par personne et stocker le resultat dans la variable outputs. Il faudra garder uniquement la colonne amount après le groupby**

```python
import pandas as pd

df = pd.read_csv('transactions.csv')
outputs = df.groupby(['account_sender_name']).sum()
```

10. **Charger le dataset tumor_data.csv dans la variable df. La première colonne est la colonne d'index. Il faudra spécifier le paramètre index_col lors du chargement avec read_csv. Les deux autres colonnes sont: size et p53 concentration**

    - **Selectionner les 100 premières lignes du dataset df et stocker le résultat dans la variable df_100**

    - **A l'aide d'un mask sélectionner les lignes du dataframe complet où la taille de la tumeur est plus grande ou égale à 0.01 et la concentration en marqueur p53 est plus grande ou égale à 0.01. Stocker le résultat dans la variable selected_lines**

```python
import pandas as pd
import numpy as np

df = pd.read_csv('tumor_data.csv', index_col=0)
df_100 = df.head(100)
m = (df['size'] < 0.01) | (df['p53 concentration'] < 0.01)
df = df.astype(np.float).mask(m, None)
selected_lines = df.dropna()
```