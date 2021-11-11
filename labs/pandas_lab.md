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

```