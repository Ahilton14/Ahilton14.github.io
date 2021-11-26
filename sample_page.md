# Prediciendo si los clientes abandonan las tarjetas de crédito 💳 

## 1 Descripción del Proyecto

### 1.1 Contexto del problema

Un gerente de banco se siente incómodo con que cada vez más clientes abandonen sus servicios de tarjeta de crédito. Realmente agradecerían que alguien pudiera predecir quién se verá afectado para poder acudir de manera proactiva al cliente para brindarle mejores servicios y cambiar las decisiones del cliente en la dirección opuesta.

### 1.2 El objetivo

Este proyecto se lleva a cabo en una secuencia de pasos, el primero de los cuales consiste en un análisis exploratorio, donde el objetivo es conocer el comportamiento de las variables y analizar atributos que indican una fuerte relación con la cancelación de los clientes del servicio de tarjetas de crédito. Tras la segunda parte, que consiste en aplicar técnicas de ingeniería de recursos, el tercer acto consiste en aplicar un algoritmo de aprendizaje automático para encontrar los mejores recursos para construir el modelo. Al final del proyecto, una vez finalizados todos los pasos, se desarrollará un modelo de aprendizaje automático, capaz de predecir, en base a los datos de un sistema, si un cliente dejará el servicio de tarjeta de crédito o no.

### 1.3 Conjunto de datos

Este conjunto de datos consta de 10,127 clientes que mencionan su edad, salario, estado civil, límite de tarjeta de crédito, categoría de tarjeta de crédito, etc.
Solo tenemos un 16,07% de clientes que han cancelado. Por lo tanto, es un poco difícil entrenar nuestro modelo para predecir la rotación de clientes.

#### variables más representativas:
+ **Attrition Flag:** Esta es nuestra variable objetivo, significa si nuestro cliente decidió dejar la organización o si existe una alta probabilidad de que el cliente se vaya.
+ **Gender:** Masculino o femenino.
+ **Customer age:** Edad del cliente
+ **Income category:** A qué categoría de ingresos pertenece el cliente.
+ **Card category:** Categoría de tarjeta que tiene el cliente.
+ **Months Inactive:** Meses de inactividad de usar la tarjeta de crédito.
+ **Credit Limit:** Límite de crédito que el cliente tiene actualmente.
+ **Total Revolving Balance:** La parte no pagada que se transfiere al mes siguiente cuando un cliente no paga.
+ **Average Utilization Ratio:** Mide la cantidad de crédito que está usando en comparación con la cantidad que tiene disponible.
+ **Open to buy:** La cantidad de crédito disponible en un momento dado en la cuenta del titular de una tarjeta de crédito. Por lo tanto, el promedio de apertura para comprar es el crédito promedio disponible asignado a un cliente específico.



```R
#Cargamos las librerias a usar en este proyecto
library(randomForest)
library(highcharter)
library(billboarder)
library(stringr)
library(ggplot2)
library(xgboost)
library(viridis)
library(timetk)
library(tidyr)
library(MASS)
library(dplyr)
library(caret)
library(DMwR)
library(ROSE)
library(glue)
library(pROC)
library(DT)
```


```R
#Cargando el conjunto de datos
BankChurners <- read.csv("./BankChurners.csv")
```


```R
# Eliminar columnas que no se utilizarán durante el análisis
BankChurners <- select(BankChurners, -c("CLIENTNUM", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1", "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"))

#Observamos las primeras 6 filas de la variable
head(BankChurners)
```


<table class="dataframe">
<caption>A data.frame: 6 × 20</caption>
<thead>
	<tr><th></th><th scope=col>Attrition_Flag</th><th scope=col>Customer_Age</th><th scope=col>Gender</th><th scope=col>Dependent_count</th><th scope=col>Education_Level</th><th scope=col>Marital_Status</th><th scope=col>Income_Category</th><th scope=col>Card_Category</th><th scope=col>Months_on_book</th><th scope=col>Total_Relationship_Count</th><th scope=col>Months_Inactive_12_mon</th><th scope=col>Contacts_Count_12_mon</th><th scope=col>Credit_Limit</th><th scope=col>Total_Revolving_Bal</th><th scope=col>Avg_Open_To_Buy</th><th scope=col>Total_Amt_Chng_Q4_Q1</th><th scope=col>Total_Trans_Amt</th><th scope=col>Total_Trans_Ct</th><th scope=col>Total_Ct_Chng_Q4_Q1</th><th scope=col>Avg_Utilization_Ratio</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>Existing Customer</td><td>45</td><td>M</td><td>3</td><td>High School</td><td>Married</td><td>$60K - $80K   </td><td>Blue</td><td>39</td><td>5</td><td>1</td><td>3</td><td>12691</td><td> 777</td><td>11914</td><td>1.335</td><td>1144</td><td>42</td><td>1.625</td><td>0.061</td></tr>
	<tr><th scope=row>2</th><td>Existing Customer</td><td>49</td><td>F</td><td>5</td><td>Graduate   </td><td>Single </td><td>Less than $40K</td><td>Blue</td><td>44</td><td>6</td><td>1</td><td>2</td><td> 8256</td><td> 864</td><td> 7392</td><td>1.541</td><td>1291</td><td>33</td><td>3.714</td><td>0.105</td></tr>
	<tr><th scope=row>3</th><td>Existing Customer</td><td>51</td><td>M</td><td>3</td><td>Graduate   </td><td>Married</td><td>$80K - $120K  </td><td>Blue</td><td>36</td><td>4</td><td>1</td><td>0</td><td> 3418</td><td>   0</td><td> 3418</td><td>2.594</td><td>1887</td><td>20</td><td>2.333</td><td>0.000</td></tr>
	<tr><th scope=row>4</th><td>Existing Customer</td><td>40</td><td>F</td><td>4</td><td>High School</td><td>Unknown</td><td>Less than $40K</td><td>Blue</td><td>34</td><td>3</td><td>4</td><td>1</td><td> 3313</td><td>2517</td><td>  796</td><td>1.405</td><td>1171</td><td>20</td><td>2.333</td><td>0.760</td></tr>
	<tr><th scope=row>5</th><td>Existing Customer</td><td>40</td><td>M</td><td>3</td><td>Uneducated </td><td>Married</td><td>$60K - $80K   </td><td>Blue</td><td>21</td><td>5</td><td>1</td><td>0</td><td> 4716</td><td>   0</td><td> 4716</td><td>2.175</td><td> 816</td><td>28</td><td>2.500</td><td>0.000</td></tr>
	<tr><th scope=row>6</th><td>Existing Customer</td><td>44</td><td>M</td><td>2</td><td>Graduate   </td><td>Married</td><td>$40K - $60K   </td><td>Blue</td><td>36</td><td>3</td><td>1</td><td>2</td><td> 4010</td><td>1247</td><td> 2763</td><td>1.376</td><td>1088</td><td>24</td><td>0.846</td><td>0.311</td></tr>
</tbody>
</table>




```R
#Asignamos a otra variable para ver la correlación de todas las variables
BankChurners_cor <- BankChurners
